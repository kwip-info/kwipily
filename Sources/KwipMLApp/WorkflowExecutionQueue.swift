import Foundation
import Combine
import Darwin
import KwipMLCore

/// A receipt, not a persisted runnable request. Live requests retain their existing capture owners.
struct WorkflowQueueCapture: Codable, Equatable {
    enum Kind: String, Codable { case training, experiments }
    let projectID: String
    let title: String
    let kind: Kind
    let requestIdentities: [String]
    let documentRevision: UInt64
    let externalLocationID: UUID?
    let sourceHashes: [String]
    let settings: [ExecutionSettings]
    let budget: ResolvedExecutionBudget?
    init(projectID: String, title: String, kind: Kind, requestIdentities: [String], document: WorkflowDocument,
         sourceHashes: [String], settings: [ExecutionSettings] = [], budget: ResolvedExecutionBudget? = nil, externalLocationID: UUID? = nil) throws {
        self.projectID = projectID; self.title = title; self.kind = kind; self.requestIdentities = requestIdentities
        documentRevision = document.revision; self.externalLocationID = externalLocationID; self.sourceHashes = sourceHashes; self.settings = settings; self.budget = budget
        guard document.projectID == projectID else { throw WorkflowArchiveError("Queue capture project differs from its source.") }
        try validate()
    }
    func validate() throws {
        let hash: (String) -> Bool = { $0.utf8.count == 64 && $0.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
        guard UUID(uuidString: projectID) != nil, !title.isEmpty, title.utf8.count <= 1024,
              (1...256).contains(requestIdentities.count), requestIdentities.allSatisfy(hash),
              sourceHashes.count <= 128, sourceHashes.allSatisfy(hash), settings.count <= 256 else {
            throw WorkflowArchiveError("Queue capture has invalid identity or exceeds its supported bounds.")
        }
    }
}

@MainActor final class WorkflowExecutionQueue: ObservableObject {
    enum State: String, Codable { case queued, running, cancelling, completed, failed, cancelled, paused
        var terminal: Bool { [.completed, .failed, .cancelled].contains(self) }
    }
    struct Entry: Codable, Equatable, Identifiable {
        let id: UUID
        let capture: WorkflowQueueCapture
        let createdAt: Date
        var state: State
        var message: String
    }
    private struct Record: Codable { var schemaVersion = 1; var entries: [Entry] }
    private struct Job { let start: () throws -> Void; let cancel: () -> Void }
    private static var sharedQueues: [String: WorkflowExecutionQueue] = [:]
    static func shared(directory: URL) -> WorkflowExecutionQueue {
        // Foundation can spell the same /var directory differently before and after creation.
        // Service identity must use the actual canonical directory, just like its file lease.
        let resolved = (try? resolveDirectory(directory)) ?? directory
        let key = resolved.path
        if let value = sharedQueues[key] { return value }
        let value = WorkflowExecutionQueue(directory: resolved); sharedQueues[key] = value; return value
    }
    @Published private(set) var entries: [Entry] = []
    @Published private(set) var error: String?
    private var directory: URL
    private var lease: WorkflowFileLease?
    private var jobs: [UUID: Job] = [:]
    private var activeID: UUID?
    private var closed = false
    private let parts = ["queue.json"]

    init(directory: URL) {
        self.directory = directory
        do {
            self.directory = try Self.resolveDirectory(directory)
            lease = try WorkflowFileLease(directory: self.directory, name: "owner.lock", mode: .exclusive, create: true)
            if let data = try DiagnosticFileIO.readIfPresent(root: self.directory, parts: parts, limit: 2_000_000) {
                let record = try JSONDecoder().decode(Record.self, from: data)
                guard record.schemaVersion == 1, record.entries.count <= 256,
                      Set(record.entries.map(\.id)).count == record.entries.count else { throw WorkflowArchiveError("Queue receipt version or size is unsupported.") }
                for entry in record.entries { try entry.capture.validate(); guard entry.message.utf8.count <= 4096 else { throw WorkflowArchiveError("Queue receipt message exceeds its bound.") } }
                var restored = record.entries
                for index in restored.indices where !restored[index].state.terminal {
                    restored[index].state = .paused
                    restored[index].message = "Interrupted queue request. Review the original project and capture a new request; saved inputs are not automatically resumed."
                }
                try persist(restored); entries = restored
            }
        } catch { self.error = "Training queue unavailable; no queued worker will start. " + error.localizedDescription }
    }
    private static func resolveDirectory(_ directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var info = stat()
        guard lstat(directory.path, &info) == 0, info.st_mode & S_IFMT == S_IFDIR,
              let canonical = realpath(directory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        return URL(fileURLWithPath: String(cString: canonical))
    }
    @discardableResult func enqueue(capture: WorkflowQueueCapture, start: @escaping () throws -> Void, cancel: @escaping () -> Void) throws -> UUID {
        guard !closed, error == nil, lease != nil else { throw WorkflowArchiveError(error ?? "The training queue is closed.") }
        try capture.validate()
        guard entries.count < 256 else { throw WorkflowArchiveError("Queue receipts are full. Clear finished queue entries; saved run History is unaffected.") }
        let id = UUID()
        var next = entries
        next.append(.init(id: id, capture: capture, createdAt: Date(), state: .queued, message: "Captured and waiting for the training slot."))
        try persist(next); entries = next; jobs[id] = .init(start: start, cancel: cancel)
        RunFeedbackController.shared.begin(id, state: .queued)
        schedule(); return id
    }
    func cancel(_ id: UUID) {
        guard !closed, let index = entries.firstIndex(where: {$0.id == id}), !entries[index].state.terminal else { return }
        if entries[index].state == .running || entries[index].state == .cancelling {
            update(id, state: .cancelling, message: "Cancelling the active captured request…")
            jobs[id]?.cancel()
        } else {
            let job = jobs.removeValue(forKey: id)
            update(id, state: .cancelled, message: "Cancelled before dispatch; no queued worker was launched.")
            job?.cancel(); schedule()
        }
    }
    func finish(_ id: UUID, state: State, message: String) {
        guard !closed, state.terminal, entries.contains(where: {$0.id == id && !$0.state.terminal}) else { return }
        update(id, state: state, message: message)
        jobs.removeValue(forKey: id)
        if activeID == id { activeID = nil }
        schedule()
    }
    func clearFinished() {
        guard !closed, error == nil else { return }
        let next = entries.filter { !$0.state.terminal }
        do { try persist(next); entries = next } catch { self.error = error.localizedDescription }
    }
    /// Used when a service is explicitly retired. Restart never reconstructs runnable closures.
    func close() {
        guard !closed else { return }; closed = true
        let pending = Array(jobs.values)
        for id in jobs.keys { RunFeedbackController.shared.forget(id) }
        jobs.removeAll()
        pending.forEach { $0.cancel() }; lease?.close(); lease = nil
    }
    private func schedule() { Task { [weak self] in self?.dispatchNext() } }
    private func dispatchNext() {
        guard !closed, error == nil, activeID == nil,
              let entry = entries.first(where: {$0.state == .queued}), let job = jobs[entry.id] else { return }
        update(entry.id, state: .running, message: "Running captured request.")
        guard error == nil else { return }
        activeID = entry.id
        do { try job.start() }
        catch { finish(entry.id, state: .failed, message: error.localizedDescription); job.cancel() }
    }
    private func update(_ id: UUID, state: State, message: String) {
        guard let index = entries.firstIndex(where: {$0.id == id}) else { return }
        var next = entries; next[index].state = state; next[index].message = String(message.prefix(1000))
        do { try persist(next); entries = next
            if let feedbackState = RunFeedbackController.State(rawValue: state.rawValue) { RunFeedbackController.shared.transition(id, state: feedbackState) }
        }
        catch {
            self.error = "Queue state could not be saved; further dispatch stopped. " + error.localizedDescription
            // Actual terminal work must leave the live registry even if its queue receipt
            // failed to persist. Report failed feedback, never a durable-success claim.
            if state.terminal { RunFeedbackController.shared.transition(id, state: .failed) }
        }
    }
    private func persist(_ next: [Entry]) throws {
        guard let lease else { throw WorkflowFileLeaseError.closed }; try lease.checkIdentity()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(Record(entries: next))
        guard data.count <= 2_000_000 else { throw WorkflowArchiveError("Queue receipts exceed 2 MB. Clear finished entries before adding work.") }
        try DiagnosticFileIO.writeAtomic(data, root: directory, parts: parts)
        try lease.checkIdentity()
    }
}
