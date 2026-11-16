import Foundation
import Combine
import KwipMLCore

/// Independent file authority. Window lifetime and parent workflows do not own its pending edits.
@MainActor final class FlowDocumentOwner: ObservableObject {
    let url: URL
    let sessionID = UUID()
    @Published private(set) var document: FlowDocument
    @Published private(set) var error: String?
    @Published var referenceNotice: String?
    @Published private(set) var recoveryNotice: String?
    @Published private(set) var recoveryCandidates: [FlowDocumentRecoveryRecord] = []
    private var recoveryRecord: FlowDocumentRecoveryRecord?
    private var recoveryAncestors: [FlowDocumentRecoveryRecord] = []
    private var recoverySessionID = UUID()
    private var saved: FlowDocument
    private var undoStates: [FlowDocument] = []
    private var redoStates: [FlowDocument] = []
    var isDirty: Bool { document != saved }
    var canUndo: Bool { !undoStates.isEmpty }
    var canRedo: Bool { !redoStates.isEmpty }

    init(url: URL) throws {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              let size = values.fileSize, size <= FlowDocumentCodec.maximumBytes else { throw FlowDocumentError.invalidDocument }
        self.url = url.resolvingSymlinksInPath()
        let loaded = try FlowDocumentCodec.decode(Data(contentsOf: url))
        document = loaded; saved = loaded
        let records = try FlowDocumentRecoveryStore.records(beside: self.url, documentID: loaded.id)
        recoveryCandidates = records
        let pin = try FlowDocumentCodec.reference(loaded)
        let matching = records.filter { $0.base == pin }
        if matching.count == 1, let candidate = matching.first {
            try adoptRecovery(candidate)
        } else if !records.isEmpty {
            recoveryNotice = "Recovered editing sessions are retained. Choose a matching session to restore; changed-base sessions need separate recovery."
        }
    }

    @discardableResult func edit(_ mutation: (inout FlowDocument) throws -> Void) -> Bool {
        do {
            var next = document
            try mutation(&next)
            guard next.id == document.id, next.revision == document.revision,
                  sameKind(next, document) else { throw FlowDocumentError.invalidDocument }
            _ = try FlowDocumentCodec.encode(next)
            guard next != document else { error = nil; return true }
            try persistRecovery(next)
            undoStates.append(document); trimHistory(&undoStates)
            redoStates.removeAll(); document = next; error = nil; return true
        } catch { self.error = error.localizedDescription; return false }
    }

    @discardableResult func undo() -> Bool {
        guard let previous = undoStates.last else { return false }
        do {
            let next = rebased(previous, revision: saved.revision)
            try persistRecovery(next)
            undoStates.removeLast(); redoStates.append(document); trimHistory(&redoStates)
            document = next; error = nil; return true
        } catch { self.error = error.localizedDescription; return false }
    }
    @discardableResult func redo() -> Bool {
        guard let nextState = redoStates.last else { return false }
        do {
            let next = rebased(nextState, revision: saved.revision)
            try persistRecovery(next)
            redoStates.removeLast(); undoStates.append(document); trimHistory(&undoStates)
            document = next; error = nil; return true
        } catch { self.error = error.localizedDescription; return false }
    }
    @discardableResult func save() -> Bool {
        guard isDirty else { error = nil; return true }
        do {
            guard saved.revision < UInt64.max else { throw FlowDocumentError.resourceLimit }
            let next = rebased(document, revision: saved.revision + 1)
            try FlowDocumentStore.save(next, at: url, expected: FlowDocumentCodec.reference(saved))
            saved = next; document = next; error = nil
            do { try clearRecovery() }
            catch { recoveryNotice = "Document saved; recovery cleanup is pending: " + error.localizedDescription }
            return true
        } catch { self.error = error.localizedDescription; return false }
    }
    @discardableResult func revert() -> Bool {
        do {
            try clearRecovery()
            document = saved; undoStates.removeAll(); redoStates.removeAll(); error = nil; return true
        } catch { self.error = error.localizedDescription; return false }
    }
    @discardableResult func restoreRecovery(_ record: FlowDocumentRecoveryRecord) -> Bool {
        do {
            guard !isDirty else { throw FlowDocumentError.revisionMismatch }
            try adoptRecovery(record); error = nil; return true
        } catch { self.error = error.localizedDescription; return false }
    }
    private func adoptRecovery(_ record: FlowDocumentRecoveryRecord) throws {
        guard record.base == (try FlowDocumentCodec.reference(saved)) else { throw FlowDocumentError.revisionMismatch }
        let restored = try record.document()
        guard sameKind(restored, saved) else { throw FlowDocumentError.invalidDocument }
        let all = try FlowDocumentRecoveryStore.records(beside: url, documentID: saved.id, includeSuperseded: true)
        guard all.contains(record) else { throw FlowDocumentError.revisionMismatch }
        recoveryAncestors = all.filter { record.supersedes.contains($0.sessionID) && $0.base == record.base }
        recoveryRecord = record; recoverySessionID = record.sessionID
        document = restored; undoStates.removeAll(); redoStates.removeAll()
        recoveryNotice = "Restored unsaved edits from an interrupted session. Save or revert this document."
    }
    private func persistRecovery(_ next: FlowDocument) throws {
        if next == saved { try clearRecovery(); return }
        // A save may have succeeded before cleanup failed. Clear the exact old record before reusing its session ID.
        if let record = recoveryRecord, record.base != (try FlowDocumentCodec.reference(saved)) { try clearRecovery() }
        recoveryRecord = try FlowDocumentRecoveryStore.write(next, base: saved, beside: url,
            sessionID: recoverySessionID, supersedes: recoveryRecord?.supersedes ?? [], replacing: recoveryRecord)
    }
    private func clearRecovery() throws {
        for ancestor in recoveryAncestors { try FlowDocumentRecoveryStore.remove(ancestor, beside: url) }
        if let record = recoveryRecord { try FlowDocumentRecoveryStore.remove(record, beside: url) }
        recoveryAncestors.removeAll(); recoveryRecord = nil; recoveryNotice = nil
        recoveryCandidates = try FlowDocumentRecoveryStore.records(beside: url, documentID: saved.id)
        if !recoveryCandidates.isEmpty { recoveryNotice = "Other recovered editing sessions remain available." }
    }

    private func rebased(_ value: FlowDocument, revision: UInt64) -> FlowDocument {
        var result = value; result.revision = revision
        if case .workflow(var workflow) = result.payload { workflow.revision = revision; result.payload = .workflow(workflow) }
        return result
    }
    private func sameKind(_ lhs: FlowDocument, _ rhs: FlowDocument) -> Bool {
        switch (lhs.payload, rhs.payload) {
        case (.workflow, .workflow), (.modelBlock, .modelBlock): return true
        default: return false
        }
    }
    private func trimHistory(_ values: inout [FlowDocument]) {
        var bytes = 0, keep = 0
        for value in values.reversed() {
            let count = (try? FlowDocumentCodec.encode(value).count) ?? FlowDocumentCodec.maximumBytes
            guard keep < 100, bytes + count <= 16_000_000 else { break }
            bytes += count; keep += 1
        }
        if keep < values.count { values.removeFirst(values.count - keep) }
    }
}
