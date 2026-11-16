import Foundation
import Combine

protocol WorkflowTrainedExportIO: Actor {
    func stage(_ reference: WorkflowHistoryReference, root: URL) async throws -> WorkflowTrainedExportPackage
    func publish(_ package: WorkflowTrainedExportPackage, to destination: URL) async throws
}
actor WorkflowTrainedExportWriter: WorkflowTrainedExportIO {
    private let accountingRoot: URL?
    private let storageLimitBytes: Int64?
    init(accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil) { self.accountingRoot = accountingRoot; self.storageLimitBytes = storageLimitBytes }
    func stage(_ reference: WorkflowHistoryReference, root: URL) async throws -> WorkflowTrainedExportPackage {
        try Task.checkCancellation()
        let accountingRoot=accountingRoot,limit=storageLimitBytes
        let package = try await WorkflowHistoryVerificationWork.run(discard: {$0.remove()}) {
            try WorkflowHistoryReader.stageTrainedExport(reference, root: root, accountingRoot: accountingRoot, storageLimitBytes: limit)
        }
        do { try Task.checkCancellation(); return package }
        catch { package.remove(); throw error }
    }
    func publish(_ package: WorkflowTrainedExportPackage, to destination: URL) async throws {
        try package.publish(to: destination)
    }
}

/// Owns the reviewed bytes until asynchronous publication has reached its terminal state.
@MainActor final class WorkflowTrainedExportStore: ObservableObject {
    @Published private(set) var package: WorkflowTrainedExportPackage?
    @Published private(set) var isBusy = false
    @Published private(set) var savedURL: URL?
    @Published private(set) var statusMessage = "Prepare a verified package to review before saving."
    private let reference: WorkflowHistoryReference
    private let root: URL
    private let runID: String, requestIdentity: String, modelManifestSHA256: String
    private let reservation: WorkflowExecutionReservation
    private let io: any WorkflowTrainedExportIO
    private let isCurrent: () -> Bool
    private var files: WorkflowExportFileLease?
    private var active = true
    private var cancelled = false
    private var task: Task<Void, Never>?

    init(reference: WorkflowHistoryReference, root: URL, runID: String, requestIdentity: String,
         modelManifestSHA256: String, reservation: WorkflowExecutionReservation,
         io: any WorkflowTrainedExportIO = WorkflowTrainedExportWriter(), isCurrent: @escaping () -> Bool) {
        self.reference = reference; self.root = root; self.runID = runID
        self.requestIdentity = requestIdentity; self.modelManifestSHA256 = modelManifestSHA256
        self.reservation = reservation; self.io = io; self.isCurrent = isCurrent
    }
    @discardableResult func prepare() -> Bool {
        guard active, isCurrent(), !isBusy else { return false }
        guard let token = reservation.acquire() else { statusMessage = "Wait for the current workspace operation."; return false }
        clean(); savedURL = nil; cancelled = false; isBusy = true
        statusMessage = "Verifying saved model and preparing review…"
        task = Task { [self] in
            defer { finish(token) }
            do {
                let staged = try await io.stage(reference, root: root)
                files = WorkflowExportFileLease(package: staged)
                package = staged
                guard active, !cancelled, !Task.isCancelled, isCurrent() else { throw CancellationError() }
                guard staged.reference == reference, staged.runID == runID,
                      staged.requestIdentity == requestIdentity, staged.modelManifestSHA256 == modelManifestSHA256 else {
                    throw WorkflowArchiveError("Export does not match the selected saved run.")
                }
                statusMessage = "Package ready for review. Prepared inputs are required for inference."
            } catch {
                clean()
                statusMessage = error is CancellationError ? "Export preparation cancelled." : "Could not prepare export: " + error.localizedDescription
            }
        }
        return true
    }
    @discardableResult func save(to destination: URL, access: URL? = nil) -> Bool {
        guard active, isCurrent(), !isBusy, let package else { return false }
        guard let token = reservation.acquire() else { statusMessage = "Wait for the current workspace operation."; return false }
        let scoped = access?.startAccessingSecurityScopedResource() ?? false
        cancelled = false; isBusy = true; savedURL = nil
        statusMessage = "Saving verified model package…"
        task = Task { [self] in
            defer {
                if scoped { access?.stopAccessingSecurityScopedResource() }
                if cancelled || !active { clean() }
                finish(token)
            }
            do {
                try await io.publish(package, to: destination)
                // Publication's rename is its commit point. A later cancellation cannot undo success.
                savedURL = destination
                statusMessage = "Model package saved."
            } catch {
                statusMessage = error is CancellationError ? "Export cancelled." : "Could not save export: " + error.localizedDescription
            }
        }
        return true
    }
    func cancel() {
        if isBusy {
            cancelled = true; task?.cancel(); statusMessage = "Cancelling export…"
        } else { clean(); statusMessage = "Export review closed." }
    }
    func invalidate() { active = false; cancel() }
    private func clean() {
        package = nil; files = nil
    }
    private func finish(_ token: UUID) {
        reservation.release(token); task = nil; isBusy = false
    }
}

/// Destruction needs no actor hop; the store retains this lease through active I/O.
private final class WorkflowExportFileLease {
    let package: WorkflowTrainedExportPackage
    init(package: WorkflowTrainedExportPackage) { self.package = package }
    deinit { package.remove() }
}
