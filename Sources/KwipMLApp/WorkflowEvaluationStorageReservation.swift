import Foundation
import Darwin
import KwipMLCore

/// One admission spans early source capture, private staging, worker execution and
/// finalization. Transfer preserves the original allowance instead of reserving twice.
final class WorkflowEvaluationStorageReservation {
    let projectID: String
    let evaluationID: UUID
    let stageID: UUID
    private let workspace: URL
    private let workspaceIdentity: WorkflowFileIdentity
    private let lock = NSLock()
    private var reservation: WorkflowStorageAdmission.Reservation?

    init(workspace: URL, accountingRoot: URL? = nil, projectID: String, evaluationID: UUID, stageID: UUID, limitBytes: Int64) throws {
        guard UUID(uuidString: projectID)?.uuidString == projectID, limitBytes >= 0 else {
            throw WorkflowStorageAdmission.Failure.invalidScope
        }
        let scopes = try [WorkflowStorageAdmission.Scope(namespace: .evaluationStaging, components: [stageID.uuidString]),
                          WorkflowStorageAdmission.Scope(namespace: .workflowEvaluations, components: [projectID, evaluationID.uuidString])]
        let admission = try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot)
        let pinned = try Self.workspacePin(workspace)
        // Two native snapshot copies (including receipts), existing worker scratch
        // plus output envelope, bounded native protocol/terminal records and
        // the shared 100M disk-table admission scratch ceiling.
        let token = try admission.reserve(scopes: scopes, peakBytes: 1_227_512_000, limitBytes: limitBytes)
        do {
            let current = try Self.workspacePin(workspace)
            guard current.url == pinned.url, current.identity == pinned.identity else { throw WorkflowFileLeaseError.unsafePath }
        } catch { token.close(); throw error }
        self.workspace = pinned.url; self.workspaceIdentity = pinned.identity
        self.projectID = projectID; self.evaluationID = evaluationID; self.stageID = stageID; self.reservation = token
    }

    /// Foreign or stale callers cannot consume or release the original reservation.
    func take(workspace: URL, accountingRoot: URL? = nil, projectID: String, evaluationID: UUID) throws -> WorkflowStorageAdmission.Reservation {
        lock.lock(); defer { lock.unlock() }
        guard let reservation, projectID == self.projectID, evaluationID == self.evaluationID else { throw WorkflowStorageAdmission.Failure.invalidScope }
        let current = try Self.workspacePin(workspace)
        guard current.url == self.workspace, current.identity == workspaceIdentity else { throw WorkflowFileLeaseError.unsafePath }
        self.reservation = nil
        return reservation
    }

    /// Once transferred, only the retained execution token controls its lifetime.
    func close() {
        lock.lock(); defer { lock.unlock() }
        reservation?.close(); reservation = nil
    }
    deinit { close() }

    private static func workspacePin(_ workspace: URL) throws -> (url: URL, identity: WorkflowFileIdentity) {
        guard workspace.isFileURL, !workspace.path.contains("\0"), let pointer = realpath(workspace.path, nil) else {
            throw WorkflowFileLeaseError.unsafePath
        }
        defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer), isDirectory: true)
        let fd = try WorkflowEvaluationOwnership.openDirectory(root)
        defer { Darwin.close(fd) }
        return (root, try WorkflowEvaluationOwnership.directoryIdentity(fd))
    }
}
