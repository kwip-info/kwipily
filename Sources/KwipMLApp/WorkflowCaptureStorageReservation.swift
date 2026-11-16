import Foundation
import Darwin
import KwipMLCore

/// One admission spans early source capture, private staging, worker execution and
/// finalization. Transfer preserves the original allowance instead of reserving twice.
final class WorkflowCaptureStorageReservation {
    let projectID: String
    let operationID: String
    let purpose: WorkflowPreviewCapture.Purpose
    let stageID: UUID
    let maximumCaptureBytes: Int
    private let reservedWorkerBytes: Int64
    private let workspace: URL
    private let workspaceIdentity: WorkflowFileIdentity
    private let lock = NSLock()
    private var reservation: WorkflowStorageAdmission.Reservation?

    init(workspace: URL, accountingRoot: URL? = nil, projectID: String, operationID: String = UUID().uuidString,
         purpose: WorkflowPreviewCapture.Purpose, stageID: UUID = UUID(), maximumCaptureBytes: Int,
         additionalScopes: [WorkflowStorageAdmission.Scope] = [], additionalBytes: Int64 = 0,
         limitBytes: Int64, workload: ExecutionWorkloadPolicy? = nil) throws {
        guard UUID(uuidString: projectID)?.uuidString == projectID,
              UUID(uuidString: operationID)?.uuidString == operationID,
              (0...purpose.captureBytes(workload)).contains(maximumCaptureBytes), additionalBytes >= 0,
              limitBytes >= 0 else { throw WorkflowStorageAdmission.Failure.invalidScope }
        let workerBytes = try Self.workerGrowth(purpose:purpose,workload:workload)
        reservedWorkerBytes = workerBytes
        /* legacy values remain in workerGrowth */
        var peak = Int64(maximumCaptureBytes)
        for bytes in [workerBytes, 1_000_000, additionalBytes] {
            let (sum, overflow) = peak.addingReportingOverflow(bytes)
            guard !overflow else { throw WorkflowStorageAdmission.Failure.invalidScope }
            peak = sum
        }
        guard let namespace = WorkflowStorageAdmission.Namespace(rawValue: purpose.namespace) else {
            throw WorkflowStorageAdmission.Failure.invalidScope
        }
        let scopes = try [WorkflowStorageAdmission.Scope(namespace: namespace, components: [stageID.uuidString]),
                          WorkflowStorageAdmission.Scope(namespace: .workflowOperations, components: [projectID, operationID])] + additionalScopes
        let admission = try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot)
        let pinned = try Self.workspacePin(workspace)
        let token = try admission.reserve(scopes: scopes, peakBytes: peak, limitBytes: limitBytes)
        do {
            let current = try Self.workspacePin(workspace)
            guard current.url == pinned.url, current.identity == pinned.identity else { throw WorkflowFileLeaseError.unsafePath }
        } catch { token.close(); throw error }
        self.workspace = pinned.url; self.workspaceIdentity = pinned.identity
        self.projectID = projectID; self.operationID = operationID; self.purpose = purpose
        self.stageID = stageID; self.maximumCaptureBytes = maximumCaptureBytes; self.reservation = token
    }

    /// Foreign or stale callers cannot consume or release the original reservation.
    func take(workspace: URL, accountingRoot: URL? = nil, projectID: String, operationID: String,
              purpose: WorkflowPreviewCapture.Purpose, workload: ExecutionWorkloadPolicy? = nil) throws -> WorkflowStorageAdmission.Reservation {
        lock.lock(); defer { lock.unlock() }
        guard let reservation, projectID == self.projectID, operationID == self.operationID,
              purpose == self.purpose else { throw WorkflowStorageAdmission.Failure.invalidScope }
        guard try Self.workerGrowth(purpose:purpose,workload:workload) <= reservedWorkerBytes else {throw WorkflowStorageAdmission.Failure.invalidScope}
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

    static func workerGrowth(purpose: WorkflowPreviewCapture.Purpose, workload: ExecutionWorkloadPolicy?) throws -> Int64 {
        if let workload, purpose.usesPreparationWorkload {
            let (bytes,overflow) = workload.limit(.scratchBytes).addingReportingOverflow(workload.limit(.retainedBytes))
            guard !overflow else {throw WorkflowStorageAdmission.Failure.invalidScope}
            return bytes+2_000_000
        }
        switch purpose {
        case .selfPlay:return 1_000_000_000
        case .gameSession:return 16_000_000
        case .preview:return 33_100_000
        case .backendFit:return 360_100_000
        case .modelTraining:return 443_000_000
        case .preparation:return 302_000_000
        case .preparationInspection:return 100_000_000
        case .sourceInspection:return 0
        case .tableViewQuery:return 200_000_000
        }
    }

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
