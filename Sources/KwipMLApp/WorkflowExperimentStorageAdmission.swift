import Foundation
import KwipMLCore

/// One group owns its source/control allowance and only the current variant's
/// growth. Prior variants remain charged as retained run archives.
final class WorkflowExperimentStorageAdmission {
    let projectID: String
    let groupID: String
    private let groupDirectory: URL
    private let scope: WorkflowStorageAdmission.Scope
    private let peak: Int64
    private let limit: Int64
    private let variantCount: Int
    private let token: WorkflowStorageAdmission.Reservation
    private var bound = false
    private var admitted = Set<String>()
    private let lock = NSLock()

    init(workspace: URL, accountingRoot: URL? = nil, projectID: String, groupID: String, sourceBytes: Int64,
         variantCount: Int, hasFileCollections: Bool = false, limitBytes: Int64) throws {
        guard (0...Int64(WorkflowCapturedSourceArchive.aggregateLimit(hasCollections: hasFileCollections))).contains(sourceBytes), (1...32).contains(variantCount) else {
            throw WorkflowStorageAdmission.Failure.invalidScope
        }
        let scope = try WorkflowStorageAdmission.Scope(namespace: .workflowExperiments, components: [projectID, groupID])
        let admission = try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot)
        guard let canonical = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let root = URL(fileURLWithPath: String(cString: canonical)); free(canonical)
        // Authoring 2MB; legacy inputs/group receipts <1MB. Collections retain
        // up to2MB of exact member inventory plus the existing group controls.
        // Each of two phases per variant retains a launch, slot and claim (520KB).
        let peak = sourceBytes + (hasFileCollections ? 5_000_000 : 3_000_000) + Int64(variantCount) * 2 * 520_000
        self.token = try admission.reserve(scopes: [scope], peakBytes: peak, limitBytes: limitBytes)
        self.projectID = projectID; self.groupID = groupID; self.scope = scope
        self.groupDirectory = root.appendingPathComponent("workflow-experiments/\(projectID)/\(groupID)")
        self.peak = peak; self.limit = limitBytes; self.variantCount = variantCount
    }
    func bind(to owner: WorkflowGroupOwner) throws {
        lock.lock(); defer { lock.unlock() }
        guard !bound, owner.metadata.groupID == groupID else { throw WorkflowStorageAdmission.Failure.invalidScope }
        try owner.validateDirectory(groupDirectory)
        try token.bind(to: .init(scope: scope, directoryIdentity: owner.metadata.directoryIdentity,
            ownerLockIdentity: owner.metadata.ownerLockIdentity, activityLockIdentity: owner.metadata.activityLockIdentity))
        bound = true
    }
    func admitVariant(directoryID: String, runPeakBytes: Int64) throws {
        lock.lock(); defer { lock.unlock() }
        guard bound, runPeakBytes > 0, admitted.count < variantCount, !admitted.contains(directoryID) else {
            throw WorkflowStorageAdmission.Failure.invalidScope
        }
        let run = try WorkflowStorageAdmission.Scope(namespace: .workflowRuns, components: [projectID, directoryID])
        let (nextPeak, overflow) = peak.addingReportingOverflow(runPeakBytes)
        guard !overflow else { throw WorkflowStorageAdmission.Failure.invalidScope }
        try token.revise(scopes: [scope, run], peakBytes: nextPeak, limitBytes: limit)
        admitted.insert(directoryID)
    }
    func close() { token.close() }
    deinit { close() }
}
