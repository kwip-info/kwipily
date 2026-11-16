import Foundation
import KwipMLCore

/// One ordinary run's archive allowance. Each phase binds its own existing worker
/// authority; readmission occurs only after the preceding worker has really exited.
final class WorkflowExecutionStorageAdmission {
    private let projectID: String
    private let preparationOperationID: String
    private let trainingOperationID: String
    private let limit: Int64
    private let lock = NSLock()
    private var phase: WorkflowLaunchIntent.Phase = .preparation
    private var token: WorkflowStorageAdmission.Reservation?
    private var bound = false

    nonisolated static func peakBytes(sourceByteCounts: [Int], checkpointFiles: [WorkflowArchiveFile], hasFileCollections: Bool = false) throws -> Int64 {
        guard sourceByteCounts.count <= 128 else { throw WorkflowStorageAdmission.Failure.invalidScope }
        var sourceTotal: Int64 = 0
        let aggregate = WorkflowCapturedSourceArchive.aggregateLimit(hasCollections: hasFileCollections)
        let perSource = hasFileCollections ? aggregate : 10_000_000
        for count in sourceByteCounts {
            guard (0...perSource).contains(count) else { throw WorkflowStorageAdmission.Failure.invalidScope }
            sourceTotal += Int64(count)
            guard sourceTotal <= Int64(aggregate) else { throw WorkflowStorageAdmission.Failure.invalidScope }
        }
        var checkpointTotal: Int64 = 0
        if !checkpointFiles.isEmpty {
            let limits = try WorkflowCheckpointInventory.structuralLimits(names:Set(checkpointFiles.map(\.path)))
            guard checkpointFiles.count == limits.count, Set(checkpointFiles.map(\.path)) == Set(limits.keys) else {
                throw WorkflowStorageAdmission.Failure.invalidScope
            }
            for file in checkpointFiles {
                guard let maximum = limits[file.path], (1...maximum).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256) else {
                    throw WorkflowStorageAdmission.Failure.invalidScope
                }
                checkpointTotal += Int64(file.bytes)
            }
        }
        // 202M preparation + 341M training. Native controls: archive14.256M,
        // preparation graph2M/bindings1M, training graph2M/composite2M,
        // terminal256K and two operation envelopes/ownership/claims <600K.
        // 24M leaves bounded metadata headroom; copies of inputs count separately.
        // The phases share a 100M disk-row scratch ceiling and do not overlap.
        return sourceTotal + checkpointTotal + 202_000_000 + 343_000_000 + 100_000_000 + 24_000_000
    }

    init(workspace: URL, accountingRoot: URL? = nil, projectID: String, directoryID: String, preparationOperationID: String,
         trainingOperationID: String, sourceByteCounts: [Int], checkpointFiles: [WorkflowArchiveFile], hasFileCollections: Bool = false, limitBytes: Int64) throws {
        guard preparationOperationID != trainingOperationID else { throw WorkflowStorageAdmission.Failure.invalidScope }
        let scopes = try [WorkflowStorageAdmission.Scope(namespace: .workflowRuns, components: [projectID, directoryID]),
            .init(namespace: .workflowOperations, components: [projectID, preparationOperationID]),
            .init(namespace: .workflowOperations, components: [projectID, trainingOperationID])]
        let peak = try Self.peakBytes(sourceByteCounts: sourceByteCounts, checkpointFiles: checkpointFiles, hasFileCollections: hasFileCollections)
        let admission = try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot)
        let token = try admission.reserve(scopes: scopes, peakBytes: peak, limitBytes: limitBytes)
        self.limit = limitBytes
        self.projectID = projectID; self.preparationOperationID = preparationOperationID
        self.trainingOperationID = trainingOperationID; self.token = token
    }

    func bind(to owner: WorkflowOperationOwner, phase: WorkflowLaunchIntent.Phase) throws {
        lock.lock(); defer { lock.unlock() }
        let expectedID = phase == .preparation ? preparationOperationID : trainingOperationID
        guard let token, !bound, phase == self.phase, owner.metadata.projectID == projectID,
              owner.metadata.operationID == expectedID else { throw WorkflowStorageAdmission.Failure.invalidScope }
        try owner.check()
        try token.bind(to: .init(scope: .init(namespace: .workflowOperations, components: [projectID, expectedID]),
            directoryIdentity: owner.metadata.directoryIdentity, ownerLockIdentity: owner.metadata.ownerLockIdentity,
            activityLockIdentity: owner.metadata.activityLockIdentity))
        bound = true
    }

    func transitionToTraining() throws {
        lock.lock(); defer { lock.unlock() }
        guard phase == .preparation, let token else { throw WorkflowStorageAdmission.Failure.invalidScope }
        try token.prepareNextPhase(limitBytes: limit)
        phase = .training; bound = false
    }
    func close() {
        lock.lock(); defer { lock.unlock() }
        token?.close(); token = nil
    }
    deinit { close() }
}
