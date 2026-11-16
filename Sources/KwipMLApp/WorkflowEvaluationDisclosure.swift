import Foundation
import Darwin
import KwipMLCore

/// Metrics leave this gate only after durable local inspection accounting.
/// The journal is advisory: its origin never establishes a previously unseen holdout.
enum WorkflowEvaluationDisclosure {
    struct DisclosedEvaluation {
        let value:WorkflowEvaluationArtifacts.Verified
        let outcome:WorkflowEvaluationOutcome
        /// Includes the current evaluation, but precedes this call's inspection event.
        let priorExposure:WorkflowEvaluationExposureStore.Report
    }
    private struct Failure:LocalizedError {
        let message:String
        var errorDescription:String? {message}
    }
    static func recordEvaluation(_ value:WorkflowEvaluationArtifacts.Verified,root:URL,exposureRoot:URL? = nil)throws {
        try Task.checkCancellation()
        let population=try population(value),store=try exposureStore(root:root,exposureRoot:exposureRoot)
        try store.startTracking()
        try Task.checkCancellation()
        try store.record(event(.evaluated,value,population))
    }
    static func load(_ reference:WorkflowEvaluationHistoryReference,root:URL,exposureRoot:URL? = nil)throws->DisclosedEvaluation {
        try load(reference,root:root,exposureRoot:exposureRoot,verifiedLoader:{WorkflowEvaluationHistoryReader.load($0,root:$1)})
    }
    /// Export retains both source locks independently until its child has exited.
    /// Use the same durable accounting gate without reacquiring those locks.
    static func load(_ reference:WorkflowEvaluationHistoryReference,root:URL,exposureRoot:URL? = nil,inspection:WorkflowEvaluationInspection)throws->DisclosedEvaluation {
        let record=try WorkflowEvaluationHistoryReader.load(reference,inspection:inspection)
        let result=try load(reference,root:root,exposureRoot:exposureRoot,verifiedLoader:{_,_ in record})
        try inspection.check()
        return result
    }
    /// Internal sequencing seam; production callers always use the independent history reader.
    static func load(_ reference:WorkflowEvaluationHistoryReference,root:URL,exposureRoot:URL? = nil,
                     verifiedLoader:(WorkflowEvaluationHistoryReference,URL)->WorkflowEvaluationHistoryLoad)throws->DisclosedEvaluation {
        try Task.checkCancellation()
        let record:VerifiedWorkflowEvaluationRecord
        switch verifiedLoader(reference,root) {
        case .terminal(let value):record=value
        case .inUse:throw Failure(message:"Evaluation is still active. Wait for its worker to exit before inspecting results.")
        case .incomplete(let reason),.unavailable(let reason):throw Failure(message:reason)
        }
        guard !record.captureIncomplete,record.outcome.status == .completed,let value=record.result,
              record.outcome.schemaVersion==1,record.outcome.kind=="frozenEvaluationOutcome",
              record.outcome.projectID==reference.projectID,record.outcome.evaluationID==reference.evaluationID.uuidString,
              value.manifest.projectID==record.outcome.projectID,value.manifest.evaluationID==record.outcome.evaluationID,
              value.manifest.requestIdentity==record.outcome.requestIdentity,
              value.manifestSHA256==record.outcome.manifestSHA256 else {
            throw Failure(message:"Only a complete, verified evaluation matching this history entry can be disclosed.")
        }
        let population=try population(value),store=try exposureStore(root:root,exposureRoot:exposureRoot)
        try Task.checkCancellation()
        try store.startTracking()
        try Task.checkCancellation()
        try store.record(event(.evaluated,value,population))
        try Task.checkCancellation()
        let prior=try store.recordInspectionReturningPrior(event(.inspectionRecorded,value,population))
        // Cancellation after publication conservatively preserves recorded exposure.
        try Task.checkCancellation()
        return .init(value:value,outcome:record.outcome,priorExposure:prior)
    }
    static func recordImported(evaluationID:String,projectID:String,receipt:String,population:WorkflowEvaluationExposureStore.Population,root:URL,exposureRoot:URL?)throws->WorkflowEvaluationExposureStore.Report {
        let store=try exposureStore(root:root,exposureRoot:exposureRoot);try store.startTracking();try Task.checkCancellation()
        try store.record(.init(kind:.evaluated,evaluationID:evaluationID,projectID:projectID,manifestSHA256:receipt,snapshotReceiptSHA256:receipt,population:population))
        return try store.recordInspectionReturningPrior(.init(kind:.inspectionRecorded,evaluationID:evaluationID,projectID:projectID,manifestSHA256:receipt,snapshotReceiptSHA256:receipt,population:population))
    }
    private static func exposureStore(root: URL, exposureRoot: URL?) throws -> WorkflowEvaluationExposureStore {
        if let exposureRoot, exposureRoot.resolvingSymlinksInPath().path != root.resolvingSymlinksInPath().path {
            // Older location-local journals must never disappear from disclosure accounting.
            // Preserve them for explicit reconciliation instead of resetting or guessing a merge.
            let previous = root.appendingPathComponent("workflow-evaluation-exposure")
            var info = stat()
            if lstat(previous.path, &info) == 0 || errno != ENOENT {
                throw Failure(message: "Separate local holdout history requires reconciliation before disclosure. Existing history was retained.")
            }
        }
        return WorkflowEvaluationExposureStore(root: exposureRoot ?? root)
    }
    private static func population(_ value:WorkflowEvaluationArtifacts.Verified)throws->WorkflowEvaluationExposureStore.Population {
        let manifest=value.manifest
        try WorkflowEvaluationComputationCompatibility.validate(version:manifest.schemaVersion,actualDevice:manifest.runtime["device"])
        guard manifest.kind=="frozenEvaluation",manifest.status=="completed",
              UUID(uuidString:manifest.projectID)?.uuidString==manifest.projectID,
              UUID(uuidString:manifest.evaluationID)?.uuidString==manifest.evaluationID,
              manifest.parent.projectID==manifest.projectID,manifest.parent.hasValidPins,
              [manifest.requestIdentity,manifest.requestSHA256,manifest.snapshotReceiptSHA256,value.manifestSHA256].allSatisfy(WorkflowArchiveCodec.isHash),
              value.observationIDs.count==value.summary.evaluatedCount,
              case .string(let source)=manifest.testProvenance["sourceHash"],
              case .object(let preparation)=manifest.testProvenance["preparation"],
              case .string(let membership)=preparation["membershipDigest"] else {
            throw Failure(message:"Verified evaluation population or identity is incomplete.")
        }
        let population=try WorkflowEvaluationExposureStore.Population(checkpointSHA256:manifest.parent.checkpointSHA256,sourceSHA256:source,observationIDs:value.observationIDs)
        guard population.membershipSHA256==membership else {
            throw Failure(message:"Evaluation predictions differ from the verified protected membership.")
        }
        return population
    }
    private static func event(_ kind:WorkflowEvaluationExposureStore.Event.Kind,_ value:WorkflowEvaluationArtifacts.Verified,
                              _ population:WorkflowEvaluationExposureStore.Population)->WorkflowEvaluationExposureStore.Event {
        .init(kind:kind,evaluationID:value.manifest.evaluationID,projectID:value.manifest.projectID,
              manifestSHA256:value.manifestSHA256,snapshotReceiptSHA256:value.manifest.snapshotReceiptSHA256,population:population)
    }
}
