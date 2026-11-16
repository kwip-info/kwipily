import Foundation
import KwipMLCore

extension MeasuredDiagnosticInput {
    /// This bridge accepts the verified PyTorch preparation/training archive,
    /// not live/sampled metrics or a user-authored assertion of completeness.
    /// The current WorkflowPhaseArtifacts training contract is the fixed
    /// two-class, mean CrossEntropyLoss recipe with prepared tensors loaded once
    /// and eval/no-grad validation each epoch. Other backend/loss adapters must
    /// supply a separate reviewed contract rather than reuse this attestation.
    static func generalizationContext(from verified: VerifiedWorkflowOutcome) throws -> MeasuredGeneralizationCheck.VerifiedContext? {
        guard verified.outcome.status == .completed,
              verified.plan.trainingMode == .fresh, verified.capture.checkpointParent == nil,
              let content = verified.modelContent, let partitions = content.partitionEvidence,
              content.capture == verified.capture, content.outcome == verified.outcome,
              verified.outcome.preparationManifestSHA256 != nil,
              content.metrics.count == verified.plan.settings.epochs,
              content.metrics.enumerated().allSatisfy({ $0.element.epoch == $0.offset + 1 }),
              verified.capture.requestIdentity == verified.outcome.requestIdentity else { return nil }
        let partitionFields: [String: String] = [
            "source": partitions.sourceSHA256,
            "trainingMembership": partitions.trainingMembershipSHA256,
            "validationMembership": partitions.validationMembershipSHA256,
            "trainingValues": partitions.trainingDataVersion,
            "validationValues": partitions.validationDataVersion
        ]
        guard partitionFields.values.allSatisfy(WorkflowArchiveCodec.isHash) else { return nil }
        let evaluation: [String: JSONValue] = [
            "contract": .string("kwip.prepared-two-class-mean-cross-entropy.v1"),
            "runtimeEvidence": .string(verified.capture.runtimeEvidence.sha256),
            "modelManifest": .string(verified.outcome.modelManifestSHA256 ?? ""),
            "featureColumns": .array(verified.plan.featureColumns.map(JSONValue.string)),
            "targetColumn": verified.plan.targetColumn.map(JSONValue.string) ?? .null,
            "validationMode": .string("eval-no-grad-fixed-prepared-tensors")
        ]
        return try .init(runID: verified.capture.runID, requestSHA256: verified.capture.requestIdentity,
            partitionSHA256: WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(partitionFields)),
            evaluationSHA256: WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(evaluation)),
            completeEpochCapture: true, unchangedPartitions: true, unchangedEvaluation: true,
            binaryCrossEntropy: true)
    }
}
