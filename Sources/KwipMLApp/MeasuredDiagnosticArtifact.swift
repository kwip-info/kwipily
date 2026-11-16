import Foundation
import KwipMLCore

extension MeasuredDiagnosticInput {
    /// Re-enter the strict wire boundary even for a typed artifact: Codable alone
    /// does not establish identity, cadence, counts or capture-budget validity.
    mutating func installGradients(_ artifact:GradientCaptureArtifact,expectedWorkerRunID:String)throws {
        // Synthesized Codable omits nil optionals; the wire contract requires
        // explicit null norms for non-finite/absent gradients. Restore those
        // keys before strict validation without inventing a numerical value.
        guard var wire=try JSONSerialization.jsonObject(with:JSONEncoder().encode(artifact)) as? [String:Any],
              var summaries=wire["summaries"] as? [[String:Any]] else {throw DiagnosticContractError("Gradient encoding is not a supported artifact.")}
        for index in summaries.indices {
            guard var batches=summaries[index]["batches"] as? [[String:Any]] else {throw DiagnosticContractError("Gradient encoding lacks batch observations.")}
            for batch in batches.indices {
                if batches[batch]["l2Norm"]==nil {batches[batch]["l2Norm"]=NSNull()}
                if batches[batch]["maxAbs"]==nil {batches[batch]["maxAbs"]=NSNull()}
            }
            summaries[index]["batches"]=batches
        }
        wire["summaries"]=summaries
        let checked=try GradientCaptureCodec.decodeArtifact(JSONSerialization.data(withJSONObject:wire),expectedRunID:expectedWorkerRunID,expectedDevice:artifact.device)
        let samples=checked.summaries.flatMap {epoch in epoch.batches.map {batch in
            MeasuredGradientBatch(epoch:epoch.epoch,batch:batch.batchIndex,presentElements:batch.presentElementCount,
                nonfiniteElements:batch.nonfiniteElementCount,zeroElements:batch.zeroElementCount,
                missingParameters:batch.missingParameterCount,l2Norm:batch.l2Norm)
        }}
        var next=self;next.gradients=samples;try next.validate();self=next
    }
    /// Only the verified History reader supplies this bridge. Missing/private
    /// artifacts remain missing; capture budget exhaustion never becomes zero.
    static func from(_ verified:VerifiedWorkflowOutcome)throws->Self {
        var input=Self(runID:verified.capture.runID,requestSHA256:verified.capture.requestIdentity,
                       metrics:verified.modelContent?.metrics ?? [])
        input.workflowSHA256=verified.capture.workflow.sha256
        input.capturedTrain=verified.capture.train
        input.continuation=verified.plan.trainingMode != .fresh || verified.capture.checkpointParent != nil
        input.validationRows=verified.modelContent?.partitionEvidence?.validationCount
        input.rawError=verified.outcome.error
        if let artifact=verified.gradientArtifact {
            guard let worker=verified.outcome.workerRunID else {throw DiagnosticContractError("Verified gradient observations lack a worker identity.")}
            try input.installGradients(artifact,expectedWorkerRunID:worker)
        }
        if case .object(let captures)=verified.modelContent?.activationPreviews {
            for node in verified.plan.model.graph.nodes where ["relu","tanh"].contains(node.type) {
                guard case .object(let capture)=captures[node.id],case .array(let values)=capture["values"] else {continue}
                let numbers=values.compactMap(\.doubleValue)
                guard numbers.count==values.count else {throw DiagnosticContractError("Saved activation sample contains nonnumeric values.")}
                input.activations.append(.init(nodeID:node.id,component:node.type,values:numbers))
            }
        }
        try input.validate();return input
    }
}
