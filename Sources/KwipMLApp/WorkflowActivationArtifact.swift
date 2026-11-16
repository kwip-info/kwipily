import Foundation
import KwipMLCore

/// Bounded saved activation samples; validation requires no installed runtime.
enum WorkflowActivationArtifact {
    static func decode(_ data:Data, metadata:JSONValue, runID:String, graphHash:String,
                       graph:GraphDocument, sampleCapture:TrainingSampleCapturePolicy?, objectiveContext:WorkflowObjectiveArtifactContext? = nil, capturedRegistry:ComponentRegistry? = nil, bagContext:WorkflowBagArtifactContext? = nil) throws -> JSONValue {
        let value = try decodeIdentity(data,metadata:metadata,runID:runID,graphHash:graphHash,sampleCapture:sampleCapture)
        guard case .object(let previews)=value else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowModelTrainingExecution.verifyActivationPreviews(previews,graph:graph,sampleCapture:sampleCapture,objectiveContext:objectiveContext,capturedRegistry:capturedRegistry,bagContext:bagContext)
        return .object(previews)
    }
    /// Identity-only phase for the private inventory verifier. Its public wrapper
    /// validates the graph/tensors after the large inventory frame has returned.
    @inline(never)
    static func decodeIdentity(_ data:Data,metadata:JSONValue,runID:String,graphHash:String,sampleCapture:TrainingSampleCapturePolicy?) throws -> JSONValue {
        guard case .object(let pin) = metadata,
              Set(pin.keys) == ["schemaVersion","file","sha256","bytes"],
              pin["schemaVersion"] == 1, pin["file"] == .string("activations.json"),
              pin["bytes"]?.integerValue == data.count, data.count <= 2_000_000,
              pin["sha256"] == .string(WorkflowPhaseArtifacts.hash(data)) else {
            throw WorkflowPhaseError("Saved activation samples differ from their manifest hash or size.")
        }
        let object = try JSONDecoder().decode([String:JSONValue].self,from:data)
        let expectedPolicy = try sampleCapture.map { try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode($0)) }
        let keys:Set<String> = ["schemaVersion","runID","graphHash","selection","previews"]
        guard Set(object.keys) == (sampleCapture == nil ? keys : keys.union(["sampleCapture"])),
              object["schemaVersion"] == 1, object["runID"] == .string(runID),
              object["graphHash"] == .string(graphHash), graphHash.count == 64,
              object["selection"] == .string("firstValidationRows"),
              object["sampleCapture"] == expectedPolicy,
              case .object(let previews) = object["previews"] else {
            throw WorkflowPhaseError("Saved activation samples differ from the captured run, graph or policy.")
        }
        return .object(previews)
    }

}
