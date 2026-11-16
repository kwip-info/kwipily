import Foundation
import KwipMLCore

/// Archived preview validation is independent of the live project and installed registry.
enum WorkflowPredictionArtifact {
    static func decode(_ data: Data, metadata: JSONValue, runID: String,
                       graph: GraphDocument, evaluationCount: JSONValue?) throws -> PredictionPreview {
        struct Metadata: Decodable { let schemaVersion: Int; let file: String; let sha256: String; let bytes: Int }
        struct Identity: Decodable { let schemaVersion: Int; let runID: String; let selection: String }
        let pin = try JSONDecoder().decode(Metadata.self, from: WorkflowPhaseArtifacts.encode(metadata))
        guard [1, 2, 4].contains(pin.schemaVersion), pin.file == "predictions.json", pin.bytes == data.count,
              data.count <= 1_000_000, pin.sha256 == WorkflowPhaseArtifacts.hash(data) else {
            throw WorkflowPhaseError("Saved prediction preview differs from its manifest hash or size.")
        }
        let identity = try JSONDecoder().decode(Identity.self, from: data)
        guard identity.schemaVersion == pin.schemaVersion, identity.runID == runID,
              identity.selection == "firstRowsOfValidationPartition" else {
            throw WorkflowPhaseError("Saved prediction preview has a different run or sampling policy.")
        }
        if graph.namedInputs != nil {
            let preview=try PredictionPreview.decodeNamed(data,graph:graph)
            guard evaluationCount?.integerValue==preview.evaluationCount else {throw WorkflowPhaseError("Named prediction count differs from the complete held-out population.")}
            return preview
        }
        let preview = try PredictionPreview.decode(data)
        let inputs = graph.nodes.filter { ["input", "index_input"].contains($0.type) }
        guard inputs.count == 1, let source = inputs.first,
              evaluationCount?.integerValue == preview.evaluationCount,
              (preview.captureStatus == .omittedByPolicy ? preview.rows.isEmpty : preview.rows.count == min(200, preview.evaluationCount)) else {
            throw WorkflowPhaseError("Saved prediction preview differs from the evaluation count or input boundary.")
        }
        let expected: TypedTensorContract
        if source.type == "index_input" {
            expected = try IndexInputConfiguration(values: source.config).tensor
        } else {
            guard let width = source.config["features"]?.integerValue, (1...512).contains(width) else {
                throw WorkflowPhaseError("Captured floating input dimensions are invalid.")
            }
            expected = try TypedTensorContract(sample: [Int(width)], dtype: .float32)
        }
        guard preview.dtype == expected.dtype.rawValue, preview.sampleShape == expected.layout.sample.dimensions else {
            throw WorkflowPhaseError("Saved prediction preview differs from the captured input shape or type.")
        }
        return preview
    }
}
