import XCTest
@testable import KwipMLCore

final class LayerNormBundleMetadataTests: XCTestCase {
    func testAllExistingAssessmentsAndBothPreparedScopesHaveMetadataLinksWithoutChangingFieldsOrState() throws {
        let registry = try ComponentRegistry.builtIn()
        let descriptor = try XCTUnwrap(try BuiltInCapabilityBridge.declarations(registry: registry).first { $0.id == "layer_norm" })
        XCTAssertEqual(descriptor.version, ContractVersion())
        XCTAssertEqual(descriptor.adapterID, "kwip.runner.graph-v2.layer_norm")
        XCTAssertEqual(descriptor.fields, LayerNormComponent.definition.fields)
        XCTAssertEqual(descriptor.fields.map(\.key), ["normalized_shape", "eps", "elementwise_affine", "bias"])
        XCTAssertEqual(descriptor.lifecycle.state.map(\.kind), [.parameters])
        XCTAssertEqual(Set(descriptor.learning.lessonIDs), Set(try LayerNormLearningTask.allCases.map { try LayerNormAssessment.plan(task: $0, runtimeVersion: "reviewed-test-runtime", seed: 42).id }))
        XCTAssertEqual(descriptor.learning.exampleIDs, ["kwip.example.layer-norm-rows-classifier.v1", "kwip.example.layer-norm-sample-classifier.v1"])
        XCTAssertEqual(descriptor.ports.map { $0.schema.tensor?.leadingAxes }, [.init(minimum: 2, maximum: 8), .init(minimum: 2, maximum: 8)])
        for whole in [false, true] {
            let example = try PreparedWorkflowExample.layerNorm(normalizeWholeSample: whole).document(device: "cpu")
            let config = try XCTUnwrap(example.nodes.compactMap { node -> [String: JSONValue]? in
                if case .modelPrimitive(let type, _, let config) = node.payload, type == "layer_norm" { return config }; return nil
            }.first)
            XCTAssertEqual(config["normalized_shape"], .array(whole ? [2,3] : [3]))
            XCTAssertEqual(config["elementwise_affine"], .bool(true))
            XCTAssertEqual(config["bias"], .bool(true))
        }
    }
}
