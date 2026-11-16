import XCTest
@testable import KwipMLCore

final class SoftmaxBundleMetadataTests: XCTestCase {
    func testExistingAxisLessonsAndPreparedExampleHaveRealMetadataLinks() throws {
        let registry = try ComponentRegistry.builtIn()
        let definition = try XCTUnwrap(registry.component(type: "softmax"))
        let declaration = try XCTUnwrap(try BuiltInCapabilityBridge.declarations(registry: registry).first { $0.id == "softmax" })
        XCTAssertEqual(declaration.version, ContractVersion())
        XCTAssertEqual(declaration.adapterID, "kwip.runner.graph-v2.softmax")
        XCTAssertEqual(declaration.fields, definition.fields)
        XCTAssertEqual(declaration.fields.map(\.key), ["dim"])
        XCTAssertEqual(declaration.learning.lessonIDs, ["assessment.softmax-last-axis", "assessment.softmax-first-sample-axis"])
        XCTAssertEqual(declaration.learning.exampleIDs, ["kwip.example.softmax-classifier.v1"])
        XCTAssertTrue(declaration.lifecycle.state.isEmpty)
        XCTAssertEqual(declaration.ports.map { $0.schema.tensor?.leadingAxes }, [.init(minimum: 2, maximum: 8), .init(minimum: 2, maximum: 8)])
        let example = try WorkflowExamples.preparedSoftmaxClassifier(device: "cpu")
        let configs = example.nodes.compactMap { node -> [String: JSONValue]? in
            if case .modelPrimitive(let type, _, let config) = node.payload, type == "softmax" { return config }; return nil
        }
        XCTAssertEqual(configs, [["dim": -1]])
        for operation in ["softmax-last-axis", "softmax-first-sample-axis"] {
            let task = try TensorLearningTask.plan(operation: operation, runtimeVersion: "reviewed-test-runtime", seed: 42)
            XCTAssertTrue(declaration.learning.lessonIDs.contains(task.id))
        }
    }
}
