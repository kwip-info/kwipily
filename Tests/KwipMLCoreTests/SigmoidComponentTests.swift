import XCTest
@testable import KwipMLCore

final class SigmoidComponentTests: XCTestCase {
    func testBundledIdentityAndBridgeExposeCanonicalParameterlessContract() throws {
        let registry = try ComponentRegistry.builtIn()
        let definition = try XCTUnwrap(registry.component(type: "sigmoid"))
        XCTAssertEqual(definition, SigmoidComponent.definition)
        XCTAssertEqual(definition.inputs, ["in"]); XCTAssertEqual(definition.outputs, ["out"])
        XCTAssertTrue(definition.fields.isEmpty)
        let descriptor = try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry: registry).first { $0.id == "sigmoid" })
        XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
        XCTAssertEqual(descriptor.learning.lessonIDs, ["assessment.activation-sigmoid"])
        XCTAssertEqual(descriptor.adapterID, "kwip.runner.graph-v2.sigmoid")
    }
    func testStrictConfigurationAndShapePreservation() throws {
        let registry = try ComponentRegistry.builtIn()
        var graph = ExampleFlows.geluClassifier()
        let index = try XCTUnwrap(graph.nodes.firstIndex { $0.type == "gelu" })
        graph.nodes[index].type = "sigmoid"; graph.nodes[index].config = [:]
        let validation = GraphValidator.validate(graph, registry: registry)
        XCTAssertTrue(validation.isValid, "\(validation.issues)")
        XCTAssertEqual(validation.tensorLayouts[graph.nodes[index].id], validation.tensorLayouts["features"])
        XCTAssertNoThrow(try GraphMigration.effectiveConfiguration(of: graph.nodes[index], schemaVersion: 2, registry: registry))
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: graph.nodes[index], schemaVersion: 1, registry: registry))
        graph.nodes[index].config = ["temperature": 1]
        XCTAssertFalse(GraphValidator.validate(graph, registry: registry).isValid)
    }
    func testPreparedExampleCompilesAndPersistsWithoutApplyingSigmoidToClassScores() throws {
        let registry = try ComponentRegistry.builtIn()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let example = PreparedWorkflowExample.component("sigmoid")
        XCTAssertEqual(example.observationsResource, "workflow-observations")
        let document = try example.document(device: "cpu")
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: registry, dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
        let graph = plan.model.graph
        XCTAssertEqual(graph.nodes.filter { $0.type == "sigmoid" }.count, 1)
        XCTAssertEqual(plan.settings.device, "cpu")
        let activation = try XCTUnwrap(graph.nodes.first { $0.type == "sigmoid" })
        XCTAssertTrue(activation.config.isEmpty)
        let output = try XCTUnwrap(graph.nodes.first { $0.type == "output" })
        let finalEdge = try XCTUnwrap(graph.edges.first { $0.target == output.id })
        let head = try XCTUnwrap(graph.nodes.first { $0.id == finalEdge.source })
        XCTAssertEqual(head.type, "linear", "The classifier objective requires raw class scores after the hidden activation.")
        let validation = GraphValidator.validate(graph, registry: registry, context: .preparedWorkflow)
        XCTAssertTrue(validation.isValid, "\(validation.issues)")
        XCTAssertEqual(validation.tensorLayouts[activation.id]?.sample.dimensions, [8])
        XCTAssertEqual(validation.tensorLayouts[output.id]?.sample.dimensions, [2])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
        XCTAssertEqual(BundledActivationComponents.entries["sigmoid"]?.learning?.exampleID, "prepared-sigmoid-classifier")
    }

}
