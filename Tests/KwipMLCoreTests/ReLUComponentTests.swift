import XCTest
@testable import KwipMLCore

final class ReLUComponentTests: XCTestCase {
    func testAuthoredBundlePreservesIdentityAndAddsSpecificLearning() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "relu"), ReLUComponent.definition)
        let entry = try XCTUnwrap(BundledActivationComponents.entries["relu"])
        XCTAssertEqual(entry.presentation.icon, "relu-v2")
        let descriptor = try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry: registry).first { $0.id == "relu" })
        XCTAssertEqual(descriptor.version, ContractVersion(1,1,0))
        XCTAssertEqual(descriptor.adapterID, "kwip.runner.graph-v2.relu")
        XCTAssertTrue(descriptor.fields.isEmpty)
        XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
        XCTAssertTrue(descriptor.learning.lessonIDs.contains("assessment.activation-relu"))
        XCTAssertTrue(descriptor.learning.lessonIDs.contains("kwip.course.parallel-classifier.v1"))
        for schema in [1,2] {
            XCTAssertNoThrow(try GraphMigration.effectiveConfiguration(of: .init(id: "t", type: "relu"), schemaVersion: schema, registry: registry))
            XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: .init(id: "t", type: "relu", config: ["gain": 2]), schemaVersion: schema, registry: registry))
        }
    }
    func testPreparedExampleKeepsFinalRawClassScoresAndRoundTrips() throws {
        let registry = try ComponentRegistry.builtIn()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let example = PreparedWorkflowExample.component("relu")
        XCTAssertEqual(example.observationsResource, "workflow-observations")
        let document = try example.document(device: "cpu")
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress, modelRegistry: registry, dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
        let graph = plan.model.graph
        XCTAssertEqual(graph.nodes.filter { $0.type == "relu" }.count, 1)
        let output = try XCTUnwrap(graph.nodes.first { $0.type == "output" })
        let finalEdge = try XCTUnwrap(graph.edges.first { $0.target == output.id })
        XCTAssertEqual(graph.nodes.first { $0.id == finalEdge.source }?.type, "linear")
        XCTAssertTrue(GraphValidator.validate(graph, registry: registry, context: .preparedWorkflow).isValid)
        XCTAssertEqual(try JSONDecoder().decode(WorkflowDocument.self, from: JSONEncoder().encode(document)), document)
    }
}
