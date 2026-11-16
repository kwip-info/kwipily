import XCTest
@testable import KwipMLCore

final class PreparedWorkflowGraphValidationTests: XCTestCase {
    private func graph(width: Int = 3) -> GraphDocument {
        GraphDocument(schemaVersion: 2, nodes: [
            GraphNode(id: "input", type: "input", config: ["features": .integer(Int64(width))]),
            GraphNode(id: "linear", type: "linear", config: ["in_features": .integer(Int64(width)), "out_features": 2, "bias": true]),
            GraphNode(id: "output", type: "output")
        ], edges: [GraphEdge(id: "a", source: "input", target: "linear"), GraphEdge(id: "b", source: "linear", target: "output")])
    }
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func compile(_ document: WorkflowDocument) throws -> WorkflowExecutionPlan {
        try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress, modelRegistry: .builtIn(),
                                     dataRegistry: DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testPreparedContextAcceptsDeclaredWidthWithoutRelaxingLegacyDatasetBoundary() throws {
        let registry = try ComponentRegistry.builtIn(), model = graph()
        XCTAssertFalse(GraphValidator.validate(model, registry: registry).isValid)
        XCTAssertTrue(GraphValidator.validate(model, registry: registry, context: .preparedWorkflow).isValid)
        XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(.legacyFlat(model), registry: registry))
        let snapshot = try ModelExecutionSnapshotBuilder.build(.legacyFlat(model), registry: registry, context: .preparedWorkflow)
        XCTAssertEqual(snapshot.graph, model)
        XCTAssertFalse(BuiltInCapabilityBridge.validate(graph: model, registry: registry, runtimeVersion: "2.14.1").isValid)
        XCTAssertTrue(BuiltInCapabilityBridge.validate(graph: model, registry: registry, runtimeVersion: "2.14.1", context: .preparedWorkflow).isValid)
        XCTAssertEqual(try WorkflowMappingRequirements.inputContract(model, registry: registry).layout.sample.dimensions, [3])
        let two = graph(width: 2)
        XCTAssertTrue(GraphValidator.validate(two, registry: registry).isValid)
        XCTAssertEqual(try ModelExecutionSnapshotBuilder.build(.legacyFlat(two), registry: registry),
                       try ModelExecutionSnapshotBuilder.build(.legacyFlat(two), registry: registry, context: .preparedWorkflow))
    }
    func testPreparedContextPropagatesThroughBlockExpansionAndPreservesStructuralRefusal() throws {
        let registry = try ComponentRegistry.builtIn()
        var source = CompositeExamples.parallelClassifier
        let inputIndex = source.root.nodes.firstIndex { $0.type == "input" }!
        source.root.nodes[inputIndex].config["features"] = 3
        for i in source.definitions[0].body.nodes.indices where source.definitions[0].body.nodes[i].type == "linear" {
            source.definitions[0].body.nodes[i].config["in_features"] = 3
        }
        let document = ModelProjectDocument.blockModel(.init(projectID: UUID().uuidString, revision: 0, source: source))
        XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(document, registry: registry))
        XCTAssertThrowsError(try CompositeExpander.expand(source, registry: registry))
        let snapshot = try ModelExecutionSnapshotBuilder.build(document, registry: registry, context: .preparedWorkflow)
        XCTAssertNotNil(snapshot.provenance)
        XCTAssertTrue(GraphValidator.validate(snapshot.graph, registry: registry, context: .preparedWorkflow).isValid)
        var invalid = graph()
        invalid.nodes[1].config["in_features"] = 2
        XCTAssertFalse(GraphValidator.validate(invalid, registry: registry, context: .preparedWorkflow).isValid)
        XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(.legacyFlat(invalid), registry: registry, context: .preparedWorkflow))
    }
    func testWorkflowCompilationRequiresExactOrderedMappingWidth() throws {
        var document = try WorkflowExamples.preparedClassifier(projectID: UUID().uuidString, device: "cpu", model: graph())
        let mapping = document.nodes.firstIndex { $0.address == .init(domain: .control, localID: "mapping") }!
        document.nodes[mapping].payload = .trainingData(config: ["featureColumns": .array([.string("a"), .string("b"), .string("c")]), "targetColumn": .string("label")])
        let plan = try compile(document)
        XCTAssertEqual(plan.featureColumns, ["a", "b", "c"])
        XCTAssertEqual(plan.model.graph.nodes.first { $0.type == "input" }?.config["features"], 3)
        document.nodes[mapping].payload = .trainingData(config: ["featureColumns": .array([.string("a"), .string("b")]), "targetColumn": .string("label")])
        XCTAssertThrowsError(try compile(document)) { error in
            XCTAssertEqual((error as? WorkflowCompilerError)?.code, .configuration)
            XCTAssertTrue(error.localizedDescription.contains("requires 3 ordered columns"))
        }
    }
}
