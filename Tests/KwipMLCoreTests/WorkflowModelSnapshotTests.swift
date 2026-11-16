import XCTest
@testable import KwipMLCore

final class WorkflowModelSnapshotTests: XCTestCase {
    private let definition = WorkflowNodeAddress(domain: .control, localID: "model")
    private func fixture(block: Bool = false) -> WorkflowDocument {
        let plain = ExampleFlows.parallelClassifier
        let composite = CompositeExamples.parallelClassifier
        let primitives = block ? composite.root.nodes : plain.nodes
        var nodes = primitives.map { WorkflowNode(address: .init(domain: .model, localID: $0.id), name: $0.name, x: $0.x, y: $0.y, payload: .modelPrimitive(type: $0.type, configurationVersion: block ? (composite.root.configurationVersion ?? 1) : plain.schemaVersion, config: $0.config)) }
        if block { nodes += composite.root.instances.map { .init(address: .init(domain: .model, localID: $0.id), name: $0.name, x: $0.x, y: $0.y, payload: .modelBlock(reference: $0.definition, overrides: $0.overrides)) } }
        nodes.append(.init(address: definition, payload: .modelDefinition))
        var edges = (block ? composite.root.edges : plain.edges).map { WorkflowEdge(id: $0.id, from: .init(address: .init(domain: .model, localID: $0.source), port: $0.sourcePort), to: .init(address: .init(domain: .model, localID: $0.target), port: $0.targetPort), kind: .tensor) }
        edges.append(.init(from: .init(address: .init(domain: .model, localID: "output"), port: "definition"), to: .init(address: definition, port: "source"), kind: .modelDefinition))
        return .init(projectID: UUID().uuidString, revision: 9, name: "Untrained model", nodes: nodes, edges: edges, definitions: block ? composite.definitions : [])
    }
    private func snapshot(_ document: WorkflowDocument, definition: WorkflowNodeAddress? = nil) throws -> ModelExecutionSnapshot {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.modelSnapshot(document, definition: definition ?? self.definition, modelRegistry: .builtIn(), dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testValidModelGeneratesBeforeAnyDataOrTrainingConfiguration() throws {
        var document = fixture()
        document.nodes += [.init(address: .init(domain: .data, localID: "input"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:])), .init(address: .init(domain: .control, localID: "unfinished"), payload: .runSetup(config: ["epochs": .string("-")]))]
        let original = document
        let result = try snapshot(document)
        var expected = ExampleFlows.parallelClassifier
        expected.name = document.name; expected.execution = nil
        XCTAssertEqual(result.graph, expected)
        XCTAssertEqual(document, original)
        XCTAssertNil(result.graph.execution)
    }
    func testPinnedBlockSnapshotMatchesExistingExpansionWithoutTrainingSettings() throws {
        let document = fixture(block: true)
        var source = CompositeExamples.parallelClassifier
        source.name = document.name; source.execution = nil
        let expected = try ModelExecutionSnapshotBuilder.build(.blockModel(.init(projectID: document.projectID, revision: document.revision, source: source)), registry: .builtIn())
        XCTAssertEqual(try snapshot(document), expected)
    }
    func testWrongBoundaryAndTypedWireCannotBypassModelValidation() throws {
        let original = fixture()
        XCTAssertEqual(try snapshot(original, definition: .init(domain: .model, localID: "output")).graph,try snapshot(original).graph)
        XCTAssertThrowsError(try snapshot(original,definition:.init(domain:.model,localID:"linear")))
        var wrongKind = original
        wrongKind.edges[0].kind = .table
        XCTAssertThrowsError(try snapshot(wrongKind))
        var duplicate = original
        var edge = try XCTUnwrap(duplicate.edges.last)
        edge.id = UUID().uuidString
        duplicate.edges.append(edge)
        XCTAssertThrowsError(try snapshot(duplicate))
        var mixed = original
        let index = try XCTUnwrap(mixed.nodes.firstIndex { if case .modelPrimitive("linear", _, _) = $0.payload { return true }; return false })
        if case .modelPrimitive(let type, _, let config) = mixed.nodes[index].payload { mixed.nodes[index].payload = .modelPrimitive(type: type, configurationVersion: 1, config: config) }
        XCTAssertThrowsError(try snapshot(mixed))
    }
    func testSelectedDefinitionExcludesUnrelatedIncompleteModelAndRejectsMissingDependency() throws {
        var document = fixture()
        document.nodes.append(.init(address: .init(domain: .model, localID: "other"), payload: .modelPrimitive(type: "linear", configurationVersion: 2, config: [:])))
        XCTAssertFalse(try snapshot(document).graph.nodes.contains { $0.id == "other" })
        document.nodes.removeAll { $0.address == .init(domain: .model, localID: "input") }
        XCTAssertThrowsError(try snapshot(document))
    }
}
