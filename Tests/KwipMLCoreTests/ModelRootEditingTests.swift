import XCTest
@testable import KwipMLCore

final class ModelRootEditingTests: XCTestCase {
    func testFlatMoveAndDuplicateRemainLegacyWithoutConfigurationMigration() throws {
        var graph = ExampleFlows.parallelClassifier; graph.schemaVersion = 1
        for i in graph.nodes.indices { graph.nodes[i].config.removeValue(forKey: "bias") }
        let result = try ModelRootEditing.apply(.move(nodeIDs: ["branch-a"], offset: .init(x: 7, y: 9)), to: .legacyFlat(graph), registry: .builtIn())
        guard case .legacyFlat(let moved) = result.document else { return XCTFail("Implicit promotion") }
        XCTAssertEqual(moved.schemaVersion, 1)
        XCTAssertEqual(moved.nodes.first { $0.id == "branch-a" }?.config, graph.nodes.first { $0.id == "branch-a" }?.config)
        XCTAssertEqual(moved.nodes.first { $0.id == "branch-a" }!.x, graph.nodes.first { $0.id == "branch-a" }!.x + 7)
        let duplicate = try ModelRootEditing.apply(.duplicate(nodeIDs: ["branch-a"], offset: .init(), semantics: .independent), to: result.document, registry: .builtIn(), makeID: { "copy" })
        guard case .legacyFlat(let copied) = duplicate.document else { return XCTFail("Implicit promotion") }
        XCTAssertNil(copied.nodes.last?.config["bias"])
    }

    func testBlockDuplicatePreservesPinsRawOverridesAndPrimitiveConfigs() throws {
        var source = CompositeExamples.parallelClassifier
        source.root.configurationVersion = 1
        for i in source.root.nodes.indices { source.root.nodes[i].config.removeValue(forKey: "bias") }
        source.root.instances[0].overrides = ["branchWidth": 5]
        let original = ModelAuthoringDocument(projectID: UUID().uuidString, revision: 4, source: source)
        var nextID = 0
        let result = try ModelRootEditing.apply(.duplicate(nodeIDs: ["features", "classifier"], offset: .init(), semantics: .independent), to: .blockModel(original), registry: .builtIn(), makeID: { nextID += 1; return "copy-\(nextID)" })
        guard case .blockModel(let changed) = result.document else { return XCTFail("Missing Block source") }
        XCTAssertEqual(changed.revision, 5)
        XCTAssertEqual(changed.source.root.configurationVersion, 1)
        XCTAssertEqual(changed.source.definitions, source.definitions)
        let instance = changed.source.root.instances.first { $0.id == result.command.copies.nodes["features"] }
        XCTAssertEqual(instance?.definition, source.root.instances[0].definition)
        XCTAssertEqual(instance?.overrides, ["branchWidth": 5])
        let primitive = changed.source.root.nodes.first { $0.id == result.command.copies.nodes["classifier"] }
        XCTAssertEqual(primitive?.config, source.root.nodes.first { $0.id == "classifier" }?.config)
        XCTAssertEqual(changed.source.execution, source.execution)
        let selected = try ModelRootEditing.apply(.select(.init(nodeIDs: ["features"]), mode: .replace), to: result.document, registry: .builtIn())
        XCTAssertEqual(selected.document, result.document)
        XCTAssertEqual(selected.command.after.selection.nodeIDs, ["features"])
        let noop = try ModelRootEditing.apply(.move(nodeIDs: ["features"], offset: .init(x: 0, y: 0)), to: result.document, registry: .builtIn())
        XCTAssertEqual(noop.document, result.document)
    }

    func testFirstInsertionPreservesFlatSourceAndRejectsPinConflictAndDraft() throws {
        var flat = ExampleFlows.parallelClassifier; flat.schemaVersion = 1
        for i in flat.nodes.indices { flat.nodes[i].config.removeValue(forKey: "bias") }
        let definition = CompositeExamples.parallelClassifier.definitions[0]
        let instance = CompositeInstance(id: "new-block", definition: definition.reference, overrides: ["branchWidth": 6])
        let id = UUID().uuidString
        let result = try ModelRootEditing.insertBlock(.legacyFlat(flat), definition: definition, instance: instance, projectID: id)
        guard case .blockModel(let model) = result else { return XCTFail("Expected promotion") }
        XCTAssertEqual(model.source.root.nodes, flat.nodes)
        XCTAssertEqual(model.source.root.edges, flat.edges)
        XCTAssertEqual(model.source.root.configurationVersion, 1)
        XCTAssertEqual(model.source.execution, flat.execution)
        XCTAssertEqual(model.revision, 1)
        var conflict = definition; conflict.title = "Different immutable pin"
        XCTAssertThrowsError(try ModelRootEditing.insertBlock(result, definition: conflict, instance: .init(id: "another", definition: definition.reference), projectID: id))
        XCTAssertThrowsError(try ModelRootEditing.insertBlock(result, definition: definition, instance: instance, projectID: id))
        let draft = try ModelBlockEditing.beginDraft(model, path: ["new-block"])
        XCTAssertThrowsError(try ModelRootEditing.apply(.move(nodeIDs: ["input"], offset: .init()), to: .blockModel(draft), registry: .builtIn()))
        XCTAssertThrowsError(try ModelRootEditing.insertBlock(.blockModel(draft), definition: definition, instance: .init(id: "another", definition: definition.reference), projectID: id))
    }
}
