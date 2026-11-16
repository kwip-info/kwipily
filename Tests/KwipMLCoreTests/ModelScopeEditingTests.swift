import XCTest
@testable import KwipMLCore

final class ModelScopeEditingTests: XCTestCase {
    func testProjectionExposesPinnedPortsAndDefaultsWithoutChangingSource() throws {
        let source = CompositeExamples.parallelClassifier
        let projection = try ModelScopeEditing.project(source.root, definitions: source.definitions, name: source.name, execution: source.execution, registry: .builtIn())
        let instance = try XCTUnwrap(projection.graph.nodes.first { $0.id == "features" })
        let component = try XCTUnwrap(projection.registry.component(type: instance.type))
        XCTAssertEqual(component.inputs, ["in"])
        XCTAssertEqual(component.outputs, ["out"])
        XCTAssertEqual(instance.config["branchWidth"], 8)
        XCTAssertTrue(source.root.instances[0].overrides.isEmpty)
        XCTAssertEqual(projection.graph.execution, source.execution)
    }

    func testChildScopeDuplicateAndNoOpPreserveRawConfigurationAndPins() throws {
        let source = CompositeExamples.parallelClassifier
        var scope = source.definitions[0].body
        scope.configurationVersion = nil
        for i in scope.nodes.indices { scope.nodes[i].config.removeValue(forKey: "bias") }
        scope.instances = [.init(id: "nested", definition: source.definitions[0].reference, overrides: [:])]
        var counter = 0
        let result = try ModelScopeEditing.apply(.duplicate(nodeIDs: ["branch-a", "nested"], offset: .init(), semantics: .independent), to: scope, definitions: source.definitions, registry: .builtIn(), makeID: { counter += 1; return "duplicate-\(counter)" })
        XCTAssertNil(result.scope.configurationVersion)
        let copiedNode = result.scope.nodes.first { $0.id == result.command.copies.nodes["branch-a"] }
        XCTAssertEqual(copiedNode?.config, scope.nodes.first { $0.id == "branch-a" }?.config)
        let copiedInstance = result.scope.instances.first { $0.id == result.command.copies.nodes["nested"] }
        XCTAssertEqual(copiedInstance?.definition, scope.instances[0].definition)
        XCTAssertEqual(copiedInstance?.overrides, [:])
        let noOp = try ModelScopeEditing.apply(.move(nodeIDs: ["nested"], offset: .init(x: 0, y: 0)), to: scope, definitions: source.definitions, registry: .builtIn())
        XCTAssertEqual(noOp.scope, scope)
        let deleted = try ModelScopeEditing.apply(.delete(.init(nodeIDs: ["nested"])), to: result.scope, definitions: source.definitions, registry: .builtIn())
        XCTAssertFalse(deleted.scope.instances.contains { $0.id == "nested" })
        XCTAssertEqual(deleted.scope.nodes, result.scope.nodes)
    }

    func testStructuralConnectionsRejectWrongPortsAndOccupancyAndCycles() throws {
        let source = CompositeExamples.parallelClassifier
        XCTAssertThrowsError(try ModelScopeEditing.apply(.connect(source: .init("input", "out"), target: .init("features", "missing")), to: source.root, definitions: source.definitions, registry: .builtIn()))
        XCTAssertThrowsError(try ModelScopeEditing.apply(.connect(source: .init("input", "out"), target: .init("features", "in")), to: source.root, definitions: source.definitions, registry: .builtIn()))
        var unconnected = source.root; unconnected.edges = []
        let connected = try ModelScopeEditing.apply(.connect(source: .init("features", "out"), target: .init("classifier", "in")), to: unconnected, definitions: source.definitions, registry: .builtIn())
        XCTAssertThrowsError(try ModelScopeEditing.apply(.connect(source: .init("classifier", "out"), target: .init("features", "in")), to: connected.scope, definitions: source.definitions, registry: .builtIn()))
        XCTAssertEqual(unconnected.edges, [])
    }

    func testMalformedProjectionFailsInsteadOfTrappingOnDuplicateMaps() throws {
        let source = CompositeExamples.parallelClassifier
        XCTAssertThrowsError(try ModelScopeEditing.project(source.root, definitions: source.definitions + source.definitions, registry: .builtIn()))
        var bad = source.root; bad.nodes.append(bad.nodes[0])
        XCTAssertThrowsError(try ModelScopeEditing.project(bad, definitions: source.definitions, registry: .builtIn()))
        XCTAssertThrowsError(try ModelScopeEditing.project(source.root, definitions: [], registry: .builtIn()))
    }
}
