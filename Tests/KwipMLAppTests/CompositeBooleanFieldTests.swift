import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CompositeBooleanFieldTests: XCTestCase {
    func testExposedFieldsRequireMatchingKindsAndNarrowerNumericBounds() {
        let boolean = ComponentField(key: "bias", title: "Bias", kind: .boolean, defaultValue: true, min: nil, max: nil)
        let integer = ComponentField(key: "width", title: "Width", defaultValue: 8, min: 1, max: 512)
        XCTAssertTrue(RegistryFieldCompatibility.canExpose(boolean, to: boolean))
        XCTAssertFalse(RegistryFieldCompatibility.canExpose(boolean, to: integer))
        XCTAssertFalse(RegistryFieldCompatibility.canExpose(integer, to: boolean))
        var narrow = integer; narrow.min = 2; narrow.max = 16
        XCTAssertTrue(RegistryFieldCompatibility.canExpose(narrow, to: integer))
        XCTAssertFalse(RegistryFieldCompatibility.canExpose(integer, to: narrow))
    }
    @MainActor func testBooleanDefaultPinnedPublicationOverrideUndoAndReopen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        let store = CompositeWorkspaceStore(workspace: workspace)
        let old = store.project.root.instances[0].definition
        store.openDefinition(instanceID: "features"); store.beginRevision()
        let boolean = ComponentField(key: "branchBias", title: "Branch bias", kind: .boolean, defaultValue: true, min: nil, max: nil)
        store.updateBoundary { $0.parameters.append(CompositeParameter(boolean, targets: [CompositeParameterTarget("branch-a", "bias")])) }
        let context = store.boundaryRevision
        XCTAssertFalse(store.setBoundaryValue(key: "branchBias", value: .integer(0), context: context))
        XCTAssertTrue(store.setBoundaryValue(key: "branchBias", value: .bool(false), context: context))
        XCTAssertEqual(store.graph.nodes.first { $0.id == "branch-a" }?.config["bias"], .bool(false))
        XCTAssertTrue(store.canvasSummary(try XCTUnwrap(store.graph.nodes.first { $0.id == "branch-a" })).contains("no bias"))
        try store.publishRevision()
        XCTAssertEqual(store.project.root.instances[0].definition, old)
        let published = try XCTUnwrap(store.activeDefinition?.reference)
        store.showRoot(); try store.upgradeInstance("features", to: published); store.selectNode("features")
        let defaultExpansion = try CompositeExpander.expand(store.project, registry: workspace.registry)
        let branchID = try XCTUnwrap(defaultExpansion.nodeOrigins.first { $0.value.localNodeID == "branch-a" }?.key)
        XCTAssertEqual(defaultExpansion.graph.nodes.first { $0.id == branchID }?.config["bias"], .bool(false))
        XCTAssertTrue(store.setConfigValue("branchBias", value: .bool(true), nodeID: "features"))
        XCTAssertEqual(store.project.root.instances[0].overrides["branchBias"], .bool(true))
        store.undo(); XCTAssertNil(store.project.root.instances[0].overrides["branchBias"])
        store.redo(); XCTAssertEqual(store.project.root.instances[0].overrides["branchBias"], .bool(true))
        let expected = store.state; store.save(); store.state.project.name = "Discard me"; store.reopen()
        XCTAssertEqual(store.state, expected)
        let expanded = try CompositeExpander.expand(store.project, registry: workspace.registry)
        XCTAssertTrue(expanded.graph.nodes.contains { $0.type == "linear" && $0.config["bias"] == .bool(true) })
    }
}
