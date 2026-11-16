import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CompositeWorkspaceTests: XCTestCase {
    @MainActor private func store() -> CompositeWorkspaceStore {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        return CompositeWorkspaceStore(workspace: workspace)
    }
    @MainActor func testDraftPublishLeavesPinnedInstancesUnchangedUntilExplicitUpgrade() throws {
        let store = store()
        let old = store.project.root.instances[0].definition
        store.openDefinition(instanceID: "features")
        store.beginRevision()
        store.selectNode("activation-a", extending: false)
        store.renameSelected("My nonlinearity")
        try store.publishRevision()
        XCTAssertEqual(store.project.root.instances[0].definition, old)
        let published = try XCTUnwrap(store.project.definitions.last?.reference)
        XCTAssertNotEqual(published, old)
        store.showRoot()
        try store.upgradeInstance("features", to: published)
        XCTAssertEqual(store.project.root.instances[0].definition, published)
        XCTAssertNoThrow(try CompositeExpander.expand(store.project, registry: store.primitiveRegistry))
    }
    @MainActor func testInstanceCopiesHaveIndependentOverridesAndDraftUndo() throws {
        let store = store()
        store.selectNode("features", extending: false); store.duplicateSelection()
        XCTAssertEqual(store.project.root.instances.count, 2)
        let copy = try XCTUnwrap(store.project.root.instances.first { $0.id != "features" })
        store.selectNode(copy.id, extending: false); store.setNumeric("branchWidth", value: 4)
        XCTAssertEqual(store.project.root.instances.first { $0.id == copy.id }?.overrides["branchWidth"], 4)
        XCTAssertNil(store.project.root.instances.first { $0.id == "features" }?.overrides["branchWidth"])
        store.undo()
        XCTAssertEqual(store.project.root.instances.first { $0.id == copy.id }?.overrides, [:])
    }
    @MainActor func testPublishedDefinitionCannotBeMutatedWithoutDraft() {
        let store = store(); let before = store.project
        store.openDefinition(instanceID: "features")
        store.selectNode("branch-a", extending: false); store.setNumeric("in_features", value: 3)
        XCTAssertEqual(store.project, before)
    }
    func testPublishedInterfaceRejectsTwoParametersDrivingOneField() {
        var definition = CompositeExamples.parallelClassifier.definitions[0]
        var duplicate = definition.parameters[0]; duplicate.field.key = "anotherWidth"
        definition.parameters.append(duplicate)
        XCTAssertThrowsError(try CompositeDraftValidation.validate(definition, registry: ComponentRegistry.builtIn()))
    }

    @MainActor func testSaveReopenPreservesDraftLayoutAndPinnedSource() throws {
        let store = store()
        defer { try? FileManager.default.removeItem(at: store.workspace.workspaceDirectory) }
        store.openDefinition(instanceID: "features"); store.beginRevision()
        store.selectNode("activation-a", extending: false); store.renameSelected("Renamed draft")
        store.moveNodes(ids: ["activation-a"], dx: 17, dy: 29)
        let expected = store.state; store.save()
        let reopened = CompositeWorkspaceStore(workspace: store.workspace)
        XCTAssertEqual(reopened.state, expected)
        XCTAssertTrue(reopened.editingDraft)
        XCTAssertEqual(reopened.project.root.instances[0].definition, CompositeExamples.parallelClassifier.root.instances[0].definition)
    }

    @MainActor func testDefinitionCanvasProjectsExposedDefaultsWithoutMutatingPublishedBody() throws {
        let store = store(); store.openDefinition(instanceID: "features"); store.beginRevision()
        let original = store.activeDefinition!.body
        store.state.draft!.parameters[0].field.defaultValue = 6
        let target = try XCTUnwrap(store.activeDefinition?.parameters[0].targets.first)
        XCTAssertEqual(store.graph.nodes.first { $0.id == target.componentID }?.config[target.field], 6)
        XCTAssertEqual(store.activeDefinition?.body, original)
        store.moveNodes(ids: [target.componentID], dx: 17, dy: 0)
        XCTAssertEqual(store.activeDefinition?.body.nodes.first { $0.id == target.componentID }?.config, original.nodes.first { $0.id == target.componentID }?.config)
    }

    @MainActor func testBoundaryDelayedCommitsRejectRemovedRowsAndChangedContext() throws {
        let store=store();store.openDefinition(instanceID:"features");store.beginRevision()
        let original=try XCTUnwrap(store.activeDefinition?.parameters.first)
        var second=original;second.field.key="secondWidth";second.targets=[]
        store.updateBoundary{$0.parameters.append(second)}
        let captured=store.boundaryRevision
        store.updateBoundary{$0.parameters.removeFirst()}
        let before=store.state
        store.setBoundaryDefault(key:original.field.key,value:19,context:captured)
        store.setBoundaryDefault(key:"secondWidth",value:17,context:captured)
        XCTAssertEqual(store.state,before)
        store.setBoundaryDefault(key:"secondWidth",value:6,context:store.boundaryRevision)
        XCTAssertEqual(store.activeDefinition?.parameters.first?.field.defaultValue,6)
        let beforeUndo=store.boundaryRevision;store.undo();let undone=store.state
        store.setBoundaryDefault(key:"secondWidth",value:9,context:beforeUndo)
        XCTAssertEqual(store.state,undone)
        let focusedLast=store.boundaryRevision;store.updateBoundary{$0.parameters.removeAll()}
        store.setBoundaryDefault(key:"secondWidth",value:9,context:focusedLast)
        XCTAssertTrue(store.activeDefinition?.parameters.isEmpty == true)
        store.undo();let definitionContext=store.boundaryRevision
        store.showRoot();store.openDefinition(instanceID:"features");let returned=store.state
        store.setBoundaryDefault(key:"secondWidth",value:11,context:definitionContext)
        XCTAssertEqual(store.state,returned)
    }

}
