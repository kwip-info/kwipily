import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DataNodeEditingContextTests: XCTestCase {
    private func editor(lesson:Bool=false)->DataFlowWorkspaceStore {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at:directory) }
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.activeProjectID=UUID().uuidString
        let project=DataFlowProject(graph:.init(nodes:[
            .init(id:"source",operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")]),
            .init(id:"cast",operatorID:"kwip.data.columns",config:ColumnsConfigurationDraft.defaults),
            .init(id:"split",operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20])],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"cast",port:"table"))]))
        let store=lesson ? DataFlowWorkspaceStore(workspace:workspace,lessonProject:project) : DataFlowWorkspaceStore(workspace:workspace)
        store.project=project;store.selectNode("cast")
        return store
    }
    func testStableContextCommitsExactNodeButRejectsSelectionReplacementUndoAndRetirement() throws {
        let store=editor();let context=PreparationDataNodeContext(store:store,nodeID:"cast")
        let identity=context.contextIdentity
        let original=try XCTUnwrap(context.node).config
        var draft=try ColumnsConfigurationDraft(configuration:original)
        draft.add(source:" a,b ");draft.update(" a,b ") {$0.targetType="float64"}
        let changed=draft.configuration
        XCTAssertTrue(context.configure(changed,expected:original))
        XCTAssertEqual(context.contextIdentity,identity,"A successful value edit must not recreate its field identity.")
        XCTAssertEqual(store.project.graph.nodes.first {$0.id == "cast"}?.config,changed)
        XCTAssertFalse(context.configure([:],expected:original),"A stale configuration snapshot must not overwrite the newer edit.")
        store.selectNode("split")
        XCTAssertFalse(context.configure([:],expected:changed))
        store.selectNode("cast")
        XCTAssertFalse(context.configure([:],expected:changed),"Returning to the same node does not revive its old context.")
        let next=PreparationDataNodeContext(store:store,nodeID:"cast")
        store.undo();store.selectNode("cast")
        XCTAssertFalse(next.rename("Late name"))
        let reopened=PreparationDataNodeContext(store:store,nodeID:"cast")
        store.historySessionRevision=UUID()
        XCTAssertFalse(reopened.configure([:],expected:original))
        let retired=PreparationDataNodeContext(store:store,nodeID:"cast");retired.retire()
        XCTAssertFalse(retired.rename("Retired"))
    }
    func testNumericPendingBelongsToExactLiveFieldAndCannotReturnAfterContextEnds() {
        let store=editor();store.selectNode("split")
        let context=PreparationDataNodeContext(store:store,nodeID:"split")
        let origin=context.numericOrigin(key:"validationPercent")
        context.setNumericPending(origin:origin,key:"validationPercent",pending:true)
        XCTAssertTrue(store.hasPendingNumericEdits)
        XCTAssertEqual(context.currentNumericOrigin(key:"validationPercent"),origin)
        store.selectNode("cast")
        XCTAssertNil(context.currentNumericOrigin(key:"validationPercent"))
        XCTAssertFalse(store.hasPendingNumericEdits)
        context.setNumericPending(origin:origin,key:"validationPercent",pending:true)
        XCTAssertFalse(store.hasPendingNumericEdits)
    }
    func testSourceOperationsCannotBypassFixtureInactiveOrBusyAuthority() {
        for lesson in [true,false] {
            let store=editor(lesson:lesson);store.selectNode("source")
            var operations=0
            let context=PreparationDataNodeContext(store:store,nodeID:"source",sourceAction:{_ in operations += 1})
            if !lesson { store.isPreparing=true }
            XCTAssertFalse(context.chooseSource());XCTAssertFalse(context.refreshSource());XCTAssertFalse(context.setSeparator("\t",expected:["separator":.string(",")]))
            XCTAssertEqual(operations,0)
            store.isPreparing=false;store.workspace.activeProjectID=UUID().uuidString
            XCTAssertFalse(context.chooseSource());XCTAssertFalse(context.refreshSource())
            XCTAssertEqual(operations,0)
        }
    }
    func testFixtureAndCachedHeaderProjectionPreserveExactNamesWithoutSourceReads() {
        let lesson=editor(lesson:true)
        let fixture=PreparationDataNodeContext(store:lesson,nodeID:"cast")
        XCTAssertEqual(fixture.upstreamSchema.map(\.name),["x","y"])
        let store=editor()
        let path="/unavailable/private-source.csv"
        store.project.sourceBindings["source"]=path
        store.sourceHeaders["source"] = .init(path:path,config:["separator":.string(",")],fingerprint:"recorded",columns:[" a,b ","target"])
        let context=PreparationDataNodeContext(store:store,nodeID:"cast")
        XCTAssertEqual(context.upstreamSchema.map(\.name),[" a,b ","target"],"Unrevalidated recorded headers remain useful declared guidance, even when the source is inaccessible.")
        XCTAssertTrue(context.schemaDisclosure.contains("not revalidated"))
        store.project.graph.nodes[0].config=["separator":.string("\t")]
        XCTAssertTrue(context.upstreamSchema.isEmpty,"Mismatched source configuration must not reuse recorded columns.")
    }
    func testTableDraftValidationRetainsSimultaneousRenamesAndRejectsCollisions() throws {
        let store=editor()
        var draft=try ColumnsConfigurationDraft(configuration:ColumnsConfigurationDraft.defaults)
        draft.add(source:" a,b ");draft.add(source:"target")
        store.project.graph.nodes.append(.init(id:"select",operatorID:"kwip.data.columns",config:draft.configuration))
        store.selectNode("select")
        let context=PreparationDataNodeContext(store:store,nodeID:"select")
        let original=context.node!.config
        draft.update(" a,b ") {$0.outputName="target"};draft.update("target") {$0.outputName=" a,b "}
        let swap=draft.configuration
        XCTAssertTrue(context.applyTableConfiguration(swap,expected:original))
        draft.update("target") {$0.outputName="target"}
        let collision=draft.configuration
        XCTAssertFalse(context.applyTableConfiguration(collision,expected:swap))
        XCTAssertEqual(context.node?.config,swap)
    }
}
