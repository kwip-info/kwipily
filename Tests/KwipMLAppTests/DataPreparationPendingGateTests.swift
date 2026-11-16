import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DataPreparationPendingGateTests: XCTestCase {
    func testUncommittedNumericValueBlocksPreparationBeforeInspectionOrRuntimeWork() {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:directory) }
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.activeProjectID=UUID().uuidString
        let store=DataFlowWorkspaceStore(workspace:workspace)
        var configuration=FilterRowsConfigurationDraft.defaults
        configuration["conditions"] = .array([.object(["column":.string("value"),"comparison":.string("ge"),"value":2,"caseSensitive":.bool(true)])])
        store.project = .init(graph:.init(nodes:[.init(id:"filter",operatorID:"kwip.data.filterRows",config:configuration)]))
        store.selectNode("filter")
        let origin=store.numericEditOrigin(nodeID:"filter",key:"conditions.0.value")
        store.setNumericEditPending(origin:origin,nodeID:"filter",key:"conditions.0.value",pending:true)
        // Simulate an in-flight inspection: the pending value must veto scheduling
        // preparation or cancelling unrelated inspection, before runtime checks.
        store.isInspectingSource=true
        store.runPreparation()
        XCTAssertFalse(store.prepareAfterInspection)
        XCTAssertTrue(store.statusMessage.localizedCaseInsensitiveContains("pending numeric"),store.statusMessage)
        XCTAssertFalse(store.isPreparing)
        XCTAssertNil(store.preparationWorker)
        XCTAssertEqual(store.project.graph.nodes[0].config,configuration)
        store.isInspectingSource=false
    }
}
