import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DataFlowNumericDraftTests: XCTestCase {
    private func withEditor(_ body:(DataFlowWorkspaceStore)throws->Void) rethrows {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.activeProjectID=UUID().uuidString
        let graph=DataFlowDocument(nodes:[DataFlowNode(id:"split",operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":25,"testPercent":25])])
        let editor=DataFlowWorkspaceStore(workspace:workspace)
        editor.project=DataFlowProject(graph:graph)
        editor.selectNode("split")
        try body(editor)
    }
    func testRejectedVisibleDraftRemainsPendingUntilCommittedOrRemoved() {
        withEditor { editor in
            let origin=editor.numericEditOrigin(nodeID:"split",key:"validationPercent")
            var edit=NumericEditState(value:.integer(25))
            edit.text="25.5"
            editor.setNumericEditPending(origin:origin,nodeID:"split",key:"validationPercent",pending:edit.needsCommit)
            XCTAssertTrue(editor.hasPendingNumericEdits)
            XCTAssertFalse(edit.commit(domain:.integer(minimum:1,maximum:50),origin:origin,currentOrigin:origin) {_ in XCTFail("Fractional edit must not reach setter");return true})
            editor.setNumericEditPending(origin:origin,nodeID:"split",key:"validationPercent",pending:edit.needsCommit)
            XCTAssertTrue(editor.hasPendingNumericEdits)
            XCTAssertEqual(editor.project.graph.nodes[0].config["validationPercent"],25)
            edit.text="30"
            XCTAssertTrue(edit.commit(domain:.integer(minimum:1,maximum:50),origin:origin,currentOrigin:origin) {_ in true})
            editor.setNumericEditPending(origin:origin,nodeID:"split",key:"validationPercent",pending:edit.needsCommit)
            XCTAssertFalse(editor.hasPendingNumericEdits)
        }
    }
    func testFallbackFieldAndInactiveProjectCannotBypassContextGate() {
        withEditor { editor in
            editor.project.graph.nodes[0].config.removeValue(forKey:"seed")
            let origin=editor.numericEditOrigin(nodeID:"split",key:"seed")
            editor.setNumericEditPending(origin:origin,nodeID:"split",key:"seed",pending:true)
            XCTAssertTrue(editor.hasPendingNumericEdits,"A displayed fallback still has an editable draft.")
            editor.setNumericEditPending(origin:origin,nodeID:"split",key:"seed",pending:false)
            editor.workspace.activeProjectID=UUID().uuidString
            editor.setNumericEditPending(origin:origin,nodeID:"split",key:"seed",pending:true)
            XCTAssertFalse(editor.hasPendingNumericEdits)
        }
    }
    func testSelectionReloadAndDeletedFieldsRejectLateDraftReports() {
        withEditor { editor in
            let old=editor.numericEditOrigin(nodeID:"split",key:"seed")
            editor.setNumericEditPending(origin:old,nodeID:"split",key:"seed",pending:true)
            XCTAssertEqual(editor.pendingNumericEdits.count,1)
            editor.clearSelection();editor.selectNode("split")
            XCTAssertFalse(editor.hasPendingNumericEdits)
            editor.setNumericEditPending(origin:old,nodeID:"split",key:"seed",pending:true)
            XCTAssertFalse(editor.hasPendingNumericEdits)
            let next=editor.numericEditOrigin(nodeID:"split",key:"seed")
            editor.setNumericEditPending(origin:next,nodeID:"split",key:"seed",pending:true)
            editor.historySessionRevision=UUID()
            XCTAssertFalse(editor.hasPendingNumericEdits)
            editor.setNumericEditPending(origin:next,nodeID:"split",key:"seed",pending:true)
            XCTAssertFalse(editor.hasPendingNumericEdits)
            let removed=editor.numericEditOrigin(nodeID:"split",key:"seed")
            editor.deleteSelection()
            editor.setNumericEditPending(origin:removed,nodeID:"split",key:"seed",pending:true)
            XCTAssertFalse(editor.hasPendingNumericEdits)
        }
    }
}
