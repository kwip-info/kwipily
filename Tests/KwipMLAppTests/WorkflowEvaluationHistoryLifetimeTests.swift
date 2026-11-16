import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowEvaluationHistoryLifetimeTests:XCTestCase {
    private func fixture()throws->(WorkspaceStore,WorkflowWorkspaceSession) {
        let helper=WorkflowEvaluationRequestTests()
        let(root,owner,_,runtime)=try helper.fixture()
        let storage=root.appendingPathComponent("history-lifetime-workspace")
        try FileManager.default.createDirectory(at:storage,withIntermediateDirectories:true)
        let workspace=WorkspaceStore(rootOverride:runtime.workingDirectoryURL,storageOverride:storage,detectCapabilities:false)
        workspace.activeProjectID=owner.projectID
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.workflowSession=host
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        return(workspace,host)
    }

    func testHistoryRemainsActiveAfterPersistedCanvasEditsButRetiresWithHost()throws {
        let(workspace,host)=try fixture()
        defer {withExtendedLifetime(workspace) {}}
        XCTAssertTrue(host.evaluationHistory.isActive)
        let before=host.owner.lease()
        let address=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        XCTAssertNotNil(host.owner.apply(.rename(address:address,name:"Final protected check"),lease:before))
        XCTAssertFalse(host.owner.isCurrent(before),"The edit must actually advance editor authority.")
        XCTAssertEqual(host.owner.document.nodes.first(where:{$0.address==address})?.name,"Final protected check")
        XCTAssertTrue(host.evaluationHistory.isActive,"Saved evaluations belong to the project session, not one editor generation.")
        XCTAssertTrue(host.owner.rename("Updated project name",lease:host.owner.lease()))
        XCTAssertTrue(host.evaluationHistory.isActive)
        host.retire()
        XCTAssertFalse(host.evaluationHistory.isActive)
    }

    func testReplacementHostCannotReactivatePreviousHistoryEvenWhenOwnerIsReused()throws {
        let(workspace,previous)=try fixture()
        XCTAssertTrue(previous.evaluationHistory.isActive)
        let replacement=try WorkflowWorkspaceSession(workspace:workspace,owner:previous.owner)
        workspace.workflowSession=replacement
        XCTAssertFalse(previous.evaluationHistory.isActive,"Host identity must reject callbacks from the displaced controller.")
        XCTAssertTrue(replacement.evaluationHistory.isActive)
        workspace.activeProjectID=UUID().uuidString
        XCTAssertFalse(previous.evaluationHistory.isActive)
        XCTAssertFalse(replacement.evaluationHistory.isActive)
        workspace.activeProjectID=replacement.owner.projectID
        XCTAssertTrue(replacement.evaluationHistory.isActive)
        replacement.owner.invalidate()
        XCTAssertFalse(replacement.evaluationHistory.isActive)
    }
}
