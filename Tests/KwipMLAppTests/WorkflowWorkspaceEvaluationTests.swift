import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowWorkspaceEvaluationTests:XCTestCase {
    private let frozen=WorkflowNodeAddress(domain:.control,localID:"frozen")
    private let terminal=WorkflowNodeAddress(domain:.control,localID:"evaluate")
    private func fixture()throws->(WorkspaceStore,WorkflowWorkspaceSession,WorkflowEvaluationRequestTests) {
        let helper=WorkflowEvaluationRequestTests()
        let(root,owner,original,runtime)=try helper.fixture()
        try FileManager.default.createDirectory(at:root.appendingPathComponent("workspace"),withIntermediateDirectories:true)
        let workspace=WorkspaceStore(rootOverride:runtime.workingDirectoryURL,storageOverride:root.appendingPathComponent("workspace"),detectCapabilities:false)
        workspace.activeProjectID=owner.projectID
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.workflowSession=host
        host.evaluationStage=helper.staged
        let reference=try XCTUnwrap(original.reference(for:frozen))
        guard host.frozenBindings.setReference(reference,for:frozen,lease:owner.lease(),expectedRevision:host.frozenBindings.revision) else {
            throw WorkflowArchiveError("Fixture frozen binding failed: "+(host.frozenBindings.error ?? "unknown error"))
        }
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        return(workspace,host,helper)
    }
    func testFrozenBindingClearRequiresCurrentLeaseAndPersistsRevision()throws {
        let(workspace,host,_)=try fixture()
        defer {withExtendedLifetime(workspace) {}}
        let lease=host.owner.lease(),before=host.frozenBindings.revision
        XCTAssertTrue(host.clearFrozenModel(for:frozen,lease:lease))
        XCTAssertNil(host.frozenBindings.reference(for:frozen))
        XCTAssertGreaterThan(host.frozenBindings.revision,before)
        host.retire()
        XCTAssertFalse(host.clearFrozenModel(for:frozen,lease:lease))
    }
    func testSelectionRechecksBindingRevisionAfterAsyncVerification()async throws {
        let(workspace,host,_)=try fixture()
        let selected=try XCTUnwrap(host.frozenBindings.reference(for:frozen))
        let entered=expectation(description:"Parent verification pending")
        var release:CheckedContinuation<WorkflowFrozenModelReference,Error>?
        host.selectFrozenReference={_,_ in
            try await withCheckedThrowingContinuation {release=$0;entered.fulfill()}
        }
        let task=Task {await host.selectFrozenModel(.init(projectID:host.owner.projectID,directoryID:UUID()),for:frozen)}
        await fulfillment(of:[entered],timeout:5)
        XCTAssertTrue(workspace.isBusy)
        XCTAssertTrue(host.frozenBindings.setReference(nil,for:frozen,lease:host.owner.lease(),expectedRevision:host.frozenBindings.revision))
        release?.resume(returning:selected)
        let accepted=await task.value
        XCTAssertFalse(accepted)
        XCTAssertNil(host.frozenBindings.reference(for:frozen))
        XCTAssertFalse(host.isStagingEvaluation)
        XCTAssertFalse(workspace.isBusy)
    }
    func testEvaluationHandoffAndCancellationStayBusyUntilActualExit()async throws {
        let(workspace,host,_)=try fixture()
        let launched=expectation(description:"Evaluation worker launched")
        var exit:((Int32,String)->Void)?,cancelled=0
        host.evaluationLaunch={_,_,_,_,onExit in exit=onExit;launched.fulfill();return .init(cancel:{cancelled+=1})}
        guard host.evaluate(address:terminal) else {return XCTFail("Evaluation refused: "+host.statusMessage)}
        XCTAssertTrue(host.isStagingEvaluation)
        XCTAssertTrue(workspace.isBusy)
        XCTAssertFalse(host.evaluate(address:terminal))
        await fulfillment(of:[launched],timeout:10)
        let evaluation=try XCTUnwrap(host.evaluation)
        let staged=evaluation.request.snapshot.directory
        XCTAssertFalse(host.isStagingEvaluation)
        XCTAssertTrue(workspace.isBusy)
        workspace.cancelRun()
        XCTAssertEqual(cancelled,1)
        XCTAssertTrue(workspace.isBusy)
        XCTAssertTrue(host.evaluationOutcomes.isEmpty)
        exit?(0,"")
        XCTAssertFalse(workspace.isBusy)
        XCTAssertFalse(host.owner.isBusy)
        XCTAssertEqual(host.evaluationOutcomes.last?.outcome.status,.cancelled)
        XCTAssertFalse(FileManager.default.fileExists(atPath:staged.path))
        XCTAssertTrue(host.outcomes.isEmpty,"Evaluation cannot become training history")
    }
    func testRetiredHostWaitsForWorkerExitWithoutChangingReplacementWorkspace()async throws {
        let(workspace,host,_)=try fixture()
        let launched=expectation(description:"Evaluation worker launched")
        var exit:((Int32,String)->Void)?,cancelled=0
        host.evaluationLaunch={_,_,_,_,onExit in exit=onExit;launched.fulfill();return .init(cancel:{cancelled+=1})}
        guard host.evaluate(address:terminal) else {return XCTFail("Evaluation refused: "+host.statusMessage)}
        await fulfillment(of:[launched],timeout:10)
        let staged=try XCTUnwrap(host.evaluation).request.snapshot.directory
        host.retire()
        XCTAssertEqual(cancelled,1)
        XCTAssertTrue(workspace.isPreparingData)
        workspace.workflowSession=nil
        workspace.activeProjectID=UUID().uuidString
        workspace.statusMessage="Replacement project"
        exit?(0,"")
        XCTAssertFalse(workspace.isPreparingData)
        XCTAssertEqual(workspace.statusMessage,"Replacement project")
        XCTAssertFalse(FileManager.default.fileExists(atPath:staged.path))
    }
}
