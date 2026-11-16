import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor private func waitForGameIdle(_ host:WorkflowWorkspaceSession)async throws {
    let deadline=Date().addingTimeInterval(30)
    while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
    XCTAssertFalse(host.hasActiveOperation,host.statusMessage)
}

extension MCPWorkflowGatewayTests {
    func testGameDirectExecutionOwnsFreshIdentityAndPreservesDownstreamTables()async throws {
        let f=try fixture(document:GameEpisodePractice.document())
        let host=try XCTUnwrap(f.windows.openControllers.first?.session.workspace.workflowSession)
        let initial=host.operationIdentity
        let first=try XCTUnwrap(host.startGameSession(GameEpisodePractice.session))
        let firstIdentity=host.operationIdentity
        XCTAssertNotEqual(initial,firstIdentity)
        XCTAssertNil(host.startGameSession(GameEpisodePractice.session),"Concurrent admission must refuse")
        XCTAssertEqual(firstIdentity,host.operationIdentity)
        try await waitForGameIdle(host)
        XCTAssertEqual(first.summary["terminal"],8,first.message)
        XCTAssertEqual(firstIdentity,host.operationIdentity,"Owned downstream preparation keeps identity")
        let table=WorkflowNodeAddress(domain:.data,localID:"episodes")
        XCTAssertNotNil(host.snapshots(for:[table])[table])
        let validation=try XCTUnwrap(host.startGameSession(GameEpisodePractice.session,validateOnly:true))
        try await waitForGameIdle(host)
        XCTAssertEqual(firstIdentity,host.operationIdentity,"Validation is not a new execution")
        XCTAssertTrue(validation.validateOnly)
        XCTAssertTrue(host.gameEpisodes === first,"Validation must not replace the executed game capture")
        XCTAssertNil(host.startGameSession(.init(domain:.control,localID:"missing")))
        XCTAssertEqual(firstIdentity,host.operationIdentity,"Refused admission does not supersede execution")
        let second=try XCTUnwrap(host.startGameSession(GameEpisodePractice.session))
        XCTAssertNotEqual(firstIdentity,host.operationIdentity)
        try await waitForGameIdle(host)
        XCTAssertEqual(second.summary["terminal"],8,second.message)
    }

    func testOldMCPGameJobCannotCancelLaterNativeGameExecution()async throws {
        let f=try fixture(document:GameEpisodePractice.document())
        let host=try XCTUnwrap(f.windows.openControllers.first?.session.workspace.workflowSession)
        let terminal=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(GameEpisodePractice.session))
        let accepted=try await call(f,.runStart,["terminal":terminal])
        let job=try XCTUnwrap(accepted["jobID"])
        let ownedIdentity=host.operationIdentity
        // Deliberately do not poll the old job before a native replacement begins.
        try await waitForGameIdle(host)
        let later=try XCTUnwrap(host.startGameSession(GameEpisodePractice.session))
        XCTAssertNotEqual(ownedIdentity,host.operationIdentity)
        let cancelled=try await call(f,.runCancel,["jobID":job])
        guard case .object(let old)=cancelled["job"] else{return XCTFail("Missing job response")}
        XCTAssertEqual(old["state"],.string("superseded"))
        try await waitForGameIdle(host)
        XCTAssertEqual(later.summary["terminal"],8,later.message)
        XCTAssertEqual(host.terminalStatus(for:GameEpisodePractice.session),"Completed")
    }
}
