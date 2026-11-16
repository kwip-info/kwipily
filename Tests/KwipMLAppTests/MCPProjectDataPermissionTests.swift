import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MCPProjectDataPermissionTests:XCTestCase {
    func testHistoryControlPersistsIndependentlyAndCanBeRevoked()throws {
        let project=UUID().uuidString,other=UUID().uuidString,client=UUID().uuidString
        var settings=MCPSettings.factory
        settings.enabled=true
        settings.clients=[.init(id:client,name:"Verification",actions:[.resultsRead])]
        settings.projects[project] = .init(actions:[.resultsRead])
        settings.projects[other] = .init(actions:[.workflowInspect],readCapturedValues:true)
        let before=settings
        XCTAssertEqual(Set(MCPProjectDataPermission.allCases),[.history,.capturedValues])
        XCTAssertThrowsError(try settings.authorize(clientID:client,projectID:project,action:.resultsRead,requiresHistory:true))
        MCPProjectDataPermission.history.setAllowed(true,project:project,in:&settings)
        settings=try MCPSettings.decode(settings.encoded())
        XCTAssertNoThrow(try settings.authorize(clientID:client,projectID:project,action:.resultsRead,requiresHistory:true))
        XCTAssertThrowsError(try settings.authorize(clientID:client,projectID:project,action:.resultsRead,requiresValues:true,requiresHistory:true))
        XCTAssertFalse(MCPProjectDataPermission.capturedValues.isAllowed(project:project,in:settings))
        XCTAssertEqual(settings.projects[other],before.projects[other])
        XCTAssertEqual(settings.projects[project]?.actions,before.projects[project]?.actions)
        XCTAssertEqual(settings.clients,before.clients)
        MCPProjectDataPermission.history.setAllowed(false,project:project,in:&settings)
        XCTAssertEqual(try MCPSettings.decode(settings.encoded()),before)
    }

    func testDataControlsCannotCreateProjectAccessOrGrantHistoryImplicitly() {
        var settings=MCPSettings.factory
        let missing=UUID().uuidString,allowed=UUID().uuidString
        MCPProjectDataPermission.history.setAllowed(true,project:missing,in:&settings)
        XCTAssertNil(settings.projects[missing])
        settings.projects[allowed] = .init(actions:[.resultsRead])
        MCPProjectDataPermission.capturedValues.setAllowed(true,project:allowed,in:&settings)
        XCTAssertTrue(settings.projects[allowed]?.readCapturedValues == true)
        XCTAssertFalse(MCPProjectDataPermission.history.isAllowed(project:allowed,in:settings))
    }
}
