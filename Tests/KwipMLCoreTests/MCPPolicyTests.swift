import XCTest
@testable import KwipMLCore
final class MCPPolicyTests:XCTestCase {
    let project="90DC3005-0664-4D9E-B6C7-A4A510EDC844"
    func testFactoryDeniesAllAccessAndUsesAutomaticLocalPort()throws {
        let settings=MCPSettings.factory
        XCTAssertFalse(settings.enabled);XCTAssertNil(settings.fixedPort)
        XCTAssertThrowsError(try settings.authorize(clientID:"client",projectID:project,action:.workflowInspect))
    }
    func granted()->MCPSettings {
        var s=MCPSettings.factory;s.enabled=true
        s.clients=[MCPClientGrant(id:"client",name:"Synthetic client",actions:[.workflowInspect,.patchApply,.runStart,.resultsRead])]
        s.projects=[project:MCPProjectGrant(actions:[.workflowInspect,.patchApply,.runStart,.resultsRead],readCapturedValues:true)]
        return s
    }
    func testCreatedProjectGrantIsConservativeAndDoesNotExpandOnDecode()throws {
        let expected:Set<MCPAction>=[.projectOpen,.projectSave,.workflowInspect,.workflowValidate,.patchPreview,.patchApply,.patchUndo,.runStart,.runCancel,.jobGet,.resultsRead]
        XCTAssertEqual(MCPProjectGrant.createdProject.actions,expected)
        var policy=granted();policy.projects[project] = .createdProject
        policy.clients[0].actions=Set(MCPAction.allCases)
        for action in expected {XCTAssertNoThrow(try policy.authorize(clientID:"client",projectID:project,action:action))}
        for action:MCPAction in [.settingsGet,.settingsPreview,.settingsApply] {XCTAssertThrowsError(try policy.authorize(clientID:"client",projectID:project,action:action))}
        XCTAssertThrowsError(try policy.authorize(clientID:"client",projectID:project,action:.resultsRead,requiresValues:true))
        XCTAssertThrowsError(try policy.authorize(clientID:"client",projectID:project,action:.resultsRead,requiresHistory:true))
        policy.clients[0].actions.remove(.runStart)
        XCTAssertThrowsError(try policy.authorize(clientID:"client",projectID:project,action:.runStart))
        policy.projects[project] = .init(actions:[.workflowInspect])
        XCTAssertEqual(try MCPSettings.decode(policy.encoded()).projects[project]?.actions,[.workflowInspect])
    }
    func testClientProjectAndActionPermissionsIntersect()throws {
        var s=granted();try s.authorize(clientID:"client",projectID:project,action:.workflowInspect)
        XCTAssertThrowsError(try s.authorize(clientID:"other",projectID:project,action:.workflowInspect))
        XCTAssertThrowsError(try s.authorize(clientID:"client",projectID:"missing",action:.workflowInspect))
        s.disabledActions=[.patchApply]
        XCTAssertThrowsError(try s.authorize(clientID:"client",projectID:project,action:.patchApply))
    }
    func testRevocationAndCapturedValuesAreIndependent()throws {
        var s=granted();s.projects[project]?.readCapturedValues=false
        XCTAssertThrowsError(try s.authorize(clientID:"client",projectID:project,action:.resultsRead,requiresValues:true))
        try s.authorize(clientID:"client",projectID:project,action:.workflowInspect)
        s.enabled=false
        XCTAssertThrowsError(try s.authorize(clientID:"client",projectID:project,action:.workflowInspect))
    }
    func testHistoryRequiresSeparateGrant()throws {
        let s=granted()
        XCTAssertThrowsError(try s.authorize(clientID:"client",projectID:project,action:.resultsRead,requiresHistory:true))
    }
    func testDisabledCanonicalComponentBlocksExistingRun()throws {
        var s=granted();s.disabledComponents=["kwip.data.columns"]
        XCTAssertThrowsError(try s.authorize(clientID:"client",projectID:project,action:.runStart,components:["kwip.data.columns"]))
        try s.authorize(clientID:"client",projectID:project,action:.runStart,components:["kwip.data.view"])
    }
    func testOnlyValidNonprivilegedFixedPorts()throws {
        var s=MCPSettings.factory
        for port in [0,1,80,1023,65536] {s.fixedPort=port;XCTAssertThrowsError(try s.validated())}
        for port in [1024,8788,65535] {s.fixedPort=port;XCTAssertNoThrow(try s.validated())}
        s.fixedPort=nil;XCTAssertNoThrow(try s.validated())
    }
    func testMalformedPolicyFailsClosed()throws {
        XCTAssertThrowsError(try MCPSettings.decode(Data("{\"schemaVersion\":99}".utf8)))
        var s=granted();s.clients.append(s.clients[0]);XCTAssertThrowsError(try s.validated())
        s=granted();s.projects["../private"]=MCPProjectGrant();XCTAssertThrowsError(try s.validated())
    }
    func testPolicyRoundtripContainsNoBearerSecrets()throws {
        let s=try granted().validated();let data=try s.encoded()
        XCTAssertEqual(try MCPSettings.decode(data),s)
        XCTAssertFalse(String(decoding:data,as:UTF8.self).contains("Bearer"))
    }
}
