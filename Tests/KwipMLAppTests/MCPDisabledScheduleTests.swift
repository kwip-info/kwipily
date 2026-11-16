import XCTest
import KwipMLCore
@testable import KwipMLApp

extension MCPWorkflowGatewayTests {
    func testDisabledScheduleGatewayPreviewApplyDurableReplayAndRevocation()async throws {
        let f=try fixture(),root=f.controller.directory,version=try CanvasVersionJournal(root:root,projectID:f.owner.projectID).save(f.owner.document)
        let values:[String:JSONValue]=["versionID":.string(version.id),"versionNumber":.integer(Int64(version.number)),"versionHash":.string(version.hash),"name":.string("Review first"),"enabled":false,"target":.null,"busy":.string("skip"),"missed":.string("skip"),"timing":.object(["kind":.string("daily"),"timeZone":.string("UTC"),"hour":9,"minute":30])]
        let preview=try await call(f,.schedulePreview,["values":.object(values)])
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("scheduled-runs").path))
        XCTAssertEqual(preview["enabled"],false);XCTAssertEqual(preview["executionStarted"],false)
        let scheduler=ScheduledRunController.shared(workspace:f.windows.restorationWorkspace)
        try await scheduler.ready()
        let request:[String:JSONValue]=["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(UUID().uuidString)]
        let saved=try await call(f,.scheduleApply,request)
        XCTAssertEqual(saved["ruleRevision"],1);XCTAssertEqual(saved["outputConsent"],false)
        let nextGateway=MCPWorkflowGateway(controller:f.controller,windows:f.windows)
        var args=request;args["projectID"] = .string(f.owner.projectID)
        let replay=try await nextGateway.dispatch(clientID:f.client,action:"schedule_apply",arguments:args)
        XCTAssertEqual(replay,saved)
        var wrong=args;wrong["proposalID"] = .string(UUID().uuidString)
        do {_=try await nextGateway.dispatch(clientID:f.client,action:"schedule_apply",arguments:wrong);XCTFail("Changed retry accepted")}catch{}
        var policy=f.controller.settings;policy.clients[0].actions.remove(.scheduleApply);try f.controller.save(policy)
        do {_=try await nextGateway.dispatch(clientID:f.client,action:"schedule_apply",arguments:args);XCTFail("Revoked replay accepted")}catch{}
        XCTAssertFalse(scheduler.state.enabled);XCTAssertTrue(scheduler.state.jobs.isEmpty)
        await scheduler.shutdown()
    }
    func testDisabledScheduleDiscoveryStrictValuesAndLegacyPermissionIsolation()async throws {
        let f=try fixture(),version=try CanvasVersionJournal(root:f.controller.directory,projectID:f.owner.projectID).save(f.owner.document)
        XCTAssertFalse(MCPProjectGrant.createdProject.actions.contains(.schedulePreview));XCTAssertFalse(MCPProjectGrant.createdProject.actions.contains(.scheduleApply))
        let tools=try await f.gateway.dispatch(clientID:f.client,action:"tools_list",arguments:[:])
        let text=String(decoding:try JSONEncoder().encode(tools),as:UTF8.self);XCTAssertTrue(text.contains("schedule_preview"));XCTAssertTrue(text.contains("intervalMinutes"))
        let values:[String:JSONValue]=["versionID":.string(version.id),"versionNumber":.integer(Int64(version.number)),"versionHash":.string(version.hash),"name":.string("Draft"),"enabled":false,"target":.null,"busy":.string("skip"),"missed":.string("skip"),"timing":.object(["kind":.string("daily"),"timeZone":.string("UTC"),"hour":9,"minute":30])]
        for (key,value) in [("enabled",JSONValue.bool(true)),("allowEffects",.bool(true)),("destination",.string("/private/path"))] {
            var bad=values;bad[key]=value
            do {_=try await call(f,.schedulePreview,["values":.object(bad)]);XCTFail("Unsafe field accepted")}catch {XCTAssertFalse(String(describing:error).contains("/private/path"))}
        }
        var policy=f.controller.settings;policy.projects[f.owner.projectID] = .createdProject;try f.controller.save(policy)
        do {_=try await call(f,.schedulePreview,["values":.object(values)]);XCTFail("Legacy creation grant expanded")}catch{}
        XCTAssertFalse(FileManager.default.fileExists(atPath:f.controller.directory.appendingPathComponent("scheduled-runs").path))
    }
}

extension MCPWorkflowGatewayTests {
    func testOwnedScheduleActionsDoNotExpandLegacyGrantsOrExposeNativeCancellation()async throws {
        let f=try fixture()
        XCTAssertFalse(MCPProjectGrant.createdProject.actions.contains(.scheduleRunNow));XCTAssertFalse(MCPProjectGrant.createdProject.actions.contains(.scheduleCancel))
        let scheduler=ScheduledRunController.shared(workspace:f.windows.restorationWorkspace);try await scheduler.ready()
        do {_=try await call(f,.scheduleCancel,["jobID":.string(UUID().uuidString)]);XCTFail("Unknown or native job exposed")}catch let e as MCPSettings.Refusal {XCTAssertEqual(e.code,"schedule_owned_job_unavailable")}
        var policy=f.controller.settings;policy.projects[f.owner.projectID] = .createdProject;try f.controller.save(policy)
        for action in [MCPAction.scheduleRunNow,.scheduleCancel] {
            do {_=try await call(f,action);XCTFail("New execution grant inferred")}catch let e as MCPSettings.Refusal {XCTAssertEqual(e.code,"not_permitted")}
        }
        XCTAssertTrue(scheduler.state.jobs.isEmpty);XCTAssertFalse(scheduler.state.enabled)
        await scheduler.shutdown()
    }
}
