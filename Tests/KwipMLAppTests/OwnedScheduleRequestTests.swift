import XCTest
import KwipMLCore
@testable import KwipMLApp

final class OwnedScheduleRequestTests:XCTestCase {
    private func fixture(effects:Bool=false)throws->(URL,ScheduleRule) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("owned-schedule-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let nodes:[WorkflowNode]=effects ? [.init(address:.init(domain:.data,localID:"output"),payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:[:]))]:[]
        let doc=WorkflowDocument(projectID:UUID().uuidString,name:"One shot",nodes:nodes)
        let version=try CanvasVersionJournal(root:root,projectID:doc.projectID).save(doc)
        return (root,.init(projectID:doc.projectID,projectName:doc.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Disabled",timing:.init(),hasEffects:effects))
    }
    private func savedRule(_ ledger:ScheduledRunLedger)async throws->ScheduleRule {let state=try await ledger.snapshot();return try XCTUnwrap(state.rules.first)}
    private func request(_ rule:ScheduleRule,client:String="owner",id:String=UUID().uuidString,digest:String=String(repeating:"a",count:64))->OwnedScheduleRequest {
        .init(client:client,project:rule.projectID,id:id,digest:digest,ruleID:rule.id,versionHash:rule.versionHash,revision:rule.revision)
    }
    func testAtomicClaimReplayRestartNeverRedispatchAndDoesNotEnableRecurrence()async throws {
        let (root,rule)=try fixture(),ledger=try ScheduledRunLedger(root:root)
        try await ledger.save(rule)
        let saved=try await savedRule(ledger),r=request(saved)
        let first=try await ledger.claimOwned(r,authorize:{_ in})
        XCTAssertTrue(first.execute)
        let duplicate=try await ledger.claimOwned(r,authorize:{_ in})
        XCTAssertFalse(duplicate.execute);XCTAssertEqual(first.job.id,duplicate.job.id)
        let state=try await ledger.snapshot();XCTAssertFalse(state.enabled);XCTAssertFalse(try XCTUnwrap(state.rules.first).enabled);XCTAssertEqual(state.jobs.count,1)
        await ledger.close()
        let reopened=try ScheduledRunLedger(root:root)
        let recovered=try await reopened.claimOwned(r,authorize:{_ in})
        XCTAssertFalse(recovered.execute);XCTAssertEqual(recovered.job.id,first.job.id);XCTAssertEqual(recovered.job.state,.failed)
        do {_=try await reopened.claimOwned(request(saved,id:r.id,digest:String(repeating:"b",count:64)),authorize:{_ in});XCTFail("Conflicting request redispatched")}catch{}
        do {_=try await reopened.replayOwned(r,authorize:{_ in throw MCPSettings.Refusal(code:"not_permitted")});XCTFail("Revoked duplicate leaked outcome")}catch{}
        await reopened.close()
    }
    func testOwnedCancellationCannotTouchNativeOrOtherClientAndTruthfullyRecordsRequest()async throws {
        let (root,rule)=try fixture(),ledger=try ScheduledRunLedger(root:root);try await ledger.save(rule)
        let saved=try await savedRule(ledger),claim=try await ledger.claimOwned(request(saved),authorize:{_ in})
        for client in ["other",""] {
            do {_=try await ledger.requestOwnedCancellation(claim.job.id,client:client,project:saved.projectID,authorize:{_ in});XCTFail("Foreign cancellation admitted")}catch{}
        }
        let requested=try await ledger.requestOwnedCancellation(claim.job.id,client:"owner",project:saved.projectID,authorize:{_ in})
        XCTAssertTrue(requested.cancellationRequested);XCTAssertEqual(requested.job.state,.claimed)
        let twice=try await ledger.requestOwnedCancellation(claim.job.id,client:"owner",project:saved.projectID,authorize:{_ in});XCTAssertEqual(requested,twice)
        try await ledger.finish(claim.job.id,state:.cancelled,message:"Joined harmless test")
        let terminal=try await ledger.requestOwnedCancellation(claim.job.id,client:"owner",project:saved.projectID,authorize:{_ in});XCTAssertEqual(terminal.job.state,.cancelled)
        let native=try await ledger.claimNow(saved.id)
        do {_=try await ledger.requestOwnedCancellation(native.id,client:"owner",project:saved.projectID,authorize:{_ in});XCTFail("Native job cancelled by client")}catch{}
        await ledger.close()
    }
    func testNativeOutputAuthorityRequiredAndInterruptionRetainsUncertainty()async throws {
        let (root,rule)=try fixture(effects:true),ledger=try ScheduledRunLedger(root:root);try await ledger.save(rule)
        var saved=try await savedRule(ledger)
        do {_=try await ledger.claimOwned(request(saved),authorize:{_ in});XCTFail("Unapproved outputs admitted")}catch{}
        let empty=try await ledger.snapshot();XCTAssertTrue(empty.jobs.isEmpty)
        saved.allowEffects=true;saved.outputAccessHash=try ScheduleSavedSource.outputAccessHash(root:root,projectID:saved.projectID)
        try await ledger.save(saved);saved=try await savedRule(ledger)
        try DiagnosticFileIO.writeAtomic(Data("{}".utf8),root:root,parts:["workflow-output-destinations",saved.projectID,"bindings.json"])
        do {_=try await ledger.claimOwned(request(saved),authorize:{_ in});XCTFail("Changed destination approval admitted")}catch{}
        try FileManager.default.removeItem(at:root.appendingPathComponent("workflow-output-destinations/"+saved.projectID+"/bindings.json"))
        let r=request(saved),claim=try await ledger.claimOwned(r,authorize:{_ in});await ledger.close()
        let reopened=try ScheduledRunLedger(root:root),replay=try await reopened.claimOwned(r,authorize:{_ in})
        XCTAssertFalse(replay.execute);XCTAssertEqual(replay.job.id,claim.job.id);XCTAssertEqual(replay.job.state,.needsReview)
        await reopened.close()
    }
    func testPersistenceFailureAndStaleRuleDoNotClaimJobs()async throws {
        let (root,rule)=try fixture(),ledger=try ScheduledRunLedger(root:root);try await ledger.save(rule)
        var saved=try await savedRule(ledger)
        do {_=try await ledger.claimOwned(request(rule),authorize:{_ in});XCTFail("Stale revision admitted")}catch{}
        saved.enabled=true;try await ledger.save(saved);saved=try await savedRule(ledger)
        do {_=try await ledger.claimOwned(request(saved),authorize:{_ in});XCTFail("Recurring rule admitted")}catch{}
        saved.enabled=false;try await ledger.save(saved);saved=try await savedRule(ledger)
        let folder=root.appendingPathComponent("scheduled-runs");try FileManager.default.setAttributes([.posixPermissions:0o500],ofItemAtPath:folder.path)
        defer {try? FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:folder.path)}
        do {_=try await ledger.claimOwned(request(saved),authorize:{_ in});XCTFail("Unwritable ledger admitted")}catch{}
        let state=try await ledger.snapshot();XCTAssertTrue(state.jobs.isEmpty);XCTAssertNil(state.ownedRequests)
        await ledger.close()
    }
}

extension OwnedScheduleRequestTests {
    @MainActor func testRealOwnedRunReplaysWithoutExecutingTwiceAndCancellationJoinsAdmission()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real development Python interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        let document=WorkflowDocument(projectID:UUID().uuidString,name:"Owned harmless source",nodes:[.init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])),.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(id:"edge",from:.init(address:source,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        try host.owner.saveCanvasVersion(label:"Owned one-shot")
        let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first),controller=ScheduledRunController(workspace:workspace,startTimer:false)
        try await controller.ready()
        controller.configureJobWorkspace={w in w.workflowSession?.executionLaunch={_,args,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,onLine,onExit)}}
        let rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Disabled",timing:.init())
        try await controller.save(rule)
        let saved=try XCTUnwrap(controller.state.rules.first),r=request(saved),first=try await controller.runOwned(r,authorize:{_ in})
        XCTAssertTrue(workspace.isBusy);XCTAssertFalse(controller.state.enabled)
        let duplicate=try await controller.runOwned(r,authorize:{_ in});XCTAssertEqual(duplicate.id,first.id)
        let deadline=Date().addingTimeInterval(60)
        while controller.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertNil(controller.activeJobID);XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(controller.state.jobs.first?.state,.completed,controller.error ?? controller.state.jobs.first?.message ?? "")
        let replay=try await controller.runOwned(r,authorize:{_ in});XCTAssertEqual(replay.id,first.id);XCTAssertEqual(controller.state.jobs.count,1);XCTAssertNil(controller.activeJobID)
        let next=try await controller.runOwned(request(saved),authorize:{_ in})
        let cancellation=try await controller.cancelOwned(next.id,client:"owner",project:saved.projectID,authorize:{_ in})
        XCTAssertEqual(cancellation.1,"requested");XCTAssertTrue(workspace.isBusy,"Cancellation request must retain admission until worker ownership joins")
        let cancelDeadline=Date().addingTimeInterval(60)
        while controller.activeJobID != nil && Date()<cancelDeadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertNil(controller.activeJobID);XCTAssertFalse(workspace.isBusy);XCTAssertEqual(controller.state.jobs.first?.state,.cancelled)
        XCTAssertEqual(host.owner.document,document);XCTAssertFalse(controller.state.rules.first?.enabled ?? true)
        let policy=OwnedScheduleTestPolicy()
        controller.configureJobWorkspace={w in
            policy.revoke()
            w.workflowSession?.executionLaunch={_,args,directory,onLine,onExit in
                policy.started();return try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,onLine,onExit)
            }
        }
        let revokedRequest=request(saved)
        _=try await controller.runOwned(revokedRequest,authorize:{_ in try policy.authorize()})
        let revokeDeadline=Date().addingTimeInterval(60)
        while controller.activeJobID != nil && Date()<revokeDeadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertNil(controller.activeJobID);XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(controller.state.jobs.first?.state,.blocked);XCTAssertEqual(policy.starts,0,"Revocation before host admission must not start a worker")
        do {_=try await controller.runOwned(revokedRequest,authorize:{_ in try policy.authorize()});XCTFail("Revoked replay exposed result")}catch{}
        await controller.shutdown()
    }
}

private final class OwnedScheduleTestPolicy:@unchecked Sendable {
    private let lock=NSLock()
    private var revoked=false,count=0,exits=0
    func revoke(){lock.lock();revoked=true;lock.unlock()}
    func authorize()throws {lock.lock();defer{lock.unlock()};if revoked {throw MCPSettings.Refusal(code:"not_permitted")}}
    func started(){lock.lock();count+=1;lock.unlock()}
    var starts:Int {lock.lock();defer{lock.unlock()};return count}
    func exited(){lock.lock();exits+=1;lock.unlock()}
    var exitCount:Int {lock.lock();defer{lock.unlock()};return exits}
}

extension OwnedScheduleRequestTests {
    func testDifferentClientsMayReuseUUIDAndPrunedJobReceiptNeverRedispatches()async throws {
        let (root,rule)=try fixture(),ledger=try ScheduledRunLedger(root:root)
        try await ledger.save(rule)
        let saved=try await savedRule(ledger),id=UUID().uuidString,firstRequest=request(saved,client:"one",id:id),secondRequest=request(saved,client:"two",id:id)
        let first=try await ledger.claimOwned(firstRequest,authorize:{_ in})
        try await ledger.finish(first.job.id,state:.completed,message:"First client completed")
        let second=try await ledger.claimOwned(secondRequest,authorize:{_ in})
        XCTAssertTrue(second.execute);XCTAssertNotEqual(first.job.id,second.job.id);XCTAssertNotEqual(first.job.occurrence,second.job.occurrence)
        try await ledger.finish(second.job.id,state:.completed,message:"Second client completed")
        let same=try await ledger.claimOwned(firstRequest,authorize:{_ in});XCTAssertFalse(same.execute);XCTAssertEqual(same.job.id,first.job.id)
        try await ledger.limits(days:1,maximum:10,now:Date().addingTimeInterval(3*86400))
        let pruned=try await ledger.snapshot();XCTAssertTrue(pruned.jobs.isEmpty);XCTAssertEqual(pruned.ownedRequests?.count,2)
        await ledger.close()
        let reopened=try ScheduledRunLedger(root:root)
        for (r,original) in [(firstRequest,first.job),(secondRequest,second.job)] {
            let replay=try await reopened.claimOwned(r,authorize:{_ in})
            XCTAssertFalse(replay.execute);XCTAssertEqual(replay.job.id,original.id);XCTAssertEqual(replay.job.state,.completed)
        }
        let retained=try await reopened.snapshot();XCTAssertTrue(retained.jobs.isEmpty)
        await reopened.close()
    }
}

extension OwnedScheduleRequestTests {
    @MainActor func testOutputAuthorityChangeDuringPreparationBlocksFinalAdmission()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let (root,rule)=try fixture(effects:true),workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let controller=ScheduledRunController(workspace:workspace,startTimer:false);try await controller.ready()
        var approved=rule;approved.allowEffects=true
        try await controller.save(approved)
        let saved=try XCTUnwrap(controller.state.rules.first),policy=OwnedScheduleTestPolicy()
        controller.configureJobWorkspace={w in
            do {try DiagnosticFileIO.writeAtomic(Data("{}".utf8),root:root,parts:["workflow-output-destinations",saved.projectID,"bindings.json"])}catch{XCTFail("Fixture mutation failed: \(error)")}
            w.workflowSession?.executionLaunch={_,_,_,_,_ in policy.started();throw WorkflowArchiveError("Unexpected worker start")}
        }
        _=try await controller.runOwned(request(saved),authorize:{_ in})
        let deadline=Date().addingTimeInterval(30)
        while controller.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(25))}
        XCTAssertNil(controller.activeJobID);XCTAssertEqual(policy.starts,0);XCTAssertFalse(workspace.isBusy)
        let job=try XCTUnwrap(controller.state.jobs.first);XCTAssertEqual(job.state,.blocked)
        XCTAssertTrue(job.message.contains("Output access changed during preparation"),job.message)
        await controller.shutdown()
    }
    @MainActor func testActiveWorkerRevocationRequestsCancellationAndJoinsBeforeRelease()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select the real development Python interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        let document=WorkflowDocument(projectID:UUID().uuidString,name:"Controlled cancellation",nodes:[.init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])),.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(id:"edge",from:.init(address:source,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        try host.owner.saveCanvasVersion(label:"Cancellation proof")
        let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first),controller=ScheduledRunController(workspace:workspace,startTimer:false),policy=OwnedScheduleTestPolicy()
        try await controller.ready()
        controller.configureJobWorkspace={w in
            // A real, deliberately waiting Python child establishes the active
            // cancellation boundary without depending on CSV processing speed.
            w.workflowSession?.executionLaunch={_,_,directory,onLine,onExit in
                policy.started()
                return try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),["-c","import time; print('owned-worker-ready', flush=True); time.sleep(30)"],directory,{line in
                    if line=="owned-worker-ready" {policy.revoke();return};onLine(line)
                },{code,stderr in policy.exited();onExit(code,stderr)})
            }
        }
        let rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Disabled",timing:.init())
        try await controller.save(rule)
        _=try await controller.runOwned(request(try XCTUnwrap(controller.state.rules.first)),authorize:{_ in try policy.authorize()})
        let deadline=Date().addingTimeInterval(15)
        while controller.activeJobID != nil && Date()<deadline {
            XCTAssertTrue(workspace.isBusy,"Admission released while the job still owns its child")
            try await Task.sleep(for:.milliseconds(25))
        }
        XCTAssertNil(controller.activeJobID);XCTAssertFalse(workspace.isBusy);XCTAssertEqual(policy.starts,1);XCTAssertEqual(policy.exitCount,1)
        XCTAssertEqual(controller.state.jobs.first?.state,.cancelled)
        await controller.shutdown()
    }
}
