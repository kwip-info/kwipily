import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ScheduledRunTests:XCTestCase {
    private func date(_ text:String)->Date {ISO8601DateFormatter().date(from:text)!}
    func testDailyDSTGapAndRepeatedHourRunOnce()throws {
        var timing=ScheduleTiming(kind:.daily,timeZone:"America/Chicago",hour:2,minute:30)
        XCTAssertEqual(try timing.next(after:date("2026-03-08T07:59:00Z")),date("2026-03-08T08:00:00Z"))
        timing.hour=1
        let first=try XCTUnwrap(timing.next(after:date("2026-11-01T05:00:00Z")))
        XCTAssertEqual(first,date("2026-11-01T06:30:00Z"))
        XCTAssertEqual(try timing.next(after:first),date("2026-11-02T07:30:00Z"))
        timing.timeZone="Not/AZone";XCTAssertThrowsError(try timing.next(after:first))
    }
    func testIntervalAdvancesWithoutBacklogAndOnceDoesNotRepeat()throws {
        let anchor=date("2026-10-19T10:00:00Z")
        var timing=ScheduleTiming(kind:.interval,anchor:anchor,intervalMinutes:15)
        XCTAssertEqual(try timing.next(after:anchor.addingTimeInterval(3600)),anchor.addingTimeInterval(4500))
        timing.kind = .once
        XCTAssertEqual(try timing.next(after:anchor.addingTimeInterval(-1)),anchor)
        XCTAssertNil(try timing.next(after:anchor))
    }
    func testDurableClaimDeduplicatesAndRestartPausesPossibleWrites()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let now=date("2026-10-19T10:00:00Z")
        let ledger=try ScheduledRunLedger(root:root)
        var rule=ScheduleRule(projectID:UUID().uuidString,projectName:"Orders",versionID:UUID().uuidString,versionNumber:1,versionHash:String(repeating:"a",count:64),name:"Orders",timing:.init(kind:.interval,anchor:now,intervalMinutes:1))
        rule.enabled=true;rule.hasEffects=true;rule.allowEffects=true
        try await ledger.save(rule,now:now.addingTimeInterval(-10));try await ledger.setEnabled(true,now:now.addingTimeInterval(-1))
        let claim=try await ledger.tick(now:now,busy:false)
        XCTAssertNotNil(claim)
        let again=try await ledger.tick(now:now,busy:false);XCTAssertNil(again)
        await ledger.close()
        let restarted=try ScheduledRunLedger(root:root)
        let state=try await restarted.snapshot()
        XCTAssertEqual(state.jobs.count,1);XCTAssertEqual(state.jobs[0].state,.needsReview)
        XCTAssertFalse(state.rules[0].enabled)
        await restarted.close()
    }
    private func withLedger(_ body:(ScheduledRunLedger,Date)async throws->Void)async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let ledger=try ScheduledRunLedger(root:root)
        do {try await body(ledger,date("2026-10-19T10:00:00Z"));await ledger.close();try FileManager.default.removeItem(at:root)}
        catch {await ledger.close();try? FileManager.default.removeItem(at:root);throw error}
    }
    private func rule(_ now:Date)->ScheduleRule {ScheduleRule(projectID:UUID().uuidString,projectName:"Test",versionID:UUID().uuidString,versionNumber:1,versionHash:String(repeating:"a",count:64),name:"Test",timing:.init(kind:.interval,anchor:now,intervalMinutes:1),enabled:true)}
    func testLatestQueueReplacesOlderOccurrenceAndNeverOverlaps()async throws {
        try await withLedger {ledger,now in
            var r=self.rule(now);r.busy = .latest;r.missed = .latest
            try await ledger.save(r,now:now.addingTimeInterval(-1));try await ledger.setEnabled(true,now:now.addingTimeInterval(-1))
            let first=try await ledger.tick(now:now,busy:true);XCTAssertNil(first)
            let second=try await ledger.tick(now:now.addingTimeInterval(60),busy:true);XCTAssertNil(second)
            let state=try await ledger.snapshot();XCTAssertEqual(state.jobs.filter{$0.state == .queued}.count,1);XCTAssertEqual(state.jobs.filter{$0.state == .skipped}.count,1)
            let claim=try await ledger.tick(now:now.addingTimeInterval(61),busy:false)
            XCTAssertEqual(claim?.dueAt,now.addingTimeInterval(60))
            let overlap=try await ledger.tick(now:now.addingTimeInterval(62),busy:false);XCTAssertNil(overlap)
        }
    }
    func testUnavailableGapRecordsOneMissedWindowAndHonorsSkip()async throws {
        try await withLedger {ledger,now in
            let r=self.rule(now);try await ledger.save(r,now:now.addingTimeInterval(-1));try await ledger.setEnabled(true,now:now.addingTimeInterval(-1))
            let captured=try await ledger.tick(now:now,busy:false)
            let first=try XCTUnwrap(captured)
            try await ledger.finish(first.id,state:.completed,message:"Done",now:now)
            let late=try await ledger.tick(now:now.addingTimeInterval(600),busy:false);XCTAssertNil(late)
            let state=try await ledger.snapshot();XCTAssertEqual(state.jobs.count,2);XCTAssertEqual(state.jobs.first?.state,.missed)
            XCTAssertEqual(state.rules.first?.nextDue,now.addingTimeInterval(660))
            try await ledger.setEnabled(false,now:now.addingTimeInterval(601))
            try await ledger.setEnabled(true,now:now.addingTimeInterval(1200))
            let resume=try await ledger.tick(now:now.addingTimeInterval(1200),busy:false);XCTAssertNil(resume)
        }
    }
    func testExplicitShortSleepHonorsMissedPolicyWithoutTimerGrace()async throws {
        try await withLedger {ledger,now in
            var r=self.rule(now);r.timing.kind = .once;r.timing.anchor=now.addingTimeInterval(2)
            try await ledger.save(r,now:now);try await ledger.setEnabled(true,now:now)
            await ledger.markUnavailable()
            let claim=try await ledger.tick(now:now.addingTimeInterval(3),busy:false)
            XCTAssertNil(claim)
            let snapshot=try await ledger.snapshot();XCTAssertEqual(snapshot.jobs.first?.state,.missed)
        }
    }
    func testRetentionProtectsUnresolvedOutcomesAndRuleRemovalPreservesJobs()async throws {
        try await withLedger {ledger,now in
            var effects=self.rule(now);effects.hasEffects=true;effects.allowEffects=true
            try await ledger.save(effects,now:now.addingTimeInterval(-1))
            let unknown=try await ledger.claimNow(effects.id,now:now)
            try await ledger.finish(unknown.id,state:.needsReview,message:"Unknown",now:now)
            let safe=self.rule(now);try await ledger.save(safe,now:now.addingTimeInterval(-1));try await ledger.limits(days:30,maximum:10,now:now)
            for i in 1...12 {let job=try await ledger.claimNow(safe.id,now:now.addingTimeInterval(Double(i)));try await ledger.finish(job.id,state:.completed,message:"Done",now:now.addingTimeInterval(Double(i)))}
            try await ledger.remove(safe.id)
            let result=try await ledger.snapshot();XCTAssertEqual(result.jobs.count,10);XCTAssertTrue(result.jobs.contains{$0.id==unknown.id && !$0.reviewed});XCTAssertFalse(result.rules.contains{$0.id==safe.id})
            do {try await ledger.save(effects,now:now);XCTFail("Unreviewed rule cannot be enabled")}catch{}
        }
    }
    func testProjectAndGlobalDraftsCannotOverwriteAnotherSavedEdit()async throws {
        try await withLedger {ledger,now in
            let original=self.rule(now);try await ledger.save(original,now:now.addingTimeInterval(-1))
            let state=try await ledger.snapshot();var projectDraft=try XCTUnwrap(state.rules.first);var globalDraft=projectDraft
            projectDraft.name="Project edit";try await ledger.save(projectDraft,now:now)
            globalDraft.name="Stale global edit"
            do {try await ledger.save(globalDraft,now:now);XCTFail("Stale draft overwrote another view")}catch{}
            let updated=try await ledger.snapshot();XCTAssertEqual(updated.rules.first?.name,"Project edit")
        }
    }
    func testIdleEnabledPollingDoesNotRewriteMetadata()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let ledger=try ScheduledRunLedger(root:root),now=Date()
        var r=rule(now);r.timing.anchor=now.addingTimeInterval(3600)
        try await ledger.save(r,now:now);try await ledger.setEnabled(true,now:now)
        let file=root.appendingPathComponent("scheduled-runs/state.json")
        let before=try FileManager.default.attributesOfItem(atPath:file.path)[.systemFileNumber] as? NSNumber
        for i in 1...10 {let claim=try await ledger.tick(now:now.addingTimeInterval(Double(i)),busy:false);XCTAssertNil(claim)}
        let after=try FileManager.default.attributesOfItem(atPath:file.path)[.systemFileNumber] as? NSNumber
        XCTAssertEqual(before,after,"Idle scheduling must not replace the ledger every tick")
        await ledger.close()
    }
    func testWeeklyAndOutputConsentValidation()throws {
        let now=date("2026-10-19T13:29:00Z")
        let timing=ScheduleTiming(kind:.weekly,timeZone:"America/Chicago",hour:8,minute:30,weekday:2)
        XCTAssertEqual(try timing.next(after:now),date("2026-10-19T13:30:00Z"))
        var r=rule(now);r.hasEffects=true;XCTAssertThrowsError(try r.validate());r.allowEffects=true;XCTAssertNoThrow(try r.validate())
    }

}

@MainActor final class ScheduledRunIntegrationTests:XCTestCase {
    func testScheduledSourceSelectionsStayInMemory()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try TableOutputPracticeExample.document()
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let owner=try XCTUnwrap(workspace.workflowSession?.owner),address=WorkflowNodeAddress(domain:.data,localID:"source")
        let regular=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        XCTAssertTrue(regular.setPath("/original.csv",for:address,lease:owner.lease(),expectedRevision:regular.revision))
        let file=root.appendingPathComponent("workflow-sources/"+document.projectID+"/bindings.json"),before=try Data(contentsOf:file)
        let isolated=try WorkflowPrivateSourceBindings(owner:owner,directory:root,ephemeral:true)
        XCTAssertTrue(isolated.setPath("/captured-api.arrow",for:address,lease:owner.lease(),expectedRevision:isolated.revision))
        XCTAssertEqual(isolated.path(for:address),"/captured-api.arrow");XCTAssertEqual(regular.path(for:address),"/original.csv")
        XCTAssertEqual(try Data(contentsOf:file),before)
    }
    func testActualSavedVersionRunsWithoutMutatingOpenCanvasAndCaptureOffIsHonored()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else{throw XCTSkip("Select the real development Python interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        let document=WorkflowDocument(projectID:UUID().uuidString,name:"Pinned schedule canvas",nodes:[.init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])),.init(address:view,x:250,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(id:"edge",from:.init(address:source,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        _ = try w.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(w.beginWorkflowProject(id:document.projectID),w.statusMessage)
        let host=try XCTUnwrap(w.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        try host.owner.saveCanvasVersion(label:"For schedule")
        let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first)
        try host.owner.setRunHistoryPolicy(.none)
        XCTAssertTrue(host.owner.rename("Current unsaved canvas name",lease:host.owner.lease()))
        let current=host.owner.document,selection=host.owner.selection
        let c=ScheduledRunController(workspace:w,startTimer:false);try await c.ready()
        c.configureJobWorkspace={workspace in workspace.workflowSession?.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}}
        let rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Review schedule",timing:.init(kind:.once,anchor:Date().addingTimeInterval(60)))
        try await c.save(rule);await c.runNow(rule.id)
        XCTAssertNotNil(c.activeJobID);XCTAssertTrue(w.isBusy,"Other editor owners must respect scheduled admission")
        let deadline=Date().addingTimeInterval(60)
        while c.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let job=try XCTUnwrap(c.state.jobs.first)
        XCTAssertNil(c.activeJobID,c.error ?? c.phase);XCTAssertEqual(job.state,.completed,job.message)
        XCTAssertEqual(job.rule.versionHash,version.hash);XCTAssertEqual(job.captureDescription,"Optional output capture off");XCTAssertTrue(job.activity.isEmpty)
        XCTAssertEqual(host.owner.document,current);XCTAssertEqual(host.owner.selection,selection);XCTAssertFalse(w.isBusy);XCTAssertFalse(c.state.enabled)
        let loadedHistory=await c.preparation(for:job)
        let history=try XCTUnwrap(loadedHistory)
        XCTAssertEqual(history.1.document.name,document.name)
        guard case .workflow(let stored)=try w.projectLibrary.loadSource(id:document.projectID) else{return XCTFail("Missing original canvas")}
        XCTAssertEqual(stored,current)
        var recurring=try XCTUnwrap(c.state.rules.first);recurring.enabled=true;recurring.timing.anchor=Date().addingTimeInterval(5)
        try await c.save(recurring);await c.setEnabled(true)
        await c.tick(now:recurring.timing.anchor)
        XCTAssertNotNil(c.activeJobID,"The due path must dispatch through the same run authority")
        let dueDeadline=Date().addingTimeInterval(60)
        while c.activeJobID != nil && Date()<dueDeadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertEqual(c.state.jobs.count,2);XCTAssertEqual(c.state.jobs.first?.trigger,"Scheduled");XCTAssertEqual(c.state.jobs.first?.state,.completed,c.error ?? "")
        XCTAssertEqual(host.owner.document,current);XCTAssertFalse(w.isBusy)
        await c.setEnabled(false);await c.shutdown()
    }
    func testSavedNamedTextTrainRetainsVerifiedModelWithCaptureOff()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else{throw XCTSkip("Select the real development Python interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try TextEncodingPracticeExample.trainingDocument()
        _ = try w.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(w.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(w.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("Sources/KwipMLApp/Resources/text-training-observations.csv").path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        try host.owner.setRunHistoryPolicy(.none);try host.owner.saveCanvasVersion(label:"Scheduled training")
        let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first),before=host.owner.document
        let c=ScheduledRunController(workspace:w,startTimer:false);try await c.ready()
        c.configureJobWorkspace={workspace in workspace.workflowSession?.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}}
        let rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Text training",timing:.init(),target:WorkflowExamples.trainAddress)
        try await c.save(rule);await c.runNow(rule.id)
        let deadline=Date().addingTimeInterval(90)
        while c.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let job=try XCTUnwrap(c.state.jobs.first)
        XCTAssertNil(c.activeJobID);XCTAssertEqual(job.state,.completed,job.message);XCTAssertTrue(job.activity.isEmpty)
        XCTAssertEqual(host.owner.document,before);XCTAssertFalse(w.isBusy)
        let run=try XCTUnwrap(job.runID),id=try XCTUnwrap(UUID(uuidString:run))
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:id)
        guard case .verified(let history)=WorkflowHistoryReader.load(reference,root:root) else {await c.shutdown();return XCTFail("Scheduled training must retain verifiable model History")}
        XCTAssertEqual(history.plan.namedInputPlan?.inputs.first?.name,"tokens")
        XCTAssertNotNil(history.modelContent);XCTAssertNil(history.modelContent?.trainingProgress)
        await c.shutdown()
    }
}

@MainActor final class ScheduledRunFailureTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testUncertainAPIWritePausesRuleAndAcknowledgmentDoesNotRetryOrResume()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else{throw XCTSkip("Select the development Python interpreter.")}
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var document=try TableOutputPracticeExample.document()
        let address=WorkflowNodeAddress(domain:.data,localID:"table-output")
        let index=try XCTUnwrap(document.nodes.firstIndex{$0.address==address})
        var config=APITableSettings.defaults;config["direction"] = .string("output");config["method"] = .string("POST");config["url"] = .string("https://api.example.com/orders");config["mapping"] = .array([.object(["source":.string("order_id"),"target":.string("/id"),"type":.string("string")])])
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.outputDelimited",config:["destination":.string("api"),"api":.object(config),"filename":.string("old.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false])
        _ = try w.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(w.beginWorkflowProject(id:document.projectID),w.statusMessage)
        let host=try XCTUnwrap(w.workflowSession)
        XCTAssertTrue(host.bindings.setPath(try TableOutputPracticeExample.resourceURL().path,for:.init(domain:.data,localID:"source"),lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        let credentials=ScheduleMemoryCredentials();host.outputDestinations.databaseOutputs.credentials=credentials
        try host.outputDestinations.databaseOutputs.saveProfile(name:"Synthetic fixture",config:config,username:"",password:"",allowWrites:true,address:address)
        try host.owner.saveCanvasVersion();let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first)
        let c=ScheduledRunController(workspace:w,startTimer:false);try await c.ready()
        var calls=0
        c.configureJobWorkspace={workspace in
            workspace.workflowSession?.executionLaunch={_,args,dir,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)}
            workspace.workflowSession?.outputDestinations.databaseOutputs.credentials=credentials
            workspace.workflowSession?.outputDestinations.databaseOutputs.apiExecute={_,_,_,id,_,hash,_ in calls+=1;return ["schemaVersion":1,"kind":.string("apiWrite"),"operationID":.string(id),"status":.string("unknown"),"sourceSHA256":.string(hash),"inputRows":12,"acceptedRows":4,"requests":.array([])]}
        }
        var rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Uncertain write fixture",timing:.init(kind:.interval),enabled:true)
        do {try await c.save(rule);XCTFail("Unattended output requires explicit consent")}catch{}
        rule.allowEffects=true;try await c.save(rule);await c.runNow(rule.id)
        let deadline=Date().addingTimeInterval(50)
        while c.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let job=try XCTUnwrap(c.state.jobs.first)
        XCTAssertEqual(job.state,.needsReview,job.message);XCTAssertEqual(calls,1);XCTAssertFalse(c.state.rules.first?.enabled ?? true)
        let receipts=try await c.receipts(for:job);XCTAssertEqual(receipts.count,1);XCTAssertFalse(receipts[0].acknowledged)
        try await c.review(job)
        let reviewed=try await c.receipts(for:job);XCTAssertTrue(reviewed[0].acknowledged);XCTAssertEqual(calls,1);XCTAssertFalse(c.state.rules.first?.enabled ?? true)
        await c.tick(now:Date().addingTimeInterval(1200));XCTAssertEqual(calls,1)
        await c.shutdown()
    }
    func testChangedPrivateDestinationCannotUseEarlierUnattendedConsent()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let document=try TableOutputPracticeExample.document()
        _ = try w.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(w.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(w.workflowSession);try host.owner.saveCanvasVersion(label:"Before destination change")
        let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first)
        let c=ScheduledRunController(workspace:w,startTimer:false);try await c.ready()
        let rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Destination consent",timing:.init(),allowEffects:true)
        try await c.save(rule)
        let parts=["workflow-output-destinations",document.projectID,"bindings.json"]
        try FileManager.default.createDirectory(at:root.appendingPathComponent(parts.dropLast().joined(separator:"/")),withIntermediateDirectories:true)
        try DiagnosticFileIO.writeAtomic(Data("{}".utf8),root:root,parts:parts)
        await c.runNow(rule.id)
        let deadline=Date().addingTimeInterval(10)
        while c.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        let job=try XCTUnwrap(c.state.jobs.first)
        XCTAssertEqual(job.state,.blocked);XCTAssertTrue(job.message.contains("changed since"),job.message);XCTAssertTrue(job.receipts.isEmpty);XCTAssertFalse(w.isBusy)
        await c.shutdown()
    }
    func testMissingSourceAndCancellationReleaseAdmissionWithoutStartingOutputs()async throws {
        for cancel in [false,true] {
            let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
            let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
            var document=try TableOutputPracticeExample.document();document.nodes.removeAll{$0.address.localID=="table-output"};document.edges=[]
            _ = try w.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(w.beginWorkflowProject(id:document.projectID))
            let v=try XCTUnwrap(w.workflowSession?.owner.canvasVersions?.versions.first)
            let c=ScheduledRunController(workspace:w,startTimer:false);try await c.ready()
            let rule=ScheduleRule(projectID:document.projectID,projectName:document.name,versionID:v.id,versionNumber:v.number,versionHash:v.hash,name:"Missing source",timing:.init())
            try await c.save(rule);await c.runNow(rule.id)
            if cancel,let id=c.activeJobID {c.cancel(id)}
            let deadline=Date().addingTimeInterval(30)
            while c.activeJobID != nil && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
            XCTAssertNil(c.activeJobID,c.phase);XCTAssertEqual(c.state.jobs.first?.state,cancel ? .cancelled:.blocked,c.state.jobs.first?.message ?? "No job")
            XCTAssertFalse(w.isBusy);XCTAssertFalse(ScheduledRunAdmission.isBusy(directory:root));XCTAssertTrue(c.state.jobs.first?.receipts.isEmpty ?? false)
            await c.shutdown();try FileManager.default.removeItem(at:root)
        }
    }
}
private final class ScheduleMemoryCredentials:DatabaseSourceCredentialStorage {
    var values:[String:DatabaseSourceCredentials]=[:]
    func read(projectID:String,nodeID:String,connectionID:String)throws->DatabaseSourceCredentials? {values[projectID+nodeID+connectionID]}
    func write(_ value:DatabaseSourceCredentials,projectID:String,nodeID:String,connectionID:String)throws {values[projectID+nodeID+connectionID]=value}
    func clear(projectID:String,nodeID:String,connectionID:String)throws {values[projectID+nodeID+connectionID]=nil}
}
