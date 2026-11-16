import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class MCPAutomationReadTests:XCTestCase {
    func testAutomationReadIsExplicitDefaultDeniedPermission()throws {
        let action=try XCTUnwrap(MCPAction(rawValue:"automation_read"))
        XCTAssertTrue(action.isReadOnly)
        XCTAssertTrue(action.requiresProject)
        XCTAssertFalse(MCPProjectGrant.createdProject.actions.contains(action))
        XCTAssertFalse(MCPSettings.factory.visibleActions(clientID:"old").contains(action))
    }
}

extension MCPAutomationReadTests {
    struct Evidence {
        let root:URL,project:String,other:String,version:CanvasVersionJournal.Version,rule:ScheduleRule
    }
    func evidence()throws->Evidence {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,other=UUID().uuidString
        let doc=WorkflowDocument(projectID:project,name:"PRIVATE source path /secret",nodes:[.init(address:.init(domain:.data,localID:"source"),name:"PRIVATE node",payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:]))])
        let version=try CanvasVersionJournal(root:root,projectID:project).save(doc,label:"PRIVATE saved label")
        let rule=ScheduleRule(projectID:project,projectName:"PRIVATE project",versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"PRIVATE rule",timing:.init(),outputAccessHash:String(repeating:"b",count:64))
        return .init(root:root,project:project,other:other,version:version,rule:rule)
    }
    func write<T:Encodable>(_ value:T,root:URL,folder:String)throws {
        try DiagnosticFileIO.writeAtomic(try JSONEncoder().encode(value),root:root,parts:[folder,"state.json"])
    }
    func testSharedReadFiltersProjectAndNeverRecoversOrDisclosesSensitiveLedgerFields()async throws {
        let e=try evidence()
        var foreign=e.rule;foreign.id=UUID().uuidString;foreign.projectID=e.other
        var first=ScheduledJob(occurrence:"one",rule:e.rule,dueAt:Date(),state:.running)
        first.message="PRIVATE SQL /secret";first.activity=["PRIVATE payload"];first.phase="PRIVATE";first.captureDescription="PRIVATE";first.effectivePolicy=["PRIVATE":true]
        let other=ScheduledJob(occurrence:"other",rule:foreign,dueAt:Date(),state:.completed)
        var scheduled=ScheduledRunState();scheduled.rules=[foreign,e.rule];scheduled.jobs=[other,first]
        try write(scheduled,root:e.root,folder:"scheduled-runs")
        var endpoint=WebhookEndpoint(rule:e.rule);endpoint.id=e.rule.id;endpoint.name="PRIVATE endpoint"
        var delivery=WebhookDelivery(endpoint:endpoint,deliveryID:"PRIVATE delivery",digest:String(repeating:"c",count:64))
        delivery.state = .running;delivery.message="PRIVATE";delivery.payload=Data("PRIVATE payload".utf8);delivery.payloadSHA256=delivery.payload.map(WorkflowArchiveCodec.hash);delivery.preview=Data("PRIVATE preview".utf8)
        var webhooks=WebhookState();webhooks.endpoints=[endpoint];webhooks.deliveries=[delivery]
        try write(webhooks,root:e.root,folder:"webhooks")
        let scheduleURL=e.root.appendingPathComponent("scheduled-runs/state.json"),hookURL=e.root.appendingPathComponent("webhooks/state.json")
        let before=try Data(contentsOf:scheduleURL),beforeHook=try Data(contentsOf:hookURL)
        for family in AutomationReadSnapshot.Family.allCases {
            let snapshot=try await AutomationReadSnapshot.load(root:e.root,projectID:e.project,family:family)
            XCTAssertEqual(snapshot.items.count,1)
            let bytes=try JSONEncoder().encode(snapshot.items.map(\.record)),text=String(decoding:bytes,as:UTF8.self)
            for privateText in ["PRIVATE",e.other,endpoint.credentialID,endpoint.rule.outputAccessHash!,"/secret"] {XCTAssertFalse(text.contains(privateText),family.rawValue)}
            XCTAssertEqual(snapshot.items.first?.record.projectID,e.project)
            XCTAssertEqual(snapshot.items.first?.components,["kwip.data.sourceCSV"])
            if family == .jobs || family == .deliveries {XCTAssertEqual(snapshot.items.first?.record.state,"running","Read must not perform interruption recovery")}
        }
        XCTAssertEqual(try Data(contentsOf:scheduleURL),before)
        XCTAssertEqual(try Data(contentsOf:hookURL),beforeHook)
        XCTAssertFalse(FileManager.default.fileExists(atPath:e.root.appendingPathComponent("webhooks/owner.lock").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:e.root.appendingPathComponent("scheduled-runs/owner.lock").path))
    }
    func testStablePagingOwnershipRevocationExpiryAndMissingPins()async throws {
        let e=try evidence(),access=MCPAutomationReadAccess()
        var clock=Date();access.now={clock}
        var state=ScheduledRunState();state.rules=(0..<3).map{n in var r=e.rule;r.id=UUID().uuidString;r.name="Rule \(n)";return r}
        var foreign=e.rule;foreign.id=UUID().uuidString;foreign.projectID=e.other
        state.rules.insert(foreign,at:0)
        var missing=e.rule;missing.id=UUID().uuidString;missing.versionID=UUID().uuidString;state.rules.append(missing)
        try write(state,root:e.root,folder:"scheduled-runs")
        var permitted=true,disabled=false
        let authorize:(Set<String>)throws->Void={components in if !permitted || (disabled && components.contains("kwip.data.sourceCSV")){throw MCPSettings.Refusal(code:"not_permitted")}}
        let args:[String:JSONValue]=["family":.string("schedules"),"limit":1]
        let first=try await access.read(client:"one",project:e.project,root:e.root,args:args,authorize:authorize)
        XCTAssertTrue(first["scope"]?.stringValue?.contains("subset") == true)
        var next=args;next["snapshotID"]=first["snapshotID"];next["cursorID"]=first["nextCursorID"]
        state.rules=[];try write(state,root:e.root,folder:"scheduled-runs")
        let second=try await access.read(client:"one",project:e.project,root:e.root,args:next,authorize:authorize)
        var inspect:[String:JSONValue]=["family":.string("schedules"),"snapshotID":try XCTUnwrap(first["snapshotID"])]
        if case .array(let rows)=first["records"],case .object(let row)=rows.first {inspect["recordID"]=row["id"]}
        let detail=try await access.read(client:"one",project:e.project,root:e.root,args:inspect,authorize:authorize)
        XCTAssertEqual(detail["records"],first["records"],"Metadata get must use the pinned snapshot")
        XCTAssertNotEqual(first["records"],second["records"])
        guard case .array(let rows)=second["records"] else{return XCTFail("No page")};XCTAssertEqual(rows.count,1)
        for (client,project) in [("two",e.project),("one",e.other)] {
            do {_=try await access.read(client:client,project:project,root:e.root,args:next,authorize:authorize);XCTFail("Foreign handle")}
            catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"automation_snapshot_unavailable")}
        }
        disabled=true
        do {_=try await access.read(client:"one",project:e.project,root:e.root,args:next,authorize:authorize);XCTFail("Component revoked")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"not_permitted")}
        disabled=false;permitted=false
        do {_=try await access.read(client:"one",project:e.project,root:e.root,args:next,authorize:authorize);XCTFail("Grant revoked")}catch{}
        permitted=true;clock=clock.addingTimeInterval(301)
        do {_=try await access.read(client:"one",project:e.project,root:e.root,args:next,authorize:authorize);XCTFail("Expired")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"automation_snapshot_unavailable")}
    }
    func testRepeatedJobsAuthorizeDistinctCapturedComponentsRatherThanEveryRow()async throws {
        let e=try evidence(),access=MCPAutomationReadAccess()
        var state=ScheduledRunState()
        state.jobs=(0..<2000).map{ScheduledJob(occurrence:"job-\($0)",rule:e.rule,dueAt:Date(),state:.completed)}
        try write(state,root:e.root,folder:"scheduled-runs")
        var checks=0
        let authorize:(Set<String>)throws->Void={_ in checks+=1}
        let page=try await access.read(client:"one",project:e.project,root:e.root,args:["family":.string("jobs"),"limit":100],authorize:authorize)
        XCTAssertLessThanOrEqual(checks,4,"Repeated same-source jobs must not repeat main-actor policy validation")
        guard case .array(let rows)=page["records"] else{return XCTFail("Missing rows")}
        XCTAssertEqual(rows.count,100)
        checks=0
        _=try await access.read(client:"one",project:e.project,root:e.root,args:["family":.string("jobs"),"snapshotID":try XCTUnwrap(page["snapshotID"]),"cursorID":try XCTUnwrap(page["nextCursorID"]),"limit":100],authorize:authorize)
        XCTAssertEqual(checks,2,"Each page still rechecks current grant and captured component union")
    }
    func testAbsentAndInvalidSourcesAreDistinctAndReadDoesNotCreateStorage()async throws {
        let e=try evidence(),access=MCPAutomationReadAccess()
        let first=try await access.read(client:"one",project:e.project,root:e.root,args:["family":.string("jobs")],authorize:{_ in})
        XCTAssertEqual(first["sourceStatus"],.string("notRecorded"))
        XCTAssertFalse(FileManager.default.fileExists(atPath:e.root.appendingPathComponent("scheduled-runs").path))
        try DiagnosticFileIO.writeAtomic(Data("PRIVATE invalid".utf8),root:e.root,parts:["scheduled-runs","state.json"])
        do {_=try await access.read(client:"one",project:e.project,root:e.root,args:["family":.string("jobs")],authorize:{_ in});XCTFail("Corrupt became empty success")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"automation_source_unavailable")}
        let badArguments:[[String:JSONValue]] = [["family":.string("jobs"),"limit":101],["family":.string("jobs"),"cursorID":1],["family":.string("jobs"),"payload":true],["family":.string("unknown")]]
        let target=e.root.appendingPathComponent("private-outside.json")
        let secret=Data("PRIVATE outside".utf8);try secret.write(to:target)
        let ledger=e.root.appendingPathComponent("scheduled-runs/state.json")
        try FileManager.default.removeItem(at:ledger)
        try FileManager.default.createSymbolicLink(at:ledger,withDestinationURL:target)
        do {_=try await access.read(client:"one",project:e.project,root:e.root,args:["family":.string("jobs")],authorize:{_ in});XCTFail("Symlink ledger accepted")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"automation_source_unavailable")}
        XCTAssertEqual(try Data(contentsOf:target),secret)
        for bad in badArguments {
            do {_=try await access.read(client:"one",project:e.project,root:e.root,args:bad,authorize:{_ in});XCTFail("Invalid arguments accepted")}
            catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"automation_invalid_arguments")}
        }
    }
}

extension MCPWorkflowGatewayTests {
    func testAutomationReadDiscoveryAndExplicitPermissionDoNotOpenProject()async throws {
        let f=try fixture()
        var settings=f.controller.settings
        settings.clients[0].actions.remove(.automationRead);settings.projects[f.owner.projectID]?.actions.remove(.automationRead)
        try f.controller.save(settings)
        do {_=try await call(f,.automationRead,["family":.string("jobs")]);XCTFail("Default denied")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"not_permitted")}
        settings.clients[0].actions.insert(.automationRead);settings.projects[f.owner.projectID]?.actions.insert(.automationRead);try f.controller.save(settings)
        let page=try await call(f,.automationRead,["family":.string("jobs")])
        XCTAssertEqual(page["sourceStatus"],.string("notRecorded"))
        XCTAssertEqual(page["serviceReadiness"],.string("notObserved"))
        let unopened=WorkflowDocument(projectID:UUID().uuidString,name:"Unopened")
        _=try f.windows.restorationWorkspace.projectLibrary.saveSource(.workflow(unopened))
        settings.projects[unopened.projectID] = .init(actions:[.automationRead]);try f.controller.save(settings)
        let openIDs=Set(f.windows.openControllers.map{$0.session.workspace.workflowSession?.owner.projectID})
        _=try await f.gateway.dispatch(clientID:f.client,action:"automation_read",arguments:["projectID":.string(unopened.projectID),"family":.string("versions")])
        XCTAssertEqual(Set(f.windows.openControllers.map{$0.session.workspace.workflowSession?.owner.projectID}),openIDs,"Metadata reads must not open the target")
        let tools=try await f.gateway.dispatch(clientID:f.client,action:"tools_list",arguments:[:])
        let text=String(decoding:try JSONEncoder().encode(tools),as:UTF8.self)
        XCTAssertTrue(text.contains("automation_read"));XCTAssertTrue(text.contains("default-denied"))
        settings.projects[f.owner.projectID]?.actions.remove(.automationRead);try f.controller.save(settings)
        do {_=try await call(f,.automationRead,["family":.string("jobs"),"snapshotID":try XCTUnwrap(page["snapshotID"])]);XCTFail("Revoked")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"not_permitted")}
    }
}

extension MCPAutomationReadTests {
    func testSavedTargetChoicesUseExactOldVersionAndExcludeNamesConfigurationsAndDeniedComponents()async throws {
        let e=try evidence(),journal=try CanvasVersionJournal(root:e.root,projectID:e.project)
        var old=WorkflowDocument(projectID:e.project,name:"PRIVATE title",nodes:[.init(address:.init(domain:.control,localID:"old-train"),name:"PRIVATE node",payload:.train(config:[:]))])
        old.schemaVersion=5
        let oldVersion=try journal.save(old)
        var current=old;current.nodes=[];current.revision+=1
        let latest=try journal.save(current)
        let snapshot=try await AutomationReadSnapshot.load(root:e.root,projectID:e.project,family:.versions)
        let oldRecord=try XCTUnwrap(snapshot.items.first(where:{$0.record.versionID==oldVersion.id})?.record)
        XCTAssertEqual(oldRecord.targetChoices?.map(\.address),[.init(domain:.control,localID:"old-train")])
        XCTAssertEqual(oldRecord.targetChoices?.map(\.kind),["train"])
        XCTAssertEqual(snapshot.items.first(where:{$0.record.versionID==latest.id})?.record.targetChoices?.count,0)
        let text=String(decoding:try JSONEncoder().encode(oldRecord),as:UTF8.self)
        XCTAssertFalse(text.contains("PRIVATE"));XCTAssertFalse(text.contains("config"));XCTAssertFalse(text.contains("/secret"))
        let response=try await MCPAutomationReadAccess().read(client:"test",project:e.project,root:e.root,args:["family":.string("versions")]) {components in
            if components.contains("workflow.train") {throw MCPSettings.Refusal(code:"component_unavailable")}
        }
        let filtered=String(decoding:try JSONEncoder().encode(response),as:UTF8.self)
        XCTAssertFalse(filtered.contains("old-train"));XCTAssertFalse(filtered.contains(oldVersion.id))
        do {_=try ScheduleSavedSource.details(root:e.root,projectID:e.project,versionID:UUID().uuidString);XCTFail("Missing saved source exposed choices")}catch{}
        XCTAssertFalse(FileManager.default.fileExists(atPath:e.root.appendingPathComponent("scheduled-runs").path))
    }
}
