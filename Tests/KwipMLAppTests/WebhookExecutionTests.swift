import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WebhookExecutionTests:XCTestCase {
    final class Secrets:MCPSecretStorage {
        var values:[String:String]=[:]
        func read(clientID:String)throws->String?{values[clientID]}
        func write(_ token:String,clientID:String)throws{values[clientID]=token}
        func remove(clientID:String)throws{values[clientID]=nil}
    }
    func testActualEventUsesPinnedVersionTypedInputCaptureOffAndNoEditorMutation()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"]else{throw XCTSkip("Select existing development Python.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let input=WorkflowNodeAddress(domain:.data,localID:"event-rows"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        let doc=WorkflowDocument(projectID:UUID().uuidString,name:"Pinned event",nodes:[.init(address:input,name:"Event rows",payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:["format":.string("json"),"options":.object([:])])),.init(address:view,x:300,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(from:.init(address:input,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        _ = try w.projectLibrary.saveSource(.workflow(doc),id:doc.projectID);XCTAssertTrue(w.beginWorkflowProject(id:doc.projectID),w.statusMessage)
        let host=try XCTUnwrap(w.workflowSession);try host.owner.saveCanvasVersion(label:"Event contract");try host.owner.setRunHistoryPolicy(.none)
        let v=try XCTUnwrap(host.owner.canvasVersions?.versions.first)
        let c=WebhookController(workspace:w,secrets:Secrets(),startTransport:false);try await c.ready();try await c.scheduler.ready()
        c.scheduler.configureJobWorkspace={workspace in workspace.workflowSession?.executionLaunch={_,args,dir,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)}}
        let id=UUID().uuidString
        var e=WebhookEndpoint(id:id,rule:ScheduleRule(id:id,projectID:doc.projectID,projectName:doc.name,versionID:v.id,versionNumber:v.number,versionHash:v.hash,name:"Events",timing:.init()),enabled:true,authentication:.bearer,payload:WebhookPayload(tableNodeID:input.localID,recordsPath:"items",columns:[.init(field:"id",column:"order_id",type:.int64),.init(field:"amount",column:"amount",type:.float64)]))
        try await c.save(e);e=try XCTUnwrap(c.state.endpoints.first)
        var settings=c.state;settings.enabled=true;try await c.configure(settings)
        let original=host.owner.document,bindings=host.bindings.bindings
        let body=Data("{\"items\":[{\"id\":1042,\"amount\":2},{\"id\":1043,\"amount\":4}]}".utf8)
        let headers=["authorization":"Bearer "+(try c.secret(for:e)),"x-kwip-delivery-id":"fixed-1"]
        let response=try await c.receive(.init(path:e.path,headers:headers,body:body))
        XCTAssertEqual(response.status,202);XCTAssertNil(c.state.deliveries.first?.preview)
        let duplicate=try await c.receive(.init(path:e.path,headers:headers,body:body));XCTAssertEqual(duplicate.status,200);XCTAssertEqual(response.delivery,duplicate.delivery)
        let invalid=Data("{\"items\":[{\"id\":\"wrong\",\"amount\":2}]}".utf8)
        var invalidHeaders=headers;invalidHeaders["x-kwip-delivery-id"]="invalid-2"
        do{_ = try await c.receive(.init(path:e.path,headers:invalidHeaders,body:invalid));XCTFail("Wrong field type accepted")}catch{XCTAssertEqual((error as? WebhookRefusal)?.status,422)}
        let refusal=try XCTUnwrap(c.state.deliveries.first{$0.deliveryID=="invalid-2"});XCTAssertEqual(refusal.state,.rejected);XCTAssertNil(refusal.payload);XCTAssertNil(refusal.preview)
        await c.pump()
        let delivery=try XCTUnwrap(c.state.deliveries.first{$0.id==response.delivery});XCTAssertEqual(delivery.state,.completed,delivery.message);XCTAssertNil(delivery.payload)
        XCTAssertEqual(host.owner.document,original);XCTAssertEqual(host.bindings.bindings,bindings);XCTAssertFalse(w.isBusy)
        let job=try XCTUnwrap(c.scheduler.state.jobs.first);XCTAssertEqual(job.trigger,"Webhook");XCTAssertEqual(job.state,.completed,job.message)
        let captured=await c.scheduler.preparation(for:job);let history=try XCTUnwrap(captured)
        guard let inputNode=history.1.document.nodes.first(where:{$0.address==input}),case .dataOperator(_,let config)=inputNode.payload else{return XCTFail("No captured input")}
        XCTAssertEqual(try TableInputConfiguration.options(in:config)["schema"],.object(["order_id":.string("Int64"),"amount":.string("Float64")]))
        XCTAssertEqual(c.scheduler.state.jobs.count,1)
        try await c.rotate(e)
        do{_ = try await c.receive(.init(path:e.path,headers:headers,body:body));XCTFail("Rotated credential accepted")}catch{XCTAssertEqual((error as? WebhookRefusal)?.status,401)}
        XCTAssertFalse(String(decoding:try Data(contentsOf:root.appendingPathComponent("webhooks/state.json")),as:UTF8.self).contains(headers["authorization"]!))
        settings=c.state;settings.enabled=false;try await c.configure(settings)
        await c.shutdown();await c.scheduler.shutdown()
    }
    func testDisabledListenerNeverEnqueuesOrPublishesAuthentication()async throws {
        // Contract tests cover signature arithmetic. This seam checks the controller does not
        // publish authentication material, dispatch a worker or create a job on rejection.
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let c=WebhookController(workspace:w,secrets:Secrets(),startTransport:false);try await c.ready()
        do{_ = try await c.receive(.init(path:"/hooks/"+UUID().uuidString,headers:["authorization":"Bearer private"],body:Data("secret".utf8)));XCTFail("Disabled listener accepted work")}catch{XCTAssertEqual((error as? WebhookRefusal)?.status,503)}
        XCTAssertTrue(c.state.deliveries.isEmpty);XCTAssertTrue(c.scheduler.state.jobs.isEmpty)
        await c.shutdown();try await c.scheduler.ready();await c.scheduler.shutdown()
    }
}

extension WebhookExecutionTests {
    private final class DatabaseSecrets:DatabaseSourceCredentialStorage {
        var values:[String:DatabaseSourceCredentials]=[:]
        func read(projectID:String,nodeID:String,connectionID:String)throws->DatabaseSourceCredentials?{values[projectID+nodeID+connectionID]}
        func write(_ value:DatabaseSourceCredentials,projectID:String,nodeID:String,connectionID:String)throws{values[projectID+nodeID+connectionID]=value}
        func clear(projectID:String,nodeID:String,connectionID:String)throws{values[projectID+nodeID+connectionID]=nil}
    }
    func testUncertainWritePausesEndpointAndReviewDoesNotResend()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"]else{throw XCTSkip("Select existing development Python.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var document=try TableOutputPracticeExample.document();let address=WorkflowNodeAddress(domain:.data,localID:"table-output")
        let index=try XCTUnwrap(document.nodes.firstIndex{$0.address==address})
        var config=APITableSettings.defaults;config["direction"] = .string("output");config["method"] = .string("POST");config["url"] = .string("https://api.example.com/orders");config["mapping"] = .array([.object(["source":.string("order_id"),"target":.string("/id"),"type":.string("string")])])
        document.nodes[index].payload = .dataOperator(operatorID:"kwip.data.outputDelimited",config:["destination":.string("api"),"api":.object(config),"filename":.string("old.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false])
        _ = try w.projectLibrary.saveSource(.workflow(document),id:document.projectID);XCTAssertTrue(w.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(w.workflowSession)
        XCTAssertTrue(host.bindings.setPath(try TableOutputPracticeExample.resourceURL().path,for:.init(domain:.data,localID:"source"),lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        let credentials=DatabaseSecrets();host.outputDestinations.databaseOutputs.credentials=credentials
        try host.outputDestinations.databaseOutputs.saveProfile(name:"Synthetic unknown receipt",config:config,username:"",password:"",allowWrites:true,address:address)
        try host.owner.saveCanvasVersion();let version=try XCTUnwrap(host.owner.canvasVersions?.versions.first)
        let c=WebhookController(workspace:w,secrets:Secrets(),startTransport:false);try await c.ready();try await c.scheduler.ready()
        var calls=0
        c.scheduler.configureJobWorkspace={workspace in
            workspace.workflowSession?.executionLaunch={_,args,dir,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)}
            workspace.workflowSession?.outputDestinations.databaseOutputs.credentials=credentials
            workspace.workflowSession?.outputDestinations.databaseOutputs.apiExecute={_,_,_,id,_,hash,_ in calls+=1;return ["schemaVersion":1,"kind":.string("apiWrite"),"operationID":.string(id),"status":.string("unknown"),"sourceSHA256":.string(hash),"inputRows":12,"acceptedRows":4,"requests":.array([])]}
        }
        let id=UUID().uuidString
        let e=WebhookEndpoint(id:id,rule:ScheduleRule(id:id,projectID:document.projectID,projectName:document.name,versionID:version.id,versionNumber:version.number,versionHash:version.hash,name:"Unknown write",timing:.init(),allowEffects:true),enabled:true,authentication:.bearer)
        try await c.save(e);var s=c.state;s.enabled=true;try await c.configure(s)
        let saved=try XCTUnwrap(c.state.endpoints.first),headers=["authorization":"Bearer "+(try c.secret(for:saved)),"x-kwip-delivery-id":"one-write"]
        _ = try await c.receive(.init(path:saved.path,headers:headers,body:Data("{}".utf8)));await c.pump()
        let d=try XCTUnwrap(c.state.deliveries.first);XCTAssertEqual(d.state,.needsReview,d.message);XCTAssertFalse(c.state.endpoints[0].enabled);XCTAssertEqual(calls,1)
        try await c.review(d);XCTAssertEqual(calls,1);XCTAssertFalse(c.state.endpoints[0].enabled)
        do{_ = try await c.receive(.init(path:saved.path,headers:headers,body:Data("{}".utf8)));XCTFail("Paused endpoint accepted resend")}catch{XCTAssertEqual((error as? WebhookRefusal)?.status,503)}
        XCTAssertEqual(c.scheduler.state.jobs.count,1)
        await c.shutdown();await c.scheduler.shutdown()
    }
}
