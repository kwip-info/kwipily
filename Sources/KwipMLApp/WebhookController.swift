import Foundation
import Combine
import AppKit
import KwipMLCore
import KwipMLMCP

@MainActor final class WebhookController:NSObject,ObservableObject,MCPAppCommandProtocol {
    private static var instances:[String:WebhookController]=[:]
    static func shared(workspace:WorkspaceStore)->WebhookController {
        let key=workspace.executionAdmissionKey
        if let c=instances[key]{return c};let c=WebhookController(workspace:workspace);instances[key]=c;return c
    }
    let scheduler:ScheduledRunController
    let root:URL
    private weak var workspace:WorkspaceStore?
    private let secrets:any MCPSecretStorage
    private let transport:Bool
    private var ledger:WebhookLedger?
    private var connection:NSXPCConnection?
    private var generation=UUID()
    private var timer:Timer?
    private var dispatching=false
    private var sleeping=false
    private var observers:[NSObjectProtocol]=[]
    private var rateStart=Date()
    private var requestCount=0
    @Published private(set) var state=WebhookState()
    @Published private(set) var status="Stopped"
    @Published private(set) var error:String?
    @Published private(set) var activeDelivery:String?
    init(workspace:WorkspaceStore,secrets:(any MCPSecretStorage)?=nil,startTransport:Bool=true) {
        self.workspace=workspace;root=workspace.applicationDirectory.standardizedFileURL.resolvingSymlinksInPath();scheduler = .shared(workspace:workspace)
        self.secrets=secrets ?? MCPSecretKeychain(namespace:(Bundle.main.bundleIdentifier ?? "info.kwip.KwipML.POC")+".webhooks")
        transport=startTransport;super.init()
        Task {[weak self,root] in
            do {
                let ledger=try await Task.detached(priority:.utility){try WebhookLedger(root:root)}.value
                guard let self else{await ledger.close();return};self.ledger=ledger;await self.refresh()
                if startTransport {
                    let center=NSWorkspace.shared.notificationCenter
                    self.observers=[center.addObserver(forName:NSWorkspace.willSleepNotification,object:nil,queue:.main){[weak self]_ in Task{@MainActor in self?.sleeping=true;self?.stopListener()}},center.addObserver(forName:NSWorkspace.didWakeNotification,object:nil,queue:.main){[weak self]_ in Task{@MainActor in self?.sleeping=false;if self?.state.enabled==true{self?.startListener()}}}]
                    self.timer=Timer.scheduledTimer(withTimeInterval:1,repeats:true){[weak self]_ in Task{@MainActor in await self?.pump()}}
                    if self.state.enabled{self.startListener()}
                }
            }catch{self?.error="Webhooks unavailable. Existing evidence was preserved. "+error.localizedDescription}
        }
    }
    func ready()async throws {for _ in 0..<200 {if ledger != nil{return};if let error{throw WorkflowArchiveError(error)};try await Task.sleep(for:.milliseconds(10))};throw WebhookRefusal(503,"loading")}
    func refresh()async {guard let ledger else{return};let next=await ledger.snapshot();if next != state{state=next}}
    func shutdown()async {guard activeDelivery==nil else{return};timer?.invalidate();timer=nil;stopListener();for o in observers{NSWorkspace.shared.notificationCenter.removeObserver(o)};observers=[];await ledger?.close();ledger=nil}
    func configure(_ value:WebhookState)async throws {
        try await ready();let restart=value.port != state.port || value.enabled != state.enabled
        try await ledger?.configure(enabled:value.enabled,port:value.port,body:value.maximumBodyBytes,queued:value.maximumQueued,rate:value.requestsPerMinute,days:value.retentionDays,maximum:value.maximumDeliveries)
        await refresh();error=nil
        if restart {stopListener();if state.enabled && transport{startListener()}}
    }
    func pause()async {var s=state;s.enabled=false;do{try await configure(s)}catch{self.error=error.localizedDescription}}
    func save(_ proposed:WebhookEndpoint)async throws {
        try await ready();let details=try await scheduler.details(projectID:proposed.rule.projectID,versionID:proposed.rule.versionID)
        guard let version=try await scheduler.versions(projectID:proposed.rule.projectID).first(where:{$0.id==proposed.rule.versionID}),version.hash==proposed.rule.versionHash,version.number==proposed.rule.versionNumber else{throw WebhookRefusal(409,"saved_version_changed")}
        if !details.targets.isEmpty {guard details.targets.contains(where:{$0.address==proposed.rule.target}) else{throw WorkflowArchiveError("Choose the saved workflow's run target.")}}
        var e=proposed;e.rule.id=e.id;e.rule.name=e.name;e.rule.enabled=false;e.rule.hasEffects = !details.effects.isEmpty
        guard !e.enabled || !e.rule.hasEffects || e.rule.allowEffects else{throw WebhookRefusal(403,"authorize_outputs_before_enabling")}
        e.rule.outputAccessHash=e.rule.hasEffects && e.rule.allowEffects ? try await scheduler.outputAccessHash(projectID:e.rule.projectID):nil
        if let mapping=e.payload {
            try mapping.validate()
            guard details.document.nodes.contains(where:{if $0.address.domain == .data,$0.address.localID==mapping.tableNodeID,case .dataOperator(DataFlowSourceOperators.tableInput,let c)=$0.payload{return (try? TableInputConfiguration.format(of:c)) == .json};return false}) else{throw WorkflowArchiveError("Choose a Table Input configured for JSON document in the selected saved canvas.")}
        }
        let created=try secrets.read(clientID:e.credentialID)==nil
        if created {try secrets.write(MCPSecretKeychain.makeToken(),clientID:e.credentialID)}
        do{try await ledger?.save(e)}catch{if created{try? secrets.remove(clientID:e.credentialID)};throw error}
        await refresh();error=nil
    }
    func remove(_ e:WebhookEndpoint)async throws {try await ledger?.remove(e.id,expectedRevision:e.revision);await refresh();try secrets.remove(clientID:e.credentialID)}
    func secret(for e:WebhookEndpoint)throws->String {guard state.endpoints.contains(where:{$0==e}),let value=try secrets.read(clientID:e.credentialID) else{throw WebhookRefusal(503,"credential_unavailable")};return value}
    func rotate(_ e:WebhookEndpoint)async throws {
        var next=e;next.credentialID=UUID().uuidString;try await save(next);try secrets.remove(clientID:e.credentialID)
    }
    var listening:Bool {transport ? status=="Listening on this Mac":state.enabled}
    func url(for e:WebhookEndpoint)->String {"http://127.0.0.1:\(state.port)"+e.path}
    private func startListener() {
        guard state.enabled,!sleeping else{return};generation=UUID();let expected=generation;status="Starting"
        connection?.invalidate();let c=MCPServiceIdentity.connection(appCommands:self);connection=c
        let lost:()->Void={[weak self]in Task{@MainActor in guard let self,self.generation==expected else{return};self.status="Unavailable";self.error="Local listener stopped. Pause and enable it again from Settings."}}
        c.invalidationHandler=lost;c.interruptionHandler=lost
        guard let proxy=c.remoteObjectProxyWithErrorHandler({_ in lost()}) as? MCPServiceProtocol else{lost();return}
        let requested=state.port
        proxy.configureWebhook(requested){[weak self] port,failure in Task{@MainActor in guard let self,self.generation==expected,self.state.enabled else{return};if port==requested,failure==nil{self.status="Listening on this Mac";self.error=nil}else{self.status="Unavailable";self.error="The selected local port is unavailable. Choose another port and Apply."}}}
    }
    private func stopListener(){generation=UUID();status="Stopped";if let connection{(connection.remoteObjectProxy as? MCPServiceProtocol)?.stop{}};connection?.invalidate();connection=nil}
    nonisolated func dispatch(_ envelope:Data,reply:@escaping(Data?,String?)->Void){reply(nil,"not_permitted")}
    nonisolated func receiveWebhook(_ request:Data,reply:@escaping(Data?,String?)->Void) {
        guard request.count<=1_500_000 else{reply(try? JSONEncoder().encode(WebhookHTTPResponse(status:413,code:"payload_too_large")),nil);return}
        Task{@MainActor [weak self] in
            guard let self else{reply(nil,"unavailable");return}
            let result:WebhookHTTPResponse
            do {let value=try await Task.detached(priority:.utility){try JSONDecoder().decode(WebhookHTTPRequest.self,from:request)}.value;result=try await self.receive(value)}
            catch {result = .init(status:(error as? WebhookRefusal)?.status ?? 503,code:(error as? WebhookRefusal)?.code ?? "request_not_accepted")}
            reply(try? JSONEncoder().encode(result),nil)
        }
    }
    func receive(_ request:WebhookHTTPRequest,now:Date=Date(),origin:String="HTTP POST")async throws->WebhookHTTPResponse {
        try await ready()
        guard state.enabled,listening,!sleeping,let e=state.endpoints.first(where:{$0.path==request.path && $0.enabled}) else{throw WebhookRefusal(503,"endpoint_paused_or_unavailable")}
        if now.timeIntervalSince(rateStart)>=60{rateStart=now;requestCount=0};requestCount+=1
        guard requestCount<=state.requestsPerMinute else{throw WebhookRefusal(429,"rate_limit")}
        guard request.body.count<=state.maximumBodyBytes else{throw WebhookRefusal(413,"payload_too_large")}
        let token=try secret(for:e)
        try WebhookAuthentication.verify(method:e.authentication,secret:token,headers:request.headers,body:request.body,now:now)
        let id=try WebhookAuthentication.deliveryID(request.headers),hash=WorkflowArchiveCodec.hash(request.body)
        guard let ledger else{throw WebhookRefusal(503,"ledger_unavailable")}
        await ledger.setStorageLimit(try workspace?.executionPolicy(for:e.rule.projectID).storageLimitBytes)
        if let prior=state.deliveries.first(where:{$0.endpoint.id==e.id && $0.deliveryID==id && $0.state != .rejected}) {
            guard prior.digest==hash else{throw WebhookRefusal(409,"delivery_id_body_conflict")}
            let result=try await ledger.accept(endpoint:e,deliveryID:id,digest:hash,payload:nil,preview:nil,rows:nil,busy:false,origin:origin,now:now)
            await refresh();return .init(status:200,code:"duplicate",delivery:result.0.id,job:result.0.jobID)
        }
        let mapped:(data:Data,rows:Int)?
        do {mapped=try await Task.detached(priority:.utility){try e.payload?.decode(request.body)}.value}
        catch {try await ledger.reject(endpoint:e,deliveryID:id,digest:hash,code:(error as? WebhookRefusal)?.code ?? "invalid_payload",now:now);await refresh();throw error}
        let busy=ScheduledRunAdmission.isBusy(directory:root) || dispatching
        let optionalCapture=try await Task.detached(priority:.utility){try RunHistoryPolicyStore(root:self.root,projectID:e.rule.projectID).load().componentOutputs}.value
        var preview:Data?
        if optionalCapture,let data=mapped?.data,case .array(let rows)=try? JSONDecoder().decode(JSONValue.self,from:data){let bytes=try JSONEncoder().encode(JSONValue.array(Array(rows.prefix(8))));if bytes.count<=16384{preview=bytes}}
        let accepted=try await ledger.accept(endpoint:e,deliveryID:id,digest:hash,payload:mapped?.data,preview:preview,rows:mapped?.rows,busy:busy,origin:origin,now:now)
        await refresh()
        return .init(status:accepted.1 ? 200:202,code:accepted.1 ? "duplicate":"accepted",delivery:accepted.0.id,job:accepted.0.jobID)
    }
    func testEvent(_ e:WebhookEndpoint,body:Data)async throws->WebhookHTTPResponse {
        let stamp=String(Int(Date().timeIntervalSince1970)),id="test-"+UUID().uuidString,secret=try secret(for:e)
        var headers=["x-kwip-delivery-id":id]
        if e.authentication == .signed {headers["x-kwip-timestamp"]=stamp;headers["x-kwip-signature"]=WebhookAuthentication.signature(secret:secret,timestamp:stamp,deliveryID:id,body:body)}else{headers["authorization"]="Bearer "+secret}
        return try await receive(.init(path:e.path,headers:headers,body:body),origin:"Local test · in-app admission")
    }
    func pump()async {
        guard !dispatching,!sleeping,state.enabled,listening,!ScheduledRunAdmission.isBusy(directory:root),let ledger,
              let next=state.deliveries.reversed().first(where:{d in d.state == .queued && state.endpoints.contains(where:{$0.id==d.endpoint.id && $0.enabled})}) else{return}
        guard state.endpoints.contains(where:{$0.id==next.endpoint.id && $0.enabled}) else{return}
        dispatching=true;defer{dispatching=false;activeDelivery=nil}
        do {
            let claimed=try await ledger.claim(next.id);activeDelivery=claimed.id;await refresh()
            let types:[WebhookPayload.ValueType:String]=[.string:"String",.int64:"Int64",.float64:"Float64",.bool:"Boolean"]
            let schema=Dictionary(uniqueKeysWithValues:(claimed.endpoint.payload?.columns ?? []).filter(\.included).map{($0.column,types[$0.type]!)})
            let job=try await scheduler.executeWebhook(claimed.endpoint.rule,jobID:claimed.id,payload:claimed.payload,tableNodeID:claimed.endpoint.payload?.tableNodeID,eventSchema:schema)
            try await ledger.finish(claimed.id,job:job);await refresh()
        }catch {
            if (error as? WebhookRefusal)?.code=="workspace_busy",!scheduler.state.jobs.contains(where:{$0.id==next.id}) {try? await ledger.deferAdmission(next.id)}
            else {try? await ledger.finish(next.id,job:scheduler.state.jobs.first{$0.id==next.id});self.error="Webhook job needs attention. "+error.localizedDescription}
            await refresh()
        }
    }
    func cancel(_ d:WebhookDelivery)async {do{if d.state == .queued{try await ledger?.cancelQueued(d.id);await refresh()}else if d.state == .running{scheduler.cancel(d.id)}}catch{self.error=error.localizedDescription}}
    func review(_ d:WebhookDelivery)async throws {
        if let job=scheduler.state.jobs.first(where:{$0.id==d.id}),job.state == .needsReview,!job.reviewed {try await scheduler.review(job)}
        try await ledger?.review(d.id);await refresh()
    }
    func createExample()throws->String {
        guard let workspace else{throw WebhookRefusal(503,"workspace_unavailable")}
        let id=UUID().uuidString,source=WorkflowNodeAddress(domain:.data,localID:"event-rows"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        let document=WorkflowDocument(projectID:id,name:"Webhook intake · local review",nodes:[.init(address:source,name:"Event rows",x:80,y:160,payload:.dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:["format":.string("json"),"options":.object([:])])),.init(address:view,name:"Inspect event rows",x:430,y:160,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(from:.init(address:source,port:"table"),to:.init(address:view,port:"table"),kind:.table)])
        try WorkflowCodec.validate(document)
        let bytes=Data("[{\"order_id\":1042,\"amount\":29.5},{\"order_id\":1043,\"amount\":18.0}]".utf8)
        let parts=["webhooks","examples",id+".json"]
        try DiagnosticFileIO.writeImmutable(bytes,root:root,parts:parts,limit:4096)
        let record=try workspace.projectLibrary.saveSource(.workflow(document),id:id)
        let window=try ProjectWindowRegistry.shared.open(record)
        if let host=window.session.workspace.workflowSession {
            guard host.bindings.setPath(root.appendingPathComponent(parts.joined(separator:"/")).path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision) else{throw WorkflowArchiveError(host.bindings.error ?? "Example source unavailable")}
            try host.owner.saveCanvasVersion(label:"Webhook review starter")
        }
        return id
    }
    func clientExample(_ e:WebhookEndpoint)->String {
        let url=url(for:e)
        return """
        # Set KWIP_WEBHOOK_SECRET privately to the endpoint secret; do not put it in your project.
        # Save the JSON body as event.json. Reuse DELIVERY_ID only for an exact retry.
        export DELIVERY_ID=example-001
        export WEBHOOK_URL='\(url)'
        python3 - <<'PY'
        import os,time,hmac,hashlib,urllib.request
        body=open('event.json','rb').read()
        stamp=str(int(time.time())); delivery=os.environ['DELIVERY_ID']
        secret=os.environ['KWIP_WEBHOOK_SECRET']
        headers={'Content-Type':'application/json','X-Kwip-Delivery-ID':delivery}
        if '\(e.authentication.rawValue)' == 'signed':
            signed=stamp.encode()+b'.'+delivery.encode()+b'.'+body
            headers.update({'X-Kwip-Timestamp':stamp,'X-Kwip-Signature':hmac.new(secret.encode(),signed,hashlib.sha256).hexdigest()})
        else: headers['Authorization']='Bearer '+secret
        request=urllib.request.Request(os.environ['WEBHOOK_URL'],data=body,headers=headers,method='POST')
        print(urllib.request.urlopen(request,timeout=20).read().decode())
        PY
        """
    }
}
