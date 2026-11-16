import Foundation
import Darwin
@preconcurrency import KwipMLCore

struct WebhookEndpoint:Codable,Equatable,Identifiable,Sendable {
    enum Busy:String,Codable,CaseIterable,Sendable {case queue,reject}
    var id=UUID().uuidString
    var revision:UInt64=0
    var name="New endpoint"
    var rule:ScheduleRule
    var enabled=false
    var authentication:WebhookAuthentication.Method = .signed
    var credentialID=UUID().uuidString
    var busy:Busy = .queue
    var payload:WebhookPayload? = nil
    var path:String {"/hooks/"+id}
    func validate()throws {
        try rule.validate()
        guard UUID(uuidString:id)?.uuidString==id,UUID(uuidString:credentialID)?.uuidString==credentialID,rule.id==id,revision<UInt64.max-4,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=120 else{throw WebhookRefusal(422,"invalid_endpoint")}
        try payload?.validate()
    }
}
struct WebhookDelivery:Codable,Equatable,Identifiable,Sendable {
    enum State:String,Codable,Sendable {case queued,running,completed,failed,rejected,needsReview,cancelled}
    var id=UUID().uuidString
    var endpoint:WebhookEndpoint
    var deliveryID:String
    var digest:String
    var receivedAt=Date()
    var state:State = .queued
    var httpStatus=202
    var origin="HTTP POST"
    var message="Durably queued"
    var duplicates=0
    var rows:Int?
    var payload:Data?
    var payloadSHA256:String?
    var preview:Data?
    var jobID:String?
    var finishedAt:Date?
    var reviewed=false
    var title:String {state == .needsReview ? "Needs review":state.rawValue.capitalized}
}
struct WebhookState:Codable,Equatable,Sendable {
    var schemaVersion=1
    var enabled=false
    var port=8788
    var maximumBodyBytes=1_048_576
    var maximumQueued=32
    var requestsPerMinute=60
    var retentionDays=30
    var maximumDeliveries=1000
    var endpoints:[WebhookEndpoint]=[]
    var deliveries:[WebhookDelivery]=[]
}
actor WebhookLedger {
    private let root:URL
    private var lease:WorkflowFileLease?
    private var state:WebhookState
    private var storageLimitBytes:Int64?
    private static let parts=["webhooks","state.json"]
    static let byteLimit=32_000_000
    /// Observes durable metadata without ownership, recovery, writes or service startup.
    /// The existing family decoder and validator remain the only ledger authority.
    static func readOnlySnapshot(root:URL)throws->WebhookState? {
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:Self.parts,limit:Self.byteLimit) else {return nil}
        let value=try JSONDecoder().decode(WebhookState.self,from:bytes)
        try Self.validate(value)
        return value
    }
    init(root:URL)throws {
        self.root=root.standardizedFileURL.resolvingSymlinksInPath()
        let folder=self.root.appendingPathComponent("webhooks")
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        // Anchored read/write validates ancestry before following the private owner lease.
        _ = try DiagnosticFileIO.readIfPresent(root:self.root,parts:Self.parts,limit:Self.byteLimit)
        var info=stat();guard lstat(folder.path,&info)==0,info.st_mode & S_IFMT == S_IFDIR,let pointer=realpath(folder.path,nil) else{throw WorkflowFileLeaseError.unsafePath};let resolved=URL(fileURLWithPath:String(cString:pointer));free(pointer)
        lease=try WorkflowFileLease(directory:resolved,name:"owner.lock",mode:.exclusive,create:true)
        if let data=try DiagnosticFileIO.readIfPresent(root:self.root,parts:Self.parts,limit:Self.byteLimit){state=try JSONDecoder().decode(WebhookState.self,from:data)}else{state=WebhookState()}
        try Self.validate(state)
        for i in state.deliveries.indices where state.deliveries[i].state == .running {
            state.deliveries[i].state = .needsReview;state.deliveries[i].message="Interrupted dispatch. Review the job and output destinations before enabling this endpoint.";state.deliveries[i].payload=nil
            if let e=state.endpoints.firstIndex(where:{$0.id==state.deliveries[i].endpoint.id}){state.endpoints[e].enabled=false;state.endpoints[e].revision+=1}
        }
        try DiagnosticFileIO.writeAtomic(try WorkflowArchiveCodec.encode(state),root:self.root,parts:Self.parts)
    }
    func close(){lease?.close();lease=nil}
    func snapshot()->WebhookState {state}
    func setStorageLimit(_ value:Int64?){storageLimitBytes=value}
    private static func validate(_ s:WebhookState)throws {
        guard s.schemaVersion==1,(1024...65535).contains(s.port),(1...1_048_576).contains(s.maximumBodyBytes),(1...32).contains(s.maximumQueued),(1...600).contains(s.requestsPerMinute),(1...365).contains(s.retentionDays),(10...1000).contains(s.maximumDeliveries),s.endpoints.count<=64,s.deliveries.count<=1000,Set(s.endpoints.map(\.id)).count==s.endpoints.count,Set(s.deliveries.map(\.id)).count==s.deliveries.count else{throw WebhookRefusal(503,"invalid_or_full_ledger")}
        for e in s.endpoints{try e.validate()}
        for d in s.deliveries {try d.endpoint.validate();guard UUID(uuidString:d.id) != nil,d.deliveryID.count<=128,WorkflowArchiveCodec.isHash(d.digest),(d.payload?.count ?? 0)<=1_048_576,(d.preview?.count ?? 0)<=16384,d.message.count<=1000,d.duplicates>=0,d.payload==nil || d.payloadSHA256==d.payload.map(WorkflowArchiveCodec.hash) else{throw WebhookRefusal(503,"invalid_delivery_record")}}
    }
    private func commit(_ next:WebhookState)throws {guard lease != nil else{throw WorkflowFileLeaseError.closed};try Self.validate(next);let bytes=try WorkflowArchiveCodec.encode(next);guard bytes.count<=Self.byteLimit else{throw WebhookRefusal(507,"delivery_storage_full")};let admission=try storageLimitBytes.map{try WorkflowStorageAdmission(workspace:root,accountingRoot:root).reserve(scopes:[.init(namespace:.webhooks,components:[UUID().uuidString])],peakBytes:Int64(bytes.count),limitBytes:$0)};defer{admission?.close()};try DiagnosticFileIO.writeAtomic(bytes,root:root,parts:Self.parts);state=next}
    func configure(enabled:Bool,port:Int,body:Int,queued:Int,rate:Int,days:Int,maximum:Int)throws {
        var next=state;next.enabled=enabled;next.port=port;next.maximumBodyBytes=body;next.maximumQueued=queued;next.requestsPerMinute=rate;next.retentionDays=days;next.maximumDeliveries=maximum
        prune(&next,now:Date());try commit(next)
    }
    func save(_ e:WebhookEndpoint)throws {
        try e.validate();var next=state;var e=e
        guard !next.deliveries.contains(where:{$0.endpoint.id==e.id && ($0.state == .running || $0.state == .queued)}) else{throw WebhookRefusal(409,"endpoint_has_pending_deliveries")}
        guard !e.enabled || !next.deliveries.contains(where:{$0.endpoint.id==e.id && $0.state == .needsReview && !$0.reviewed}) else{throw WebhookRefusal(409,"review_uncertain_delivery_first")}
        if let i=next.endpoints.firstIndex(where:{$0.id==e.id}){guard next.endpoints[i].revision==e.revision else{throw WebhookRefusal(409,"stale_endpoint_draft")};e.revision+=1;next.endpoints[i]=e}else{guard e.revision==0 else{throw WebhookRefusal(409,"missing_endpoint")};e.revision=1;next.endpoints.append(e)}
        try commit(next)
    }
    func remove(_ id:String,expectedRevision:UInt64?=nil)throws {var next=state;if let expectedRevision{guard next.endpoints.first(where:{$0.id==id})?.revision==expectedRevision else{throw WebhookRefusal(409,"stale_endpoint_draft")}};guard !next.deliveries.contains(where:{$0.endpoint.id==id && [.queued,.running].contains($0.state)}) else{throw WebhookRefusal(409,"endpoint_has_pending_deliveries")};next.endpoints.removeAll{$0.id==id};try commit(next)}
    private func prune(_ next:inout WebhookState,now:Date){
        let cutoff=now.addingTimeInterval(-Double(next.retentionDays)*86400)
        func protected(_ d:WebhookDelivery)->Bool {[.queued,.running].contains(d.state) || d.state == .needsReview && !d.reviewed}
        next.deliveries.removeAll{!protected($0) && $0.receivedAt<cutoff}
        while next.deliveries.count>=next.maximumDeliveries,let i=next.deliveries.lastIndex(where:{!protected($0)}) {next.deliveries.remove(at:i)}
    }
    func accept(endpoint:WebhookEndpoint,deliveryID:String,digest:String,payload:Data?,preview:Data?,rows:Int?,busy:Bool,origin:String="HTTP POST",now:Date=Date())throws->(WebhookDelivery,Bool) {
        guard state.enabled,let current=state.endpoints.first(where:{$0.id==endpoint.id}),current==endpoint,current.enabled else{throw WebhookRefusal(503,"endpoint_paused")}
        if let i=state.deliveries.firstIndex(where:{$0.endpoint.id==endpoint.id && $0.deliveryID==deliveryID}) {
            guard state.deliveries[i].state != .rejected else{throw WebhookRefusal(422,"previously_rejected_use_new_delivery_id")}
            guard state.deliveries[i].digest==digest else{throw WebhookRefusal(409,"delivery_id_body_conflict")}
            var next=state;next.deliveries[i].duplicates=min(1_000_000,next.deliveries[i].duplicates+1);try commit(next);return (next.deliveries[i],true)
        }
        guard !busy || endpoint.busy == .queue else{throw WebhookRefusal(409,"workspace_busy")}
        guard state.deliveries.filter({[.queued,.running].contains($0.state)}).count<state.maximumQueued else{throw WebhookRefusal(429,"queue_full")}
        var next=state;prune(&next,now:now)
        guard next.deliveries.count<next.maximumDeliveries else{throw WebhookRefusal(507,"protected_history_full")}
        let d=WebhookDelivery(endpoint:endpoint,deliveryID:deliveryID,digest:digest,receivedAt:now,origin:origin,rows:rows,payload:payload,payloadSHA256:payload.map(WorkflowArchiveCodec.hash),preview:preview)
        next.deliveries.insert(d,at:0);try commit(next);return(d,false)
    }
    func reject(endpoint:WebhookEndpoint,deliveryID:String,digest:String,code:String,now:Date=Date())throws {
        var next=state;prune(&next,now:now)
        guard next.deliveries.count<next.maximumDeliveries else{return}
        next.deliveries.insert(WebhookDelivery(endpoint:endpoint,deliveryID:deliveryID,digest:digest,receivedAt:now,state:.rejected,httpStatus:422,message:code,finishedAt:now),at:0)
        try commit(next)
    }
    func claim(_ id:String)throws->WebhookDelivery {
        var next=state;guard next.enabled,!next.deliveries.contains(where:{$0.state == .running}),let i=next.deliveries.firstIndex(where:{$0.id==id && $0.state == .queued}),next.endpoints.contains(where:{$0.id==next.deliveries[i].endpoint.id && $0.enabled}) else{throw WebhookRefusal(409,"delivery_not_ready")}
        next.deliveries[i].state = .running;next.deliveries[i].jobID=id;try commit(next);return next.deliveries[i]
    }
    func deferAdmission(_ id:String)throws {
        var next=state;guard let i=next.deliveries.firstIndex(where:{$0.id==id && $0.state == .running})else{throw WebhookRefusal(409,"delivery_not_owned")}
        next.deliveries[i].state = .queued;next.deliveries[i].jobID=nil;try commit(next)
    }
    func finish(_ id:String,job:ScheduledJob?)throws {
        var next=state;guard let i=next.deliveries.firstIndex(where:{$0.id==id}),next.deliveries[i].state == .running else{throw WebhookRefusal(409,"delivery_not_owned")}
        let status=job?.state
        next.deliveries[i].state=status == .completed ? .completed:status == .needsReview || job==nil ? .needsReview:status == .cancelled ? .cancelled:.failed
        next.deliveries[i].message=String((job?.message ?? "Dispatch interrupted. Review the job before another attempt.").prefix(1000));next.deliveries[i].finishedAt=Date();next.deliveries[i].payload=nil
        if next.deliveries[i].state == .needsReview,let e=next.endpoints.firstIndex(where:{$0.id==next.deliveries[i].endpoint.id}) {next.endpoints[e].enabled=false;next.endpoints[e].revision+=1}
        try commit(next)
    }
    func cancelQueued(_ id:String)throws {var next=state;guard let i=next.deliveries.firstIndex(where:{$0.id==id && $0.state == .queued}) else{throw WebhookRefusal(409,"delivery_not_queued")};next.deliveries[i].state = .cancelled;next.deliveries[i].payload=nil;next.deliveries[i].preview=nil;next.deliveries[i].finishedAt=Date();next.deliveries[i].message="Cancelled before dispatch; no workflow run.";try commit(next)}
    func review(_ id:String)throws {var next=state;guard let i=next.deliveries.firstIndex(where:{$0.id==id && $0.state == .needsReview}) else{throw WebhookRefusal(409,"no_review_needed")};next.deliveries[i].reviewed=true;try commit(next)}
}
