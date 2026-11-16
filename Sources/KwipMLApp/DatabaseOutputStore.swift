import AppKit
import Combine
import Darwin
import KwipMLCore

/// Private connection authority and cached presentation for one canonical owner.
/// Profiles are shared with Table Input; portable configuration cannot grant writes.
@MainActor final class DatabaseOutputStore:ObservableObject {
    struct Profile:Codable,Equatable,Identifiable {
        let id:String
        var name:String
        var endpoint:[String:JSONValue]
        var allowWrites:Bool
    }
    struct Library:Codable {let schemaVersion:Int;var profiles:[Profile]}
    struct Bindings:Codable {let schemaVersion:Int;var profiles:[String:String]}
    struct Metadata {let identity:String;let value:[String:JSONValue]}
    let owner:WorkflowAuthoringStore
    private weak var workspace:WorkspaceStore?
    private weak var destinations:DataOutputDestinationProvider?
    let root:URL
    let libraryRoot:URL
    @Published private(set) var profiles:[Profile]=[]
    @Published private(set) var bindings:[String:String]=[:]
    @Published private(set) var metadata:[String:Metadata]=[:]
    @Published private(set) var records:[DatabaseOutputLedger.Record]=[]
    @Published private(set) var status="Test the connection to inspect destination tables and columns."
    @Published private(set) var isBusy=false
    var credentials:DatabaseSourceCredentialStorage=DatabaseSourceKeychain()
    var execute:(String,[String:JSONValue],DatabaseSourceCredentials,String,FileHandle?,String,Data?,String)async throws->[String:JSONValue] = {action,config,secret,id,table,hash,bookmark,filename in
        try await DatabaseOutputBrokerClient().perform(action:action,options:config,credentials:secret,operationID:id,table:table,tableSHA256:hash,directoryBookmark:bookmark,filename:filename)
    }
    var apiExecute:(String,[String:JSONValue],DatabaseSourceCredentials,String,FileHandle?,String,Int?)async throws->[String:JSONValue] = {action,config,secret,id,table,hash,memory in try await DatabaseOutputBrokerClient().perform(action:action,options:config,credentials:secret,operationID:id,table:table,tableSHA256:hash,directoryBookmark:nil,filename:"",memoryLimit:memory)}
    private var task:Task<Void,Never>?
    var ledger:DatabaseOutputLedger {.init(root:root,projectID:owner.projectID)}
    init(workspace:WorkspaceStore,owner:WorkflowAuthoringStore,destinations:DataOutputDestinationProvider) {
        self.workspace=workspace;self.owner=owner;self.destinations=destinations;root=workspace.workspaceDirectory;libraryRoot=workspace.applicationDirectory
        do {try reloadProfiles();if let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:bindingParts,limit:128_000) {
            let v=try JSONDecoder().decode(Bindings.self,from:bytes);guard v.schemaVersion==1,v.profiles.count<=128,v.profiles.allSatisfy({!$0.key.isEmpty && $0.key.utf8.count<=128 && UUID(uuidString:$0.value)?.uuidString==$0.value}) else {throw WorkflowFileLeaseError.unsafePath};bindings=v.profiles
        };records=try ledger.load()} catch {status="Private database connections or receipts need attention; existing data was preserved."}
    }
    private var bindingParts:[String] {["database-output-bindings",owner.projectID,"bindings.json"]}
    private func endpoint(_ config:[String:JSONValue])throws->[String:JSONValue] {
        if config["kind"] == .string("api") {
            let origin=try APITableSettings.origin(config)
            return ["kind":.string("api"),"origin":.string(origin),"auth":config["auth"] ?? .string("none"),"credentialHeader":config["credentialHeader"] ?? .string("X-API-Key")]
        }
        let c=try DatabaseOutputSettings.normalized(config.filter {Set(["backend","host","port","database","tlsMode"]).contains($0.key)},requireMapping:false)
        return c.filter {Set(["backend","host","port","database","tlsMode"]).contains($0.key)}
    }
    func profile(for address:WorkflowNodeAddress)->Profile? {profiles.first {$0.id==bindings[address.localID]}}
    func reloadProfiles()throws {
        guard let data=try DiagnosticFileIO.readIfPresent(root:libraryRoot,parts:["database-connections","profiles.json"],limit:256_000) else {profiles=[];return}
        let v=try JSONDecoder().decode(Library.self,from:data)
        guard v.schemaVersion==1,v.profiles.count<=128,Set(v.profiles.map(\.id)).count==v.profiles.count else {throw WorkflowFileLeaseError.unsafePath}
        for p in v.profiles {guard UUID(uuidString:p.id)?.uuidString==p.id,!p.name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,p.name.utf8.count<=80,Set(p.endpoint.keys)==["backend","host","port","database","tlsMode"] || Set(p.endpoint.keys)==["kind","origin","auth","credentialHeader"] else {throw WorkflowFileLeaseError.unsafePath};if p.endpoint["kind"] != .string("api") {_ = try endpoint(p.endpoint)} else {guard let u=p.endpoint["origin"]?.stringValue,let c=URLComponents(string:u),c.host != nil,c.user==nil,c.password==nil,["https","http"].contains(c.scheme ?? "") else {throw WorkflowFileLeaseError.unsafePath}}}
        profiles=v.profiles
    }
    private func persistProfiles(_ next:[Profile])throws {
        guard next.count<=128 else {throw WorkflowArchiveError("Connection profile limit reached.")}
        try FileManager.default.createDirectory(at:libraryRoot.appendingPathComponent("database-connections"),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        let bytes=try WorkflowArchiveCodec.encode(Library(schemaVersion:1,profiles:next));guard bytes.count<=256_000 else {throw WorkflowFileLeaseError.unsafePath}
        try DiagnosticFileIO.writeAtomic(bytes,root:libraryRoot,parts:["database-connections","profiles.json"]);profiles=next
    }
    private func mayEdit(_ address:WorkflowNodeAddress)->Bool {
        owner.isCurrent(owner.lease()) && !owner.isBusy && !isBusy && owner.selection.nodeAddresses==[address] && workspace?.activeProjectID==owner.projectID && !(workspace?.isBusy ?? true) && address.domain == .data && owner.document.nodes.contains {node in guard node.address==address,case .dataOperator(let id,_)=node.payload else{return false};return ["kwip.data.sourceCSV","kwip.data.outputDelimited"].contains(id)}
    }
    func saveProfile(name:String,config:[String:JSONValue],username:String,password:String,allowWrites:Bool,address:WorkflowNodeAddress,updating:String?=nil)throws {
        guard mayEdit(address) else {throw WorkflowArchiveError("Use the selected idle component to change a private connection.")}
        let title=name.trimmingCharacters(in:.whitespacesAndNewlines);guard !title.isEmpty,title.utf8.count<=80 else {throw WorkflowArchiveError("Name this connection using at most80bytes.")}
        try reloadProfiles();let ep=try endpoint(config),id=updating ?? UUID().uuidString
        guard updating==nil || profiles.contains(where:{$0.id==id}) else {throw WorkflowFileLeaseError.unsafePath}
        if ep["backend"] != .string("sqlite") {try credentials.write(.init(username:username,password:password),projectID:id,nodeID:"connection",connectionID:try ep["kind"] == .string("api") ? WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(ep)):DatabaseSourceSettings.connectionIdentity(ep))}
        guard mayEdit(address) else {throw WorkflowArchiveError("The selected project changed while saving credentials. Reopen its connection editor.")}
        var next=profiles.filter {$0.id != id};next.append(.init(id:id,name:title,endpoint:ep,allowWrites:allowWrites));try persistProfiles(next)
        try bind(id,address:address);status="Connection saved privately on this Mac."
    }
    func bind(_ id:String,address:WorkflowNodeAddress)throws {
        guard mayEdit(address),profiles.contains(where:{$0.id==id}),bindings.count<128 || bindings[address.localID] != nil else {throw WorkflowArchiveError("Select an idle component before linking a connection profile.")}
        var next=bindings;next[address.localID]=id
        try FileManager.default.createDirectory(at:root.appendingPathComponent(bindingParts.dropLast().joined(separator:"/")),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(Bindings(schemaVersion:1,profiles:next)),root:root,parts:bindingParts);bindings=next;metadata[address.localID]=nil;destinations?.privateAuthorityChanged()
    }
    func unbind(_ address:WorkflowNodeAddress)throws {
        guard mayEdit(address) else {throw WorkflowArchiveError("Use an idle selected component.")}
        var next=bindings;next[address.localID]=nil
        try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(Bindings(schemaVersion:1,profiles:next)),root:root,parts:bindingParts);bindings=next;metadata[address.localID]=nil;destinations?.privateAuthorityChanged()
    }
    func secret(for address:WorkflowNodeAddress,config:[String:JSONValue],write:Bool)throws->DatabaseSourceCredentials {
        try reloadProfiles()
        // SQLite authority is the per-node folder grant, not a portable profile.
        if config["backend"] == .string("sqlite") {return .init(username:"",password:"")}
        guard let p=profile(for:address),try endpoint(config)==p.endpoint,!write || p.allowWrites,let value=try credentials.read(projectID:p.id,nodeID:"connection",connectionID:try p.endpoint["kind"] == .string("api") ? WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(p.endpoint)):DatabaseSourceSettings.connectionIdentity(p.endpoint)) else {throw WorkflowArchiveError("Select a matching private connection profile\(write ? " with Table Output writes enabled":"") and its saved credential.")}
        return value
    }
    func metadataIdentity(_ config:[String:JSONValue],filename:String,address:WorkflowNodeAddress)throws->String {
        var c=try endpoint(config);c["schema"]=config["schema"] ?? .string("main");c["table"]=config["table"] ?? .string("");c["filename"] = .string(filename);c["profile"] = .string(bindings[address.localID] ?? "");c["folder"] = .string(destinations?.databasePath(for:address) ?? "")
        return WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(c))
    }
    func cachedMetadata(_ config:[String:JSONValue],filename:String,address:WorkflowNodeAddress)->[String:JSONValue]? {
        guard let cached=metadata[address.localID],let identity=try? metadataIdentity(config,filename:filename,address:address),identity==cached.identity else {return nil};return cached.value
    }
    func inspect(config:[String:JSONValue],filename:String,address:WorkflowNodeAddress) {
        guard let workspace,owner.isCurrent(owner.lease()),!isBusy,!owner.isBusy,!workspace.isBusy else {return}
        do {
            let c=try DatabaseOutputSettings.normalized(config,requireMapping:false),credential=try secret(for:address,config:c,write:false)
            let bookmark=c["backend"] == .string("sqlite") ? try destinations?.bookmark(for:address):nil
            let identity=try metadataIdentity(c,filename:filename,address:address),lease=owner.lease()
            isBusy=true;owner.setExecutionBusy(true);workspace.isPreparingData=true;workspace.cancelDataPreparation={ [weak self] in self?.cancel() };let reservation=workspace.dataPreparationReservationID
            status="Reading destination metadata…"
            task=Task { [self] in
                defer {isBusy=false;task=nil;if owner.isCurrent(lease) {owner.setExecutionBusy(false)};if workspace.dataPreparationReservationID==reservation {workspace.isPreparingData=false;workspace.cancelDataPreparation=nil}}
                do {let value=try await execute("inspect",c,credential,UUID().uuidString,nil,"",bookmark,filename);try Task.checkCancellation();guard owner.isCurrent(lease),try metadataIdentity(c,filename:filename,address:address)==identity else {throw CancellationError()};metadata[address.localID] = .init(identity:identity,value:value);status="Connected · table metadata read"}
                catch {status=Task.isCancelled ? "Metadata request cancelled.":"Could not inspect this connection. Check its saved credential, TLS, database and selected local file. "+error.localizedDescription}
            }
        } catch {status=error.localizedDescription}
    }
    func chooseDatabase(address:WorkflowNodeAddress)throws->String? {try destinations?.chooseDatabase(for:address,lease:owner.lease())}
    func cancel(){task?.cancel()}
    func acknowledge(_ id:String)throws {
        guard !owner.isBusy,!isBusy,!(workspace?.isBusy ?? true) else {throw WorkflowArchiveError("Wait for the active operation before acknowledging a write.")}
        try ledger.acknowledge(id);records=try ledger.load();status="Previous unknown outcome acknowledged. A new Run Flow will be a new write."
    }
    func latest(for address:WorkflowNodeAddress)->DatabaseOutputLedger.Record? {records.first {$0.nodeID==address.localID}}
    func reloadRecords(){do {records=try ledger.load()} catch {status="Database receipt ledger could not be read; no write was repeated."}}
    struct Authority {let configuration:[String:JSONValue];let credential:DatabaseSourceCredentials;let filename:String;let bookmark:Data?;let targetIdentity:String;let retainResponses:Bool}
    func captureAuthority(_ node:DataFlowNode)throws->Authority {
        guard case .object(let raw)=node.config[APITableSettings.isAPI(node.config) ? "api":"database"] else {throw WorkflowFileLeaseError.unsafePath}
        let config=try (APITableSettings.isAPI(node.config) ? APITableSettings.normalized(raw):DatabaseOutputSettings.normalized(raw)),address=WorkflowNodeAddress(domain:.data,localID:node.id)
        let credential=try secret(for:address,config:config,write:true),filename=node.config["databaseFilename"]?.stringValue ?? "analytics.sqlite"
        let local=config["backend"] == .string("sqlite"),bookmark=local ? try destinations?.bookmark(for:address):nil
        var target=try endpoint(config);target["url"]=config["url"];target["method"]=config["method"];target["schema"]=config["schema"];target["table"]=config["table"];if local {target["folder"] = .string(destinations?.databasePath(for:address) ?? "");target["filename"] = .string(filename)}
        let identity=WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(target))
        guard !(try ledger.load()).contains(where:{!$0.acknowledged && $0.targetIdentity==identity && ["intent","unknown"].contains($0.receipt["status"]?.stringValue ?? "")}) else {throw WorkflowArchiveError("A previous write to this destination has an unknown outcome. Inspect and acknowledge its receipt before running again.")}
        return .init(configuration:config,credential:credential,filename:filename,bookmark:bookmark,targetIdentity:identity,retainResponses:owner.runHistoryPolicy.componentOutputs)
    }
    func write(node:DataFlowNode,artifact:TableViewQuerySession.Artifact,preparationID:String,window:ExecutionBudgetWindow,authority:Authority)async throws->[String:JSONValue] {
        guard let workspace else {throw WorkflowFileLeaseError.unsafePath}
        guard case .object(let raw)=node.config[APITableSettings.isAPI(node.config) ? "api":"database"] else {throw WorkflowFileLeaseError.unsafePath}
        let address=WorkflowNodeAddress(domain:.data,localID:node.id),config=try (APITableSettings.isAPI(node.config) ? APITableSettings.normalized(raw):DatabaseOutputSettings.normalized(raw))
        guard authority.configuration==config else {throw WorkflowFileLeaseError.unsafePath}
        let credential=authority.credential,filename=authority.filename,bookmark=authority.bookmark,identity=authority.targetIdentity,operation=UUID().uuidString
        let storage=try workspace.executionPolicy(for:owner.projectID).storageLimitBytes.map {try WorkflowStorageAdmission(workspace:root,accountingRoot:workspace.applicationDirectory).reserve(scopes:[.init(namespace:.databaseOutputRecords,components:[owner.projectID])],peakBytes:2_000_000,limitBytes:$0)}
        defer {storage?.close()}
        let fd=try WorkflowEvaluationOwnership.openDirectory(artifact.indexURL.deletingLastPathComponent());defer {Darwin.close(fd)}
        let input=Darwin.openat(fd,artifact.indexURL.lastPathComponent,O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)
        guard input>=0 else {throw WorkflowFileLeaseError.unsafePath};let handle=FileHandle(fileDescriptor:input,closeOnDealloc:true);defer {try? handle.close()}
        var effective=config;effective["timeoutSeconds"] = .integer(Int64(max(1,min(Int(config["timeoutSeconds"]?.integerValue ?? 30),Int(try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime))))))
        if APITableSettings.isAPI(node.config),!authority.retainResponses {effective["responsePath"] = .string("")}
        try Task.checkCancellation()
        let record=DatabaseOutputLedger.Record(id:operation,nodeID:node.id,preparationID:preparationID,targetIdentity:identity,configuration:config,sourceSHA256:artifact.sourceSHA256,createdAt:Date(),receipt:["status":.string("intent")])
        try ledger.begin(record);records=try ledger.load()
        var receipt:[String:JSONValue]
        do {if APITableSettings.isAPI(node.config) {receipt=try await apiExecute("write",effective,credential,operation,handle,artifact.sourceSHA256,workspace.executionPolicy(for:owner.projectID).budget.resources?.maximumResidentBytes)} else {receipt=try await execute("write",effective,credential,operation,handle,artifact.sourceSHA256,bookmark,filename)}}
        catch {receipt=["schemaVersion":1,"kind":.string(APITableSettings.isAPI(node.config) ? "apiWrite":"databaseWrite"),"operationID":.string(operation),"status":.string("unknown"),"message":.string("The external service did not return a confirmed outcome. Inspect the destination before retrying.")]}
        guard receipt["operationID"] == .string(operation),receipt["status"] != .string("committed") || receipt["sourceSHA256"] == .string(artifact.sourceSHA256),["committed","notStarted","rolledBack","unknown"].contains(receipt["status"]?.stringValue ?? "") else {throw WorkflowArchiveError("Unverified database receipt; the durable intent remains unknown.")}
        if APITableSettings.isAPI(node.config),!authority.retainResponses {
            if case .array(let requests)=receipt["requests"] {receipt["requests"] = .array(requests.map {item in guard case .object(var row)=item else{return item};row.removeValue(forKey:"responseJSON");return .object(row)})}
            receipt["responseCapture"] = .string("disabledByRunPolicy")
        }
        try ledger.finish(id:operation,receipt:receipt);records=try ledger.load()
        guard receipt["status"] == .string("committed") else {throw WorkflowArchiveError(receipt["message"]?.stringValue ?? "Database write did not commit. Inspect its receipt.")}
        return receipt
    }
}
