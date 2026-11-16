import Foundation
import Combine
import KwipMLCore
import KwipMLMCP

struct MCPAuditEntry:Codable,Identifiable {
    let id:String
    let at:Date
    let clientID:String
    let action:String
    let projectID:String?
    let outcome:String
}
@MainActor final class MCPController:NSObject,ObservableObject,MCPAppCommandProtocol {
    private static var instances:[String:MCPController]=[:]
    static func shared(directory:URL)->MCPController {
        let key=directory.standardizedFileURL.path
        if let instance=instances[key]{return instance}
        let instance=MCPController(directory:directory);instances[key]=instance;return instance
    }
    let directory:URL
    let policyStore:MCPSettingsStore
    let secrets:any MCPSecretStorage
    @Published private(set) var settings=MCPSettings.factory
    @Published private(set) var endpoint:String?
    @Published private(set) var status="Stopped"
    @Published private(set) var error:String?
    @Published private(set) var audit:[MCPAuditEntry]=[]
    private let startsTransport:Bool
    private var connection:NSXPCConnection?
    private var generation=UUID()
    private var acceptingRequests=false
    /// Injectable command authority for affected tests; native operation uses the
    /// app gateway. Policy is always checked there, not delegated to the listener.
    var commandHandler:((String,String,[String:JSONValue]) async throws->[String:JSONValue])?
    lazy var gateway=MCPWorkflowGateway(controller:self)
    init(directory:URL,secrets:(any MCPSecretStorage)?=nil,startConfigured:Bool=true){
        startsTransport=startConfigured
        self.directory=directory;policyStore=MCPSettingsStore(root:directory)
        #if DEBUG
        let fixtureSecrets:(any MCPSecretStorage)?=MCPDevelopmentFixture.directory.map{MCPDevelopmentFixture.Secrets(root:$0)}
        #else
        let fixtureSecrets:(any MCPSecretStorage)?=nil
        #endif
        self.secrets=secrets ?? fixtureSecrets ?? MCPSecretKeychain(namespace:Bundle.main.bundleIdentifier ?? "info.kwip.KwipML.POC")
        super.init()
        do{settings=try policyStore.load();if let bytes=try policyStore.loadAudit(){audit=Array(try JSONDecoder().decode([MCPAuditEntry].self,from:bytes).suffix(settings.auditRetention))};if settings.enabled && startConfigured{start()}}
        catch{settings = .factory;self.error=error.localizedDescription;status="Unavailable"}
    }
    var componentChoices:[(id:String,title:String)] {
        let workspace=ProjectWindowRegistry.shared.restorationWorkspace
        guard let data=try? workspace.workflowDataRegistry(),let entries=try? WorkflowCanvasAdapter.catalog(modelRegistry:workspace.registry,dataRegistry:data)else{return []}
        var choices=Dictionary(entries.map{($0.iconType,$0.component.title)},uniquingKeysWith:{first,_ in first})
        // Retain visibility of an old task-specific denial after palette consolidation.
        if settings.disabledComponents.contains("workflow.kerasClassifier") {choices["workflow.kerasClassifier"]="Prebuilt models · existing Keras restriction"}
        return choices.map{(id:$0.key,title:$0.value)}.sorted{$0.title.localizedStandardCompare($1.title) == .orderedAscending}
    }
    func clearAudit(){audit.removeAll();persistAudit()}
    private func persistAudit(){do{try policyStore.saveAudit(JSONEncoder().encode(audit))}catch{self.error="MCP receipts could not be saved. Review local storage permissions."}}
    static func stopAll(){for instance in instances.values {instance.stop(persist:false)}}
    var availableProjects:[ProjectRecord] {(try? ProjectLibrary(directory:directory.appendingPathComponent("projects")).listSources()) ?? []}
    func save(_ proposed:MCPSettings)throws {
        let value=try proposed.validated()
        let needsRestart = !settings.enabled || settings.fixedPort != value.fixedPort || Set(settings.clients.map(\.id)) != Set(value.clients.map(\.id)) || endpoint == nil
        try policyStore.save(value)
        settings=value;error=nil
        audit=Array(audit.suffix(value.auditRetention));persistAudit()
        gateway.policyChanged()
        if value.enabled{if startsTransport && needsRestart{start()}}else{stop(persist:false)}
    }
    func stop(persist:Bool=true){
        generation=UUID();acceptingRequests=false;endpoint=nil;status="Stopped"
        if persist{var next=settings;next.enabled=false;do{try policyStore.save(next);settings=next;error=nil}catch{self.error=error.localizedDescription;settings.enabled=false}}
        gateway.policyChanged()
        if let connection{(connection.remoteObjectProxy as? MCPServiceProtocol)?.stop{}}
        connection?.invalidate();connection=nil
    }
    func addClient(name:String)throws->String {
        var next=settings;let id=UUID().uuidString,token=try MCPSecretKeychain.makeToken()
        let read:Set<MCPAction>=[.catalogList,.catalogGet,.projectsList,.projectOpen,.workflowInspect,.workflowValidate,.patchPreview,.jobsList,.jobGet,.resultsRead,.lessonRead]
        next.clients.append(.init(id:id,name:name,actions:read))
        try secrets.write(token,clientID:id)
        do{try save(next)}catch{try? secrets.remove(clientID:id);throw error}
        return id
    }
    func revokeClient(_ id:String)throws {
        var next=settings;next.clients.removeAll{$0.id==id}
        // Policy and live access revoke first even if removing the Keychain item
        // fails; a leftover credential can no longer authenticate to this service.
        try save(next);try secrets.remove(clientID:id)
    }
    func connectionConfiguration(clientID:String)throws->String {
        guard settings.clients.contains(where:{$0.id==clientID}),let endpoint,
              let token=try secrets.read(clientID:clientID) else{throw MCPSecretFailure.unavailable}
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys,.withoutEscapingSlashes]
        let configuration:JSONValue = .object(["mcpServers":.object(["kwip-ml":.object(["url":.string(endpoint),"headers":.object(["Authorization":.string("Bearer "+token)])])])])
        return String(decoding:try encoder.encode(configuration),as:UTF8.self)
    }
    private func start(){
        generation=UUID();acceptingRequests=false;let expected=generation;endpoint=nil;status="Starting"
        connection?.invalidate();connection=nil
        do {
            var clients:[MCPBrokerClient]=[]
            for client in settings.clients {
                guard let token=try secrets.read(clientID:client.id)else{throw MCPSecretFailure.unavailable}
                clients.append(.init(id:client.id,tokenSHA256:MCPHTTPAdmission.tokenDigest(token)))
            }
            let setup=try JSONEncoder().encode(MCPBrokerSetup(fixedPort:settings.fixedPort,clients:clients))
            let connection=MCPServiceIdentity.connection(appCommands:self);self.connection=connection
            let lost:()->Void={ [weak self] in Task{@MainActor in
                guard let self,self.generation==expected else{return};self.acceptingRequests=false;self.endpoint=nil;self.status="Unavailable";self.error="MCP connection stopped. Restart it from Settings."
            }}
            connection.invalidationHandler=lost;connection.interruptionHandler=lost
            guard let proxy=connection.remoteObjectProxyWithErrorHandler({_ in lost()}) as? MCPServiceProtocol else{throw MCPSecretFailure.unavailable}
            proxy.configure(setup){[weak self] data,failure in Task{@MainActor in
                guard let self,self.generation==expected,self.settings.enabled else{return}
                guard failure==nil,let data,let object=try? JSONSerialization.jsonObject(with:data) as? [String:Any],
                      let url=object["url"] as? String,let components=URLComponents(string:url),components.scheme=="http",components.host=="127.0.0.1",components.path=="/mcp",let port=components.port,
                      self.settings.fixedPort==nil || self.settings.fixedPort==port else{
                    self.status="Unavailable";self.error=self.settings.fixedPort==nil ? "MCP listener could not start." : "The selected port is unavailable. Choose another port or Automatic.";return
                }
                self.acceptingRequests=true;self.endpoint=url;self.status="Running";self.error=nil
            }}
        }catch{endpoint=nil;status="Unavailable";self.error=error.localizedDescription}
    }
    nonisolated func receiveWebhook(_ request:Data,reply:@escaping(Data?,String?)->Void){reply(nil,"not_permitted")}
    nonisolated func dispatch(_ envelope:Data,reply:@escaping(Data?,String?)->Void){
        guard envelope.count<=1_048_576 else{reply(nil,"invalid_request");return}
        Task{@MainActor [weak self] in
            guard let self else{reply(nil,"unavailable");return}
            var admitted:MCPDispatchEnvelope?
            do {
                let request=try JSONDecoder().decode(MCPDispatchEnvelope.self,from:envelope);admitted=request
                guard self.settings.enabled,(!self.startsTransport || self.acceptingRequests),self.settings.clients.contains(where:{$0.id==request.clientID})else{throw MCPSettings.Refusal(code:"not_permitted")}
                let value:[String:JSONValue]
                if let handler=self.commandHandler{value=try await handler(request.clientID,request.action,request.arguments)}else{value=try await self.gateway.dispatch(clientID:request.clientID,action:request.action,arguments:request.arguments)}
                guard self.settings.enabled,(!self.startsTransport || self.acceptingRequests),self.settings.clients.contains(where:{$0.id==request.clientID})else{throw MCPSettings.Refusal(code:"not_permitted")}
                if let action=MCPAction(rawValue:request.action){try self.settings.authorize(clientID:request.clientID,projectID:action.requiresProject ? request.arguments["projectID"]?.stringValue:nil,action:action,requiresValues:request.action == "results_read" && request.arguments["includeValues"] == .bool(true))}
                let metadataActions:Set<String>=["catalog_list","catalog_get","projects_list","lesson_read","asset_status_read","workflow_inspect"]
                // Metadata navigation sanitizes the source before selecting a path.
                // Re-sanitizing string chunks would corrupt paths/cursors mid-value.
                let output = self.commandHandler == nil && metadataActions.contains(request.action) ? JSONValue.object(value) : MCPOutputBoundary.sanitize(.object(value))
                let data=try JSONEncoder().encode(output)
                guard data.count<=MCPOutputBoundary.responseByteLimit(action:request.action) else{throw MCPSettings.Refusal(code:"result_too_large")}
                self.record(clientID:request.clientID,action:request.action,projectID:request.arguments["projectID"]?.stringValue,outcome:value["isError"] == .bool(true) ? "refused":"completed")
                reply(data,nil)
            }catch{
                // Do not echo request contents, credentials, private paths or an
                // unknown project identity into the caller's error response.
                let code=(error as? MCPSettings.Refusal)?.code ?? "request_refused"
                if let request=admitted,self.settings.clients.contains(where:{$0.id==request.clientID}),MCPAction(rawValue:request.action) != nil{self.record(clientID:request.clientID,action:request.action,projectID:nil,outcome:"refused")}
                let response=(error as? MCPWorkflowGateway.PatchDecodingFailure)?.response ?? ["isError":JSONValue.bool(true),"code":.string(code),"message":.string("The request was not applied. Review permissions, revision, drafts and component validation.")]
                reply(try? JSONEncoder().encode(response),nil)
            }
        }
    }
    private func record(clientID:String,action:String,projectID:String?,outcome:String){
        guard settings.auditRetention>0 else{clearAudit();return}
        let knownProject=projectID.flatMap {id in UUID(uuidString:id)?.uuidString==id && settings.projects[id] != nil ? id:nil}
        audit.append(.init(id:UUID().uuidString,at:Date(),clientID:clientID,action:action,projectID:knownProject,outcome:outcome))
        if audit.count>settings.auditRetention{audit.removeFirst(audit.count-settings.auditRetention)}
        persistAudit()
    }
}
private struct MCPDispatchEnvelope:Decodable {
    let clientID:String
    let action:String
    let arguments:[String:JSONValue]
}
