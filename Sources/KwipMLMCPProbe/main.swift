import Foundation
import KwipMLMCP
import KwipMLCore

if CommandLine.arguments.contains("--demo-fixture") {
    let document=try WorkflowModelBoundary.consolidate(TextGenerationPracticeExample.document(device:"cpu"))
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    print(String(decoding:try encoder.encode(document),as:UTF8.self));exit(0)
}


/// Disposable signed sandbox harness. Uses synthetic protocol data only; real app
/// policy and canvas dispatch are separate acceptance stages.
final class Commands:NSObject,MCPAppCommandProtocol {
    func receiveWebhook(_ request:Data,reply:@escaping(Data?,String?)->Void){reply(nil,"not_permitted")}
    func dispatch(_ envelope:Data,reply:@escaping(Data?,String?)->Void){
        guard let value=try? JSONSerialization.jsonObject(with:envelope) as? [String:Any],value["clientID"] as? String=="probe" else{reply(nil,"not_permitted");return}
        if value["action"] as? String=="tools_list"{reply(Data("{\"tools\":[{\"name\":\"probe_inspect\",\"description\":\"Synthetic signed app IPC proof\",\"inputSchema\":{\"type\":\"object\",\"additionalProperties\":false}}]}".utf8),nil)}
        else if value["action"] as? String=="probe_inspect"{reply(Data("{\"sandboxAppRoundtrip\":true}".utf8),nil)}
        else{reply(nil,"not_permitted")}
    }
}
let commands=Commands()
let connection=MCPServiceIdentity.connection(appCommands:commands)
connection.invalidationHandler={print("probe_connection_invalidated");fflush(stdout);exit(1)}
let input=FileHandle.standardInput.readDataToEndOfFile()
guard let setup=try? JSONDecoder().decode(MCPBrokerSetup.self,from:input)else{exit(2)}
let proxy=connection.remoteObjectProxyWithErrorHandler{_ in print("probe_service_unavailable");fflush(stdout);exit(1)} as! MCPServiceProtocol
proxy.configure(try! JSONEncoder().encode(setup)){data,error in
    if let data,error==nil{print(String(decoding:data,as:UTF8.self));fflush(stdout)}else{print(error ?? "probe_failed");fflush(stdout);exit(1)}
}
RunLoop.main.run()
