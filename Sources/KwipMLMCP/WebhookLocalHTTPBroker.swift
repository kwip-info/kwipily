import Foundation
import Network
import MCP
import KwipMLCore

/// Independent local transport in the existing signed sandbox helper. No execution authority.
public final class WebhookLocalHTTPBroker:@unchecked Sendable {
    private let queue=DispatchQueue(label:"info.kwip.KwipML.webhooks")
    private let listener:NWListener
    private let dispatch:@Sendable(Data)async throws->Data
    private let lock=NSLock()
    private var closed=false
    private var connections:[UUID:NWConnection]=[:]
    public init(port:Int,dispatch:@escaping @Sendable(Data)async throws->Data)throws {
        guard (1024...65535).contains(port) else{throw HTTPRefusal(status:400)}
        self.dispatch=dispatch
        let parameters=NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host:.ipv4(.loopback),port:NWEndpoint.Port(rawValue:UInt16(port))!)
        parameters.allowLocalEndpointReuse=false;listener=try NWListener(using:parameters)
    }
    public func start(reply:@escaping @Sendable(UInt16?,String?)->Void) {
        listener.stateUpdateHandler={[weak self] state in guard let self else{return};switch state {case .ready:reply(self.listener.port?.rawValue,nil);case .failed:reply(nil,"listener_unavailable");self.stop();default:break}}
        listener.newConnectionHandler={[weak self] in self?.accept($0)};listener.start(queue:queue)
    }
    public func stop(){lock.lock();closed=true;let all=connections.values;connections.removeAll();lock.unlock();listener.cancel();all.forEach{$0.cancel()}}
    private func accept(_ c:NWConnection){lock.lock();guard !closed,connections.count<8 else{lock.unlock();c.cancel();return};let id=UUID();connections[id]=c;lock.unlock();c.start(queue:queue);queue.asyncAfter(deadline:.now()+15){[weak self]in self?.finish(id)};receive(c,id:id,parser:.init(webhook:true))}
    private func finish(_ id:UUID){lock.lock();let c=connections.removeValue(forKey:id);lock.unlock();c?.cancel()}
    private func receive(_ c:NWConnection,id:UUID,parser:MCPHTTPParser){c.receive(minimumIncompleteLength:1,maximumLength:16384){[weak self] bytes,_,complete,error in
        guard let self else{return};var parser=parser
        do {if let bytes,let request=try parser.append(bytes){Task{await self.handle(request,c:c,id:id)};return};if complete || error != nil{self.finish(id)}else{self.receive(c,id:id,parser:parser)}}catch{self.send(c,id:id,response:.init(status:(error as? HTTPRefusal)?.status ?? 400,code:"invalid_http"))}
    }}
    private func handle(_ request:HTTPRequest,c:NWConnection,id:UUID)async {
        do {
            guard let port=listener.port?.rawValue,request.header("host")=="127.0.0.1:\(port)" || request.header("host")=="localhost:\(port)",request.header("origin")==nil else{throw HTTPRefusal(status:403)}
            guard request.header("content-type")?.split(separator:";").first?.trimmingCharacters(in:.whitespaces).lowercased()=="application/json" else{throw HTTPRefusal(status:415)}
            let value=WebhookHTTPRequest(path:request.path ?? "",headers:request.headers,body:request.body ?? Data())
            let response=try await dispatch(JSONEncoder().encode(value))
            send(c,id:id,response:try JSONDecoder().decode(WebhookHTTPResponse.self,from:response))
        }catch{send(c,id:id,response:.init(status:(error as? HTTPRefusal)?.status ?? 503,code:"request_not_accepted"))}
    }
    private func send(_ c:NWConnection,id:UUID,response:WebhookHTTPResponse) {
        lock.lock();let active = !closed && connections[id] != nil;lock.unlock();guard active else{return}
        let bytes=(try? JSONEncoder().encode(response)) ?? Data()
        var content=Data("HTTP/1.1 \(response.status) Response\r\nConnection: close\r\nCache-Control: no-store\r\nContent-Type: application/json\r\nContent-Length: \(bytes.count)\r\n\r\n".utf8);content.append(bytes)
        c.send(content:content,completion:.contentProcessed{[weak self]_ in self?.finish(id)})
    }
    deinit{stop()}
}
