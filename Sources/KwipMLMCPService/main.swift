import Foundation
import KwipMLMCP

final class MCPService:NSObject,MCPServiceProtocol,@unchecked Sendable {
    let connection:NSXPCConnection
    let lock=NSLock()
    var broker:MCPLocalHTTPBroker?
    var webhookBroker:WebhookLocalHTTPBroker?
    var closed=false
    init(connection:NSXPCConnection){self.connection=connection}
    func configure(_ setup:Data,reply originalReply:@escaping(Data?,String?)->Void){
        let once=MCPReplyGate<(Data?,String?)>{originalReply($0.0,$0.1)}
        let reply:(Data?,String?)->Void={once.resolve(($0,$1))}
        guard setup.count<=32768 else{reply(nil,"invalid_setup");return}
        do {
            let options=try JSONDecoder().decode(MCPBrokerSetup.self,from:setup).validated()
            lock.lock();guard !closed else{lock.unlock();reply(nil,"unavailable");return};broker?.stop();broker=nil;lock.unlock()
            let next=try MCPLocalHTTPBroker(setup:options){[weak self]client,action,args in
                guard let self else{throw HTTPRefusal(status:503)}
                self.lock.lock();let active = !self.closed;self.lock.unlock()
                guard active else{throw HTTPRefusal(status:503)}
                let envelope=try JSONSerialization.data(withJSONObject:["clientID":client,"action":action,"arguments":JSONSerialization.jsonObject(with:args)])
                return try await withCheckedThrowingContinuation{continuation in
                    let once=MCPReplyGate<Result<Data,Error>>{continuation.resume(with:$0)}
                    DispatchQueue.global().asyncAfter(deadline:.now()+15){once.resolve(.failure(HTTPRefusal(status:503)))}
                    let proxy=self.connection.remoteObjectProxyWithErrorHandler{_ in once.resolve(.failure(HTTPRefusal(status:503)))}
                    guard let app=proxy as? MCPAppCommandProtocol else{once.resolve(.failure(HTTPRefusal(status:503)));return}
                    app.dispatch(envelope){data,error in
                        if let data,error==nil{once.resolve(.success(data))}else{once.resolve(.failure(HTTPRefusal(status:403)))}
                    }
                }
            }
            lock.lock();guard !closed else{lock.unlock();next.stop();reply(nil,"unavailable");return};broker=next;lock.unlock()
            next.start{port,error in
                if let port {reply(try? JSONSerialization.data(withJSONObject:["port":Int(port),"url":"http://127.0.0.1:\(port)/mcp"]),nil)}
                else{reply(nil,error ?? "listener_unavailable")}
            }
        }catch{reply(nil,"invalid_setup")}
    }
    func configureWebhook(_ port:Int,reply:@escaping(Int,String?)->Void) {
        guard (1024...65535).contains(port) else{reply(0,"invalid_port");return}
        do {
            lock.lock();guard !closed else{lock.unlock();reply(0,"unavailable");return};webhookBroker?.stop();webhookBroker=nil;lock.unlock()
            let next=try WebhookLocalHTTPBroker(port:port){[weak self] request in
                guard let self else{throw HTTPRefusal(status:503)}
                return try await withCheckedThrowingContinuation {continuation in
                    let once=MCPReplyGate<Result<Data,Error>>{continuation.resume(with:$0)}
                    DispatchQueue.global().asyncAfter(deadline:.now()+15){once.resolve(.failure(HTTPRefusal(status:503)))}
                    guard let app=self.connection.remoteObjectProxyWithErrorHandler({_ in once.resolve(.failure(HTTPRefusal(status:503)))}) as? MCPAppCommandProtocol else{once.resolve(.failure(HTTPRefusal(status:503)));return}
                    app.receiveWebhook(request){data,error in if let data,error==nil{once.resolve(.success(data))}else{once.resolve(.failure(HTTPRefusal(status:503)))}}
                }
            }
            lock.lock();guard !closed else{lock.unlock();next.stop();reply(0,"unavailable");return};webhookBroker=next;lock.unlock()
            next.start{port,error in reply(Int(port ?? 0),error)}
        }catch{reply(0,"listener_unavailable")}
    }
    func stop(reply:@escaping()->Void){lock.lock();let old=broker;broker=nil;let hooks=webhookBroker;webhookBroker=nil;lock.unlock();old?.stop();hooks?.stop();reply()}
    func close(){lock.lock();closed=true;let old=broker;broker=nil;let hooks=webhookBroker;webhookBroker=nil;lock.unlock();old?.stop();hooks?.stop()}
}
final class Listener:NSObject,NSXPCListenerDelegate {
    func listener(_ listener:NSXPCListener,shouldAcceptNewConnection connection:NSXPCConnection)->Bool {
        connection.setCodeSigningRequirement(MCPServiceIdentity.requirement(MCPServiceIdentity.appIdentifier))
        connection.exportedInterface=NSXPCInterface(with:MCPServiceProtocol.self)
        connection.remoteObjectInterface=NSXPCInterface(with:MCPAppCommandProtocol.self)
        let service=MCPService(connection:connection);connection.exportedObject=service
        connection.invalidationHandler={service.close()};connection.interruptionHandler={service.close()}
        connection.resume();return true
    }
}
let delegate=Listener();let listener=NSXPCListener.service();listener.delegate=delegate;listener.resume()
