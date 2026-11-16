import Foundation
import KwipMLCore
import KwipMLNetworking

final class ServiceBudget {
    let lock = NSLock()
    private var count = 0
    func acquire() -> Bool { lock.lock(); defer { lock.unlock() }; guard count < 4 else { return false }; count += 1; return true }
    func release() { lock.lock(); defer { lock.unlock() }; count -= 1 }
}
final class NetworkService: NSObject, KwipMLNetworkProtocol {
    let budget: ServiceBudget
    let lock = NSLock()
    var tasks: [UUID: Task<Void, Never>] = [:]
    var closed = false
    init(budget: ServiceBudget) { self.budget = budget }
    func finish(_ id: UUID) {
        budget.release(); lock.lock(); tasks.removeValue(forKey: id); lock.unlock()
    }
    func close() {
        lock.lock(); closed = true; let outstanding = Array(tasks.values); lock.unlock()
        outstanding.forEach { $0.cancel() }
    }
    func modelHub(_ message:Data,reply:@escaping(Data?,String?)->Void) {
        guard message.count<=8192,let object=try? JSONSerialization.jsonObject(with:message) as? [String:Any],Set(object.keys).isSubset(of:["action","repository","revision","file","offset","total","bearer"]),let request=try? JSONDecoder().decode(ModelHubRequest.self,from:message),(try? request.prepared()) != nil else {reply(nil,"invalid_request");return}
        lock.lock()
        guard !closed,budget.acquire() else {lock.unlock();reply(nil,"unavailable");return}
        let id=UUID()
        let task=Task {
            defer {finish(id)}
            do {reply(try await ModelHubTransport().perform(request),nil)}
            catch NetworkBrokerFailure.signInRequired {reply(nil,"sign_in_required")}
            catch {reply(nil,"request_failed")}
        }
        tasks[id]=task;lock.unlock()
    }
    func perform(_ message: Data, reply: @escaping (Data?, String?) -> Void) {
        guard (try? NetworkBrokerPolicy.prepare(message)) != nil else { reply(nil, "invalid_request"); return }
        lock.lock()
        guard !closed, budget.acquire() else { lock.unlock(); reply(nil, "unavailable"); return }
        let id = UUID()
        let task = Task {
            defer { finish(id) }
            do { reply(try await BoundedNetworkTransport().perform(message), nil) }
            catch NetworkBrokerFailure.signInRequired {reply(nil,"sign_in_required")}
            catch NetworkBrokerFailure.deviceLimit {reply(nil,"device_limit")}
            catch NetworkBrokerFailure.notEntitled {reply(nil,"not_entitled")}
            catch NetworkBrokerFailure.conflict {reply(nil,"conflict")}
            catch NetworkBrokerFailure.rateLimited {reply(nil,"rate_limited")}
            catch { reply(nil, "request_failed") }
        }
        tasks[id] = task
        lock.unlock()
    }
}
final class Listener: NSObject, NSXPCListenerDelegate {
    let budget = ServiceBudget()
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        guard let requirement=try? NetworkServiceIdentity.clientRequirement() else {return false}
        connection.setCodeSigningRequirement(requirement)
        let service = NetworkService(budget: budget)
        connection.exportedInterface = NSXPCInterface(with: KwipMLNetworkProtocol.self)
        connection.exportedObject = service
        connection.invalidationHandler = { service.close() }
        connection.interruptionHandler = { service.close() }
        connection.resume()
        return true
    }
}
let delegate = Listener()
let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
