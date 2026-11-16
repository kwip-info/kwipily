import Foundation
import KwipMLCore

public enum NetworkBrokerFailure:LocalizedError {
    case unavailable, signInRequired, deviceLimit, notEntitled, conflict, rateLimited
    public var errorDescription:String? {
        switch self {
        // This includes transport and helper availability failures; do not guess a cause.
        case .unavailable:return "The network service could not complete this request. Try again; if it continues, reopen Kwip ML."
        case .signInRequired:return "Authentication is required to complete this request."
        case .deviceLimit:return "This account has reached its device limit."
        case .notEntitled:return "This account does not have access to the requested service."
        case .conflict:return "The request conflicts with the current service state. Refresh and try again."
        case .rateLimited:return "The service is receiving too many requests. Wait before trying again."
        }
    }
}

/// Only the authenticated app can use this connection. No connection/credential reaches a worker.
public final class NetworkBrokerClient {
    public init() {}
    public func modelHub(_ request:ModelHubRequest)async throws->Data {
        _=try request.prepared()
        let data=try JSONEncoder().encode(request),cancel=ModelHubCancellation()
        return try await withTaskCancellationHandler(operation:{
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation {continuation in
                let connection=NetworkServiceIdentity.connect(),completion=ReplyCompletion(connection:connection,continuation:continuation)
                connection.invalidationHandler={completion.finish(.failure(NetworkBrokerFailure.unavailable))}
                connection.interruptionHandler={completion.finish(.failure(NetworkBrokerFailure.unavailable))}
                cancel.install(completion)
                guard !Task.isCancelled else {completion.finish(.failure(CancellationError()));return}
                let proxy=connection.remoteObjectProxyWithErrorHandler {_ in completion.finish(.failure(NetworkBrokerFailure.unavailable))} as! KwipMLNetworkProtocol
                proxy.modelHub(data) {bytes,error in
                    if error=="sign_in_required" {completion.finish(.failure(NetworkBrokerFailure.signInRequired))}
                    else if error==nil,let bytes,bytes.count<=request.maximumResponseBytes {completion.finish(.success(bytes))}
                    else {completion.finish(.failure(NetworkBrokerFailure.unavailable))}
                }
                DispatchQueue.global().asyncAfter(deadline:.now()+47) {completion.finish(.failure(NetworkBrokerFailure.unavailable))}
            }
        },onCancel:{cancel.cancel()})
    }
    public func perform(operation:String,body:[String:Any]?=nil,bearer:String?=nil) async throws->Data {
        var object:[String:Any]=["schemaVersion":1,"operation":operation]
        if let body {object["body"]=body};if let bearer {object["bearer"]=bearer}
        let message=try JSONSerialization.data(withJSONObject:object)
        _=try NetworkBrokerPolicy.prepare(message)
        return try await withCheckedThrowingContinuation { continuation in
            let connection=NetworkServiceIdentity.connect()
            let completion=ReplyCompletion(connection:connection,continuation:continuation)
            connection.invalidationHandler={completion.finish(.failure(NetworkBrokerFailure.unavailable))}
            connection.interruptionHandler={completion.finish(.failure(NetworkBrokerFailure.unavailable))}
            let proxy=connection.remoteObjectProxyWithErrorHandler {_ in completion.finish(.failure(NetworkBrokerFailure.unavailable))} as! KwipMLNetworkProtocol
            proxy.perform(message) {data,error in
                if error=="sign_in_required" {completion.finish(.failure(NetworkBrokerFailure.signInRequired))}
                else if error=="device_limit" {completion.finish(.failure(NetworkBrokerFailure.deviceLimit))}
                else if error=="not_entitled" {completion.finish(.failure(NetworkBrokerFailure.notEntitled))}
                else if error=="conflict" {completion.finish(.failure(NetworkBrokerFailure.conflict))}
                else if error=="rate_limited" {completion.finish(.failure(NetworkBrokerFailure.rateLimited))}
                else if error==nil,let data,data.count<=NetworkBrokerPolicy.maximumResponseBytes {completion.finish(.success(data))}
                else {completion.finish(.failure(NetworkBrokerFailure.unavailable))}
            }
            DispatchQueue.global().asyncAfter(deadline:.now()+22) {completion.finish(.failure(NetworkBrokerFailure.unavailable))}
        }
    }
}
private final class ReplyCompletion:@unchecked Sendable {
    let lock=NSLock()
    var continuation:CheckedContinuation<Data,Error>?
    let connection:NSXPCConnection
    init(connection:NSXPCConnection,continuation:CheckedContinuation<Data,Error>) {self.connection=connection;self.continuation=continuation}
    func finish(_ result:Result<Data,Error>) {
        lock.lock();let pending=continuation;continuation=nil;lock.unlock()
        guard let pending else{return}
        connection.invalidationHandler=nil;connection.interruptionHandler=nil;connection.invalidate()
        pending.resume(with:result)
    }
}

private final class ModelHubCancellation:@unchecked Sendable {
    private let lock=NSLock()
    private var pending:ReplyCompletion?
    private var cancelled=false
    func install(_ value:ReplyCompletion) {lock.lock();pending=value;let stop=cancelled;lock.unlock();if stop {value.finish(.failure(CancellationError()))}}
    func cancel() {lock.lock();cancelled=true;let value=pending;lock.unlock();value?.finish(.failure(CancellationError()))}
}
