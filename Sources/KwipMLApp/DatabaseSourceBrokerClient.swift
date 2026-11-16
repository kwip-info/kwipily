import Foundation
import KwipMLCore
import KwipMLDatabaseNetworking

/// XPC carries private credentials to the dedicated source broker; ordinary workers receive only IPC data.
final class DatabaseSourceBrokerClient {
    func capture(options: [String: JSONValue], credentials: DatabaseSourceCredentials, output: FileHandle, preview: Bool = false,memoryLimit:Int?=nil,operationSeconds:Int?=nil) async throws -> Data {
        let request = try DatabaseSourceServiceIdentity.request(options: options, preview: preview,memoryLimit:memoryLimit,operationSeconds:operationSeconds)
        let secret = try credentials.encoded()
        let requirement = try DatabaseSourceServiceIdentity.serviceRequirement()
        let holder = DatabaseSourceCall()
        return try await withTaskCancellationHandler(operation: {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                let connection = NSXPCConnection(serviceName: DatabaseSourceServiceIdentity.identifier)
                connection.remoteObjectInterface = NSXPCInterface(with: KwipMLDatabaseSourceProtocol.self)
                connection.setCodeSigningRequirement(requirement)
                guard holder.begin(connection, continuation: continuation) else { return }
                connection.invalidationHandler = { holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable)) }
                connection.interruptionHandler = { holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable)) }
                connection.resume()
                let proxy = connection.remoteObjectProxyWithErrorHandler { _ in holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable)) } as! KwipMLDatabaseSourceProtocol
                let launch = {
                    proxy.capture(request, credentials: secret, output: output) { bytes, error in
                        if error == nil, let bytes, bytes.count <= 65_536 { holder.finish(.success(bytes)) }
                        else { holder.finish(.failure(DatabaseSourceBrokerFailure.captureFailed)) }
                    }
                }
                #if DEBUG
                do {
                    if let bookmark = try DatabaseSourceServiceIdentity.developmentRootBookmark() {
                        proxy.authorizeDevelopmentRoot(bookmark) { allowed in
                            if allowed { launch() } else { holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable)) }
                        }
                    } else { launch() }
                } catch { holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable)) }
                #else
                launch()
                #endif
                // Request supervised cancellation at the outer deadline. Keep
                // ownership until the service acknowledges child exit; merely
                // invalidating XPC could release a still-running writer.
                DispatchQueue.global().asyncAfter(deadline: .now() + 305) { holder.cancel() }
            }
        }, onCancel: { holder.cancel() })
    }
}
private final class DatabaseSourceCall: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var preserveReceipt = false
    private var cancellationRequested = false
    private var connection: NSXPCConnection?
    private var continuation: CheckedContinuation<Data, Error>?
    func begin(_ connection: NSXPCConnection, continuation: CheckedContinuation<Data, Error>, preserveReceipt:Bool=false) -> Bool {
        lock.lock()
        if cancelled || cancellationRequested { lock.unlock(); connection.invalidate(); continuation.resume(throwing: CancellationError()); return false }
        self.connection = connection; self.continuation = continuation; self.preserveReceipt=preserveReceipt; lock.unlock(); return true
    }
    func cancel() {
        lock.lock(); cancellationRequested = true
        let connection = self.connection; lock.unlock()
        guard let connection else { return }
        let proxy = connection.remoteObjectProxyWithErrorHandler { _ in } as? KwipMLDatabaseSourceProtocol
        proxy?.cancelCapture { [self] joined in
            if joined { finish(.failure(CancellationError())) }
        }
    }
    func finish(_ result: Result<Data, Error>) {
        lock.lock()
        cancelled = true
        let pending = continuation, connection = self.connection
        let completedResult: Result<Data, Error>
        if cancellationRequested && !preserveReceipt {completedResult = .failure(CancellationError())} else {completedResult=result}
        continuation = nil; self.connection = nil; lock.unlock()
        guard let pending else { return }
        connection?.invalidationHandler = nil; connection?.interruptionHandler = nil; connection?.invalidate()
        pending.resume(with: completedResult)
    }
}

/// Same authenticated helper and cancellation/join ownership, with a read-only
/// descriptor for prepared input. A delivered terminal receipt wins over late cancel.
struct DatabaseOutputBrokerClient {
    func perform(action:String,options:[String:JSONValue],credentials:DatabaseSourceCredentials,operationID:String,table:FileHandle?,tableSHA256:String,directoryBookmark:Data?,filename:String,memoryLimit:Int?=nil)async throws->[String:JSONValue] {
        guard ["inspect","write"].contains(action) else {throw DatabaseSourceBrokerFailure.invalidRequest}
        let checked=try options["kind"] == .string("api") ? APITableSettings.normalized(options):DatabaseOutputSettings.normalized(options,requireMapping:action=="write")
        var folderHash=""
        if let directoryBookmark {
            var stale=false
            let folder=try URL(resolvingBookmarkData:directoryBookmark,options:[.withoutUI],relativeTo:nil,bookmarkDataIsStale:&stale)
            defer {folder.stopAccessingSecurityScopedResource()}
            folderHash=WorkflowArchiveCodec.hash(Data(folder.standardizedFileURL.resolvingSymlinksInPath().path.utf8))
        }
        let payload=try WorkflowArchiveCodec.encode(["memoryLimit":memoryLimit.map {JSONValue.integer(Int64($0))} ?? .null,"folderSHA256":JSONValue.string(folderHash),"schemaVersion":JSONValue.integer(1),"action":.string(action),"options":.object(checked),"operationID":.string(operationID),"tableSHA256":.string(tableSHA256),"filename":.string(filename)])
        guard payload.count<=65_536 else {throw DatabaseSourceBrokerFailure.invalidRequest}
        let secret=try credentials.encoded(),requirement=try DatabaseSourceServiceIdentity.serviceRequirement(),holder=DatabaseSourceCall()
        return try await withTaskCancellationHandler(operation:{
            try Task.checkCancellation()
            let bytes:Data=try await withCheckedThrowingContinuation { continuation in
                let connection=NSXPCConnection(serviceName:DatabaseSourceServiceIdentity.identifier)
                connection.remoteObjectInterface=NSXPCInterface(with:KwipMLDatabaseSourceProtocol.self);connection.setCodeSigningRequirement(requirement)
                guard holder.begin(connection,continuation:continuation,preserveReceipt:true) else {return}
                connection.invalidationHandler={holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable))}
                connection.interruptionHandler={holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable))}
                connection.resume()
                let proxy=connection.remoteObjectProxyWithErrorHandler {_ in holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable))} as! KwipMLDatabaseSourceProtocol
                let launch={proxy.databaseOutput(payload,credentials:secret,table:table,directoryBookmark:directoryBookmark) {data,error in
                    if error==nil,let data,data.count<=65_536 {holder.finish(.success(data))} else {let code=["invalid_request","invalid_input","database_output_scope","database_output_scope_request","database_output_scope_resolution","database_output_scope_stale","database_output_scope_directory","database_output_runtime","database_output_launch","database_output_receipt"].contains(error ?? "") ? error!:"database_operation_failed";holder.finish(.failure(NSError(domain:"KwipML.Database",code:1,userInfo:[NSLocalizedDescriptionKey:"Database service: "+code])))}
                }}
                #if DEBUG
                do {if let bookmark=try DatabaseSourceServiceIdentity.developmentRootBookmark() {proxy.authorizeDevelopmentRoot(bookmark) {if $0 {launch()} else {holder.finish(.failure(DatabaseSourceBrokerFailure.unavailable))}}} else {launch()}} catch {holder.finish(.failure(error))}
                #else
                launch()
                #endif
                DispatchQueue.global().asyncAfter(deadline:.now()+305){holder.cancel()}
            }
            let receipt=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            guard receipt["schemaVersion"] == .integer(1),receipt["kind"] == .string(options["kind"] == .string("api") ? "apiWrite":action=="inspect" ? "databaseMetadata":"databaseWrite") else {throw DatabaseSourceBrokerFailure.captureFailed}
            return receipt
        },onCancel:{holder.cancel()})
    }
}
