import Foundation
import Darwin
import CryptoKit
import KwipMLCore
import KwipMLDatabaseNetworking

private func emit(_ value: [String: Any]) {
    if let raw = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) {
        FileHandle.standardOutput.write(raw); FileHandle.standardOutput.write(Data([10]))
    }
}
@main struct DatabaseProbe {
    static func main() async {
        do {
            guard CommandLine.arguments.count >= 2 else { throw DatabaseSourceBrokerFailure.invalidRequest }
            let folder = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let raw = FileHandle.standardInput.readDataToEndOfFile()
            guard raw.count <= 65_536, case .object(let input) = try JSONDecoder().decode(JSONValue.self, from: raw),
                  case .object(let options) = input["options"], case .object(let secret) = input["credentials"] else { throw DatabaseSourceBrokerFailure.invalidRequest }
            let preview: Bool
            if case .bool(let value) = input["preview"] { preview = value } else { preview = false }
            let message = try DatabaseSourceServiceIdentity.request(options: options, preview: preview)
            let credentials = try JSONEncoder().encode(secret)
            let path = folder.appendingPathComponent("capture.ipc")
            let fd = open(path.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard fd >= 0 else { throw DatabaseSourceBrokerFailure.invalidRequest }
            let output = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            defer { try? output.close() }
            let connection = NSXPCConnection(serviceName: DatabaseSourceServiceIdentity.identifier)
            connection.remoteObjectInterface = NSXPCInterface(with: KwipMLDatabaseSourceProtocol.self)
            #if DEBUG
            if input["testMismatchedPeerSignature"] == .bool(true) {
                // An intentionally impossible exact peer requirement, solely
                // for proving the native XPC trust gate fails closed.
                connection.setCodeSigningRequirement("cdhash H\"0000000000000000000000000000000000000000\"")
            } else {
                connection.setCodeSigningRequirement(try DatabaseSourceServiceIdentity.serviceRequirement())
            }
            #else
            connection.setCodeSigningRequirement(try DatabaseSourceServiceIdentity.serviceRequirement())
            #endif
            let completed = ProbeCompletion(connection)
            connection.invalidationHandler = { completed.finish(nil, "invalidated") }
            connection.interruptionHandler = { completed.finish(nil, "interrupted") }
            connection.resume()
            let proxy = connection.remoteObjectProxyWithErrorHandler { _ in completed.finish(nil, "unavailable") } as! KwipMLDatabaseSourceProtocol
            let cancelDelay: Int
            if case .integer(let value) = input["cancelAfterMilliseconds"], (0...300_000).contains(value) { cancelDelay = Int(value) } else { cancelDelay = -1 }
            if cancelDelay >= 0 {
                DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(cancelDelay)) {
                    proxy.cancelCapture { joined in emit(["event": "cancelAcknowledged", "joined": joined]) }
                }
            }
            #if DEBUG
            let timer = DispatchSource.makeTimerSource(queue: .global())
            timer.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
            timer.setEventHandler {
                proxy.debugProcessIDs { service, worker in
                    if worker > 0 { emit(["event": "databaseWorker", "servicePID": service, "workerPID": worker]); timer.cancel() }
                }
            }
            timer.resume()
            defer { timer.cancel() }
            #endif
            let receipt = await withCheckedContinuation { continuation in
                completed.begin(continuation)
                let launch = { proxy.capture(message, credentials: credentials, output: output) { data, error in completed.finish(data, error) } }
                #if DEBUG
                do {
                    if let bookmark = try DatabaseSourceServiceIdentity.developmentRootBookmark() {
                        proxy.authorizeDevelopmentRoot(bookmark) { allowed in
                            if allowed { launch() } else { completed.finish(nil, "development_grant_refused") }
                        }
                    } else { launch() }
                } catch { completed.finish(nil, "development_grant_creation_failed") }
                #else
                launch()
                #endif
            }
            connection.invalidationHandler = nil; connection.interruptionHandler = nil; connection.invalidate()
            guard let receipt else { emit(["event": "captureFailed"]); Darwin.exit(1) }
            try output.synchronize()
            let data = try Data(contentsOf: path)
            let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard let value = try JSONSerialization.jsonObject(with: receipt) as? [String: Any], value["sha256"] as? String == hash, value["bytes"] as? Int == data.count else { throw DatabaseSourceBrokerFailure.captureFailed }
            emit(["event": "databaseCaptureVerified", "receipt": value])
        } catch { emit(["event": "probeFailed"]); Darwin.exit(1) }
    }
}
private final class ProbeCompletion: @unchecked Sendable {
    let lock = NSLock(); let connection: NSXPCConnection
    var continuation: CheckedContinuation<Data?, Never>?
    var finished = false
    init(_ connection: NSXPCConnection) { self.connection = connection }
    func begin(_ value: CheckedContinuation<Data?, Never>) {
        lock.lock(); defer { lock.unlock() }
        if finished { value.resume(returning: nil) } else { continuation = value }
    }
    func finish(_ data: Data?, _ error: String?) {
        lock.lock(); guard !finished else { lock.unlock(); return }; finished = true
        let pending = continuation; continuation = nil; lock.unlock()
        if error != nil {
            let safe = Set(["invalidated", "interrupted", "unavailable", "development_grant_refused", "development_grant_creation_failed", "invalid_request", "invalid_capture", "capture_failed"])
            emit(["event": "brokerFailure", "code": safe.contains(error!) ? error! : "capture_failed"])
        }
        pending?.resume(returning: error == nil ? data : nil)
    }
}
