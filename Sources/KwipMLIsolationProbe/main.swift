// Local gate06 harness, packaged separately; never include in a customer app.
import Foundation
import KwipMLNetworking
import KwipMLCore
import Darwin

func messageProbe() -> String {
    let connection = NetworkServiceIdentity.connect()
    defer { connection.invalidate() }
    let semaphore = DispatchSemaphore(value: 0)
    let lock = NSLock()
    var result = "timeout"
    let proxy = connection.remoteObjectProxyWithErrorHandler { _ in
        lock.lock(); result = "peer_denied"; lock.unlock(); semaphore.signal()
    } as! KwipMLNetworkProtocol
    proxy.perform(Data("{}".utf8)) { data, error in
        lock.lock(); result = error ?? "unexpected_success"; lock.unlock(); semaphore.signal()
    }
    _ = semaphore.wait(timeout: .now() + 8)
    lock.lock(); defer { lock.unlock() }
    return result
}
func socketProbe() -> Bool {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }; defer { close(fd) }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = UInt16(8767).bigEndian
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    return withUnsafePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0 }
    }
}
if CommandLine.arguments.contains("--worker") {
    let file = CommandLine.arguments.last!
    let networkAllowed = socketProbe()
    let readAllowed = (try? Data(contentsOf: URL(fileURLWithPath: file))) != nil
    print("worker_inherited_fd=\(fcntl(100, F_GETFD) >= 0) worker_secret_inherited=\(ProcessInfo.processInfo.environment["KWIPML_PROBE_SECRET"] != nil)")
    print("worker_network_allowed=\(networkAllowed) worker_ungranted_read_allowed=\(readAllowed) worker_broker=\(messageProbe())")
} else {
    print("parent_broker=\(messageProbe()) parent_network_allowed=\(socketProbe())")
    setenv("KWIPML_PROBE_SECRET", "not-a-real-credential", 1)
    let fd = open("/dev/null", O_RDONLY)
    guard fd >= 0, dup2(fd, 100) == 100 else { exit(2) }
    close(fd)
    _ = fcntl(100, F_SETFD, 0)
    defer { close(100); unsetenv("KWIPML_PROBE_SECRET") }
    let done = DispatchSemaphore(value: 0)
    do {
        let child = try ProcessPump.launch(
            executable: Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers/WorkerProbe"),
            arguments: ["--worker", CommandLine.arguments.last!],
            onLine: { print($0) }, onExit: { code, error in print("worker_exit=\(code) stderr_empty=\(error.isEmpty)"); done.signal() })
        let deadline = Date().addingTimeInterval(15)
        while done.wait(timeout: .now()) != .success {
            guard Date() < deadline else { child.terminate(); exit(3) }
            RunLoop.current.run(until: Date().addingTimeInterval(0.05))
        }
    } catch { print("worker_launch_failed"); exit(1) }
}
