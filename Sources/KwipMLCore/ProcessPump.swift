import Foundation
import Darwin

/// Retain through supervised execution. Cancellation closes only the app lifetime channel;
/// the supervisor, not this handle, owns worker termination and escalation.
public final class ProcessLifetimeHandle {
    public let process:Process
    private let lifetime:ProcessLifetimeWriter
    fileprivate init(process:Process,lifetime:ProcessLifetimeWriter) {self.process=process;self.lifetime=lifetime}
    public func cancel() {lifetime.close()}
    deinit {cancel()}
}

fileprivate final class ProcessLifetimeWriter {
    private let lock=NSLock()
    private var file:FileHandle?
    init(_ file:FileHandle) {self.file=file}
    func close() {
        lock.lock();defer {lock.unlock()}
        guard let file else {return}
        self.file=nil
        try? file.close()
    }
    deinit {close()}
}

/// Drains both pipes off the UI thread. A bounded mailbox backpressures stdout
/// without blocking main or stderr; FIFO delivery guarantees exit follows all lines.
public enum ProcessPump {
    public static func launch(executable:URL,arguments:[String],directory:URL?=nil,
                              onLine:@escaping(String)->Void,onExit:@escaping(Int32,String)->Void)throws->Process {
        let process=Process(),out=Pipe(),err=Pipe()
        configure(process,executable:executable,arguments:arguments,directory:directory,out:out,err:err)
        do {try process.run()} catch {close(out);close(err);throw error}
        drain(process,out:out,err:err,reject:{if process.isRunning {process.terminate()}},exited:{},onLine:onLine,onExit:onExit)
        return process
    }
    public static func launchWithLifetime(executable:URL,arguments:[String],directory:URL?=nil,
                                          onLine:@escaping(String)->Void,onExit:@escaping(Int32,String)->Void)throws->ProcessLifetimeHandle {
        let process=Process(),out=Pipe(),err=Pipe(),input=Pipe()
        let writer=ProcessLifetimeWriter(input.fileHandleForWriting)
        configure(process,executable:executable,arguments:arguments,directory:directory,out:out,err:err)
        // Foundation maps only this read handle to fd0. CLOEXEC prevents the child
        // retaining the parent writer and masking EOF after app exit/cancellation.
        for file in [input.fileHandleForReading,input.fileHandleForWriting] {
            let flags=fcntl(file.fileDescriptor,F_GETFD)
            guard flags >= 0,fcntl(file.fileDescriptor,F_SETFD,flags|FD_CLOEXEC) == 0 else {
                let code=errno;writer.close();close(input);close(out);close(err)
                throw NSError(domain:NSPOSIXErrorDomain,code:Int(code))
            }
        }
        process.standardInput=input.fileHandleForReading
        do {try process.run()} catch {writer.close();close(input);close(out);close(err);throw error}
        try? input.fileHandleForReading.close()
        let handle=ProcessLifetimeHandle(process:process,lifetime:writer)
        // Retain the channel, never the handle: dropping the caller's handle must
        // deliver EOF even while the background drain is waiting for child output.
        drain(process,out:out,err:err,reject:{writer.close()},exited:{writer.close()},onLine:onLine,onExit:onExit)
        return handle
    }
    private static func configure(_ process:Process,executable:URL,arguments:[String],directory:URL?,out:Pipe,err:Pipe) {
        process.executableURL=executable;process.arguments=arguments;process.currentDirectoryURL=directory
        // Workers never inherit account/service secrets or loader/proxy injection.
        // Runtime paths and execution settings are passed explicitly as arguments.
        let host = ProcessInfo.processInfo.environment
        var environment = ["PATH": "/usr/bin:/bin"]
        for key in ["HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"] {
            if let value = host[key] { environment[key] = value }
        }
        process.environment = environment
        process.standardOutput=out;process.standardError=err
    }
    private static func close(_ pipe:Pipe) {try? pipe.fileHandleForReading.close();try? pipe.fileHandleForWriting.close()}
    private static func drain(_ process:Process,out:Pipe,err:Pipe,reject:@escaping()->Void,exited:@escaping()->Void,
                              onLine:@escaping(String)->Void,onExit:@escaping(Int32,String)->Void) {
        let diagnostics = BoundedBuffer()
        let delivery = ProcessDeliveryMailbox(onLine: onLine, onExit: onExit)
        let errorDone = DispatchGroup(); errorDone.enter()
        DispatchQueue(label: "kwipml.worker.stderr").async {
            while true {
                let chunk = err.fileHandleForReading.availableData
                if chunk.isEmpty { break }
                diagnostics.append(chunk)
            }
            errorDone.leave()
        }
        DispatchQueue(label: "kwipml.worker.stdout").async {
            var buffer = Data()
            var rejectedEvent = false
            func rejectOversizedEvent() {
                rejectedEvent = true
                buffer.removeAll(keepingCapacity: false)
                diagnostics.append(Data("Worker event exceeded 1 MB".utf8))
                reject()
            }
            while true {
                let chunk = out.fileHandleForReading.availableData
                if chunk.isEmpty { break }
                // Continue draining after rejection, but never publish later success events.
                if rejectedEvent { continue }
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 10) {
                    guard buffer.distance(from: buffer.startIndex, to: newline) <= 1_000_000 else {
                        rejectOversizedEvent(); break
                    }
                    let line = String(decoding: buffer[..<newline], as: UTF8.self)
                    buffer.removeSubrange(...newline)
                    delivery.append(line)
                }
                if buffer.count > 1_000_000 { rejectOversizedEvent() }
            }
            if !buffer.isEmpty {
                let line = String(decoding: buffer, as: UTF8.self)
                delivery.append(line)
            }
            process.waitUntilExit(); exited(); errorDone.wait()
            let status = process.terminationStatus
            let code: Int32 = rejectedEvent && status == 0 ? 1 : status
            let stderr = diagnostics.string
            delivery.finish(code: code, stderr: stderr)
        }
    }
}
/// A single stdout producer and a main-queue consumer. Only the producer waits:
/// lifetime cancellation uses its independent writer lock, and stderr keeps draining.
/// Limits count decoded UTF-8 bytes as replacement characters can expand malformed
/// input. A valid <=1 MB input line expands to at most3 MB, below this mailbox cap.
private final class ProcessDeliveryMailbox {
    private static let maximumLines = 128
    private static let maximumBytes = 4 * 1024 * 1024
    private static let maximumCallbacksPerTurn = 32
    private static let turnNanoseconds: UInt64 = 4_000_000
    private let condition = NSCondition()
    private var pending: [(line: String, bytes: Int)] = []
    private var pendingBytes = 0
    private var terminal: (code: Int32, stderr: String)?
    private var scheduled = false
    private let onLine: (String) -> Void
    private let onExit: (Int32, String) -> Void

    init(onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String) -> Void) {
        self.onLine = onLine; self.onExit = onExit
    }
    func append(_ line: String) {
        let bytes = line.utf8.count
        condition.lock()
        while pending.count >= Self.maximumLines || pendingBytes + bytes > Self.maximumBytes {
            condition.wait()
        }
        pending.append((line, bytes)); pendingBytes += bytes
        let needsSchedule = !scheduled
        scheduled = true
        condition.unlock()
        if needsSchedule { schedule() }
    }
    func finish(code: Int32, stderr: String) {
        condition.lock()
        terminal = (code, stderr)
        let needsSchedule = !scheduled
        scheduled = true
        condition.unlock()
        if needsSchedule { schedule() }
    }
    private func schedule() {
        DispatchQueue.main.async { self.deliver() }
    }
    private func deliver() {
        let started = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<Self.maximumCallbacksPerTurn {
            condition.lock()
            if pending.isEmpty {
                let exit = terminal
                terminal = nil; scheduled = false
                condition.unlock()
                if let exit { onExit(exit.code, exit.stderr) }
                return
            }
            let item = pending.removeFirst()
            pendingBytes -= item.bytes
            condition.signal()
            condition.unlock()
            // Never hold the mailbox lock while client code runs. The active item
            // is at most one bounded line beyond the pending-byte cap.
            onLine(item.line)
            if DispatchTime.now().uptimeNanoseconds - started >= Self.turnNanoseconds { break }
        }
        // Keep scheduled true across turns: producer/finish cannot enqueue an
        // additional main closure while this one yields for normal UI activity.
        schedule()
    }
}
private final class BoundedBuffer {
    private let lock = NSLock()
    private var bytes = Data()
    func append(_ data: Data) { lock.lock(); defer { lock.unlock() }; bytes.append(data); if bytes.count > 8000 { bytes = bytes.suffix(8000) } }
    var string: String { lock.lock(); defer { lock.unlock() }; return String(decoding: bytes, as: UTF8.self) }
}
