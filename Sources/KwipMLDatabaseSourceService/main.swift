import Foundation
import Darwin
import KwipMLCore
import KwipMLDatabaseNetworking
import OSLog
import CryptoKit

/// Only this helper receives network.client. Host/table restrictions are enforced
/// by its fixed trusted adapter; they are not a per-host kernel sandbox claim.
private final class CaptureBudget {
    private let lock = NSLock()
    private var busy = false
    func acquire() -> Bool { lock.lock(); defer { lock.unlock() }; guard !busy else { return false }; busy = true; return true }
    func release() { lock.lock(); busy = false; lock.unlock() }
}
private final class DatabaseService: NSObject, KwipMLDatabaseSourceProtocol {
    let budget: CaptureBudget
    private let lock = NSLock()
    private var process: Process?
    private var lifetimeWrite: FileHandle?
    private var active = false
    #if DEBUG
    private var developmentRoot: URL?
    #endif
    private var cancellationReplies: [(Bool) -> Void] = []
    private var closed = false
    init(budget: CaptureBudget) { self.budget = budget }
    func close() {
        lock.lock(); closed = true; let child = process, pipe = lifetimeWrite; lifetimeWrite = nil; lock.unlock()
        try? pipe?.close()
        guard let child else { return }
        if child.isRunning { child.terminate() }
        DispatchQueue.global().asyncAfter(deadline: .now() + 2) { if child.isRunning { Darwin.kill(child.processIdentifier, SIGKILL) } }
    }
    func cancelCapture(reply: @escaping (Bool) -> Void) {
        lock.lock()
        if !active { lock.unlock(); close(); reply(true); return }
        cancellationReplies.append(reply); lock.unlock(); close()
    }
    private func finish() {
        lock.lock(); process = nil; lifetimeWrite = nil; active = false
        let replies = cancellationReplies; cancellationReplies = []; lock.unlock()
        #if DEBUG
        developmentRoot?.stopAccessingSecurityScopedResource(); developmentRoot = nil
        #endif
        replies.forEach { $0(true) }; budget.release()
    }
    #if DEBUG
    func authorizeDevelopmentRoot(_ bookmark: Data, reply: @escaping (Bool) -> Void) {
        let logger = Logger(subsystem: "info.kwip.KwipML.DatabaseSource", category: "developmentGrant")
        guard bookmark.count <= 65_536 else { reply(false); return }
        do {
            var stale = false
            let root = try URL(resolvingBookmarkData: bookmark, options: [.withoutUI], relativeTo: nil, bookmarkDataIsStale: &stale)
            var app = Bundle.main.bundleURL
            while app.pathExtension != "app", app.path != "/" { app.deleteLastPathComponent() }
            let pointer = app.appendingPathComponent("Contents/Resources/ProjectRoot.txt")
            let expected = try String(contentsOf: pointer, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedPath = root.standardizedFileURL.resolvingSymlinksInPath().path
            let expectedPath = URL(fileURLWithPath: expected).standardizedFileURL.resolvingSymlinksInPath().path
            guard DatabaseSourceServiceIdentity.matchesDevelopmentRoot(root, expected: URL(fileURLWithPath: expected), bookmarkIsStale: stale) else {
                logger.error("Development bookmark identity refused stale=\(stale, privacy: .public) actual=\(root.path, privacy: .public) canonical=\(resolvedPath, privacy: .public) expected=\(expectedPath, privacy: .public)")
                reply(false); return
            }
            // The bookmark conveys identity only, never filesystem authority.
            // Finite read-only code grants are fixed by DEBUG assembly.
            let entry = root.appendingPathComponent("runner/main.py")
            guard let handle = try? FileHandle(forReadingFrom: entry) else {
                logger.error("Development bookmark read authority unavailable"); reply(false); return
            }
            try handle.close()
            // No write is performed. Merely opening this fixed code file for
            // writing must fail, or the transferred scope is too broad.
            let writable = open(entry.path, O_WRONLY | O_NOFOLLOW | O_CLOEXEC)
            guard writable < 0 else {
                Darwin.close(writable); root.stopAccessingSecurityScopedResource()
                logger.error("Development bookmark write authority refused"); reply(false); return
            }
            lock.lock()
            guard !active, !closed else { lock.unlock(); root.stopAccessingSecurityScopedResource(); reply(false); return }
            developmentRoot?.stopAccessingSecurityScopedResource(); developmentRoot = root
            lock.unlock(); reply(true)
        } catch { logger.error("Development bookmark resolution failed"); reply(false) }
    }
    func debugProcessIDs(reply: @escaping (Int32, Int32) -> Void) {
        lock.lock(); let pid = process?.processIdentifier ?? 0; lock.unlock()
        reply(getpid(), pid)
    }
    #endif
    private func runtime() throws -> RuntimeLocation {
        var app = Bundle.main.bundleURL
        while app.pathExtension != "app", app.path != "/" { app.deleteLastPathComponent() }
        guard app.pathExtension == "app" else { throw DatabaseSourceBrokerFailure.unavailable }
        let resources = Bundle.main.resourceURL ?? Bundle.main.bundleURL.appendingPathComponent("Contents/Resources")
        #if DEBUG
        // This fixed app-bundle development pointer is never supplied by an XPC caller.
        let pointer = app.appendingPathComponent("Contents/Resources/ProjectRoot.txt")
        _ = pointer
        lock.lock(); let root = developmentRoot; lock.unlock()
        return try RuntimeLocation.resolve(resources: resources, developmentRoot: root, allowDevelopment: root != nil)
        #else
        return try RuntimeLocation.resolve(resources: resources)
        #endif
    }
    func databaseOutput(_ request:Data,credentials:Data,table:FileHandle?,directoryBookmark:Data?,reply:@escaping(Data?,String?)->Void) {
        guard request.count<=65_536,credentials.count<=16_384,
              var body=try? JSONDecoder().decode([String:JSONValue].self,from:request),
              (Set(body.keys)==["schemaVersion","action","options","operationID","tableSHA256","filename","folderSHA256","memoryLimit"] || Set(body.keys)==["schemaVersion","action","options","operationID","tableSHA256","filename","folderSHA256"]),body["schemaVersion"] == .integer(1),
              let action=body["action"]?.stringValue,["inspect","write"].contains(action),case .object(let rawOptions)=body["options"],
              let options=try? (rawOptions["kind"] == .string("api") ? APITableSettings.normalized(rawOptions):DatabaseOutputSettings.normalized(rawOptions,requireMapping:action=="write")),
              let operation=body["operationID"]?.stringValue,UUID(uuidString:operation)?.uuidString==operation,
              let secret=try? JSONDecoder().decode([String:JSONValue].self,from:credentials),Set(secret.keys)==["username","password"],secret.values.allSatisfy({guard case .string(let text)=$0 else{return false};return !text.contains("\0") && text.utf8.count<=4096}) else {reply(nil,"invalid_request");return}
        if action=="write" {
            guard let table,let hash=body["tableSHA256"]?.stringValue,hash.count==64,hash.utf8.allSatisfy({(48...57).contains($0)||(97...102).contains($0)}) else {reply(nil,"invalid_input");return}
            var info=stat();let flags=fcntl(table.fileDescriptor,F_GETFL)
            guard fstat(table.fileDescriptor,&info)==0,info.st_mode & mode_t(S_IFMT)==mode_t(S_IFREG),info.st_size>0,info.st_size<=50_000_000,info.st_uid==geteuid(),flags>=0,flags & O_ACCMODE==O_RDONLY,lseek(table.fileDescriptor,0,SEEK_SET)==0 else {reply(nil,"invalid_input");return}
        }
        lock.lock();guard !closed,budget.acquire() else {lock.unlock();reply(nil,"unavailable");return};active=true;lock.unlock()
        DispatchQueue.global(qos:.userInitiated).async { [self] in
            var result:Data?,failure:String?,scope:URL?
            var stage="scope"
            // Deliver confirmed outcome before cancellation-join acknowledgment.
            defer {scope?.stopAccessingSecurityScopedResource();reply(result,failure);finish()}
            do {
                body["sqlitePath"] = .null
                if options["backend"] == .string("sqlite") {
                    stage="scope_request"
                    guard let bookmark=directoryBookmark,bookmark.count<=64_000,let filename=body["filename"]?.stringValue,!filename.isEmpty,filename.utf8.count<=200,!filename.contains("/"),!filename.contains("\\"),![".",".."].contains(filename),!filename.contains("\0") else {throw DatabaseSourceBrokerFailure.invalidRequest}
                    stage="scope_resolution"
                    var stale=false
                    let url=try URL(resolvingBookmarkData:bookmark,options:[.withoutUI],relativeTo:nil,bookmarkDataIsStale:&stale)
                    scope=url
                    stage="scope_directory"
                    // Ephemeral cross-process bookmarks can report stale even
                    // when the physical folder is unchanged. Require the exact
                    // caller-pinned canonical identity, never follow a moved grant.
                    guard let pin=body["folderSHA256"]?.stringValue,pin.utf8.count==64,SHA256.hash(data:Data(url.standardizedFileURL.resolvingSymlinksInPath().path.utf8)).map {String(format:"%02x",$0)}.joined()==pin,try url.resourceValues(forKeys:[.isDirectoryKey]).isDirectory==true else {throw DatabaseSourceBrokerFailure.invalidRequest}
                    body["sqlitePath"] = .string(url.appendingPathComponent(filename).path)
                } else if directoryBookmark != nil {throw DatabaseSourceBrokerFailure.invalidRequest}
                if options["kind"] != .string("api") {body.removeValue(forKey:"memoryLimit")}
                body.removeValue(forKey:"filename");body.removeValue(forKey:"folderSHA256");body["credentials"] = .object(secret)
                let payload=try JSONEncoder().encode(body);guard payload.count<=65_536 else {throw DatabaseSourceBrokerFailure.invalidRequest}
                stage="runtime"
                let runtime=try runtime(),child=Process(),input=Pipe(),errors=Pipe()
                child.executableURL=runtime.executableURL;child.arguments=["-I","-B",try runtime.brokerEntry(options["kind"] == .string("api") ? .apiTable : .databaseOutput).path]
                child.currentDirectoryURL=runtime.workingDirectoryURL;child.environment=["PATH":"/usr/bin:/bin","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","POLARS_MAX_THREADS":"1"]
                child.standardInput=input;child.standardError=errors;child.standardOutput=table ?? FileHandle.nullDevice
                lock.lock();guard !closed else {lock.unlock();throw DatabaseSourceBrokerFailure.cancelled};process=child;lifetimeWrite=input.fileHandleForWriting
                stage="launch"
                do {try child.run()} catch {lock.unlock();throw error};lock.unlock()
                defer {if child.isRunning {close()};child.waitUntilExit();try? input.fileHandleForWriting.close();try? errors.fileHandleForReading.close()}
                var length=UInt32(payload.count).bigEndian;try input.fileHandleForWriting.write(contentsOf:withUnsafeBytes(of:&length){Data($0)}+payload)
                let seconds=Int(options["timeoutSeconds"]?.integerValue ?? 30)
                DispatchQueue.global().asyncAfter(deadline:.now()+Double(seconds+5)){if child.isRunning {child.terminate()};DispatchQueue.global().asyncAfter(deadline:.now()+2){if child.isRunning {Darwin.kill(child.processIdentifier,SIGKILL)}}}
                var receipt=Data()
                while receipt.count<=65_536 {let block=try errors.fileHandleForReading.read(upToCount:min(4096,65_537-receipt.count)) ?? Data();if block.isEmpty {break};receipt.append(block)}
                if receipt.count>65_536 {close()}
                stage="receipt"
                child.waitUntilExit();guard child.terminationStatus==0,receipt.count<=65_536,let r=try? JSONDecoder().decode([String:JSONValue].self,from:receipt),r["schemaVersion"] == .integer(1),r["kind"] == .string(options["kind"] == .string("api") ? "apiWrite":action=="inspect" ? "databaseMetadata":"databaseWrite") else {throw DatabaseSourceBrokerFailure.captureFailed}
                result=receipt
            } catch {Logger(subsystem:"info.kwip.KwipML.DatabaseSource",category:"output").error("Database operation failed at \(stage,privacy:.public), code \((error as NSError).code,privacy:.public)");close();failure="database_output_"+stage}
        }
    }
    func capture(_ request: Data, credentials: Data, output: FileHandle, reply: @escaping (Data?, String?) -> Void) {
        guard let (config, preview) = try? DatabaseSourceServiceIdentity.validate(request), credentials.count <= 16_384,
              case .object(let secret) = try? JSONDecoder().decode(JSONValue.self, from: credentials),
              Set(secret.keys) == ["username", "password"], secret.values.allSatisfy({ if case .string(let text) = $0 { return !text.contains("\0") && text.utf8.count <= 4096 }; return false }) else { reply(nil, "invalid_request"); return }
        var info = stat()
        let flags = fcntl(output.fileDescriptor, F_GETFL)
        guard fstat(output.fileDescriptor, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              info.st_nlink == 1, info.st_uid == geteuid(), info.st_size == 0, info.st_mode & 0o077 == 0,
              flags >= 0, flags & O_ACCMODE != O_RDONLY, lseek(output.fileDescriptor, 0, SEEK_CUR) == 0 else { reply(nil, "invalid_capture"); return }
        lock.lock()
        guard !closed, budget.acquire() else { lock.unlock(); reply(nil, "unavailable"); return }
        active = true
        lock.unlock()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            var receiptResult: Data?, errorResult: String?
            #if DEBUG
            var stage = "runtime"
            #endif
            defer { finish(); reply(receiptResult, errorResult) }
            do {
                let runtime = try runtime()
                #if DEBUG
                stage = "script"
                #endif
                let script = try runtime.brokerEntry(config["kind"] == .string("api") ? .apiTable : .databaseCapture)
                guard FileManager.default.fileExists(atPath: script.path) else { throw DatabaseSourceBrokerFailure.unavailable }
                var captureFrame:[String:JSONValue]=["schemaVersion":1,"options":.object(config),"credentials":.object(secret),"preview":.bool(preview)]
                if config["kind"] == .string("api"),let original=try? JSONDecoder().decode([String:JSONValue].self,from:request) {captureFrame["memoryLimit"]=original["memoryLimit"] ?? .null;captureFrame["operationSeconds"]=original["operationSeconds"] ?? .integer(300)}
                let payload=try JSONEncoder().encode(captureFrame)
                guard payload.count <= 65_536 else { throw DatabaseSourceBrokerFailure.invalidRequest }
                let child = Process(), input = Pipe(), errors = Pipe()
                child.executableURL = runtime.executableURL
                child.arguments = ["-I", "-B", script.path]
                child.currentDirectoryURL = runtime.workingDirectoryURL
                child.environment = ["PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                     "NUMEXPR_NUM_THREADS": "1", "POLARS_MAX_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"]
                child.standardInput = input; child.standardOutput = output; child.standardError = errors
                lock.lock()
                guard !closed else { lock.unlock(); throw DatabaseSourceBrokerFailure.cancelled }
                process = child; lifetimeWrite = input.fileHandleForWriting
                #if DEBUG
                stage = "launch"
                #endif
                do { try child.run() } catch { lock.unlock(); throw error }
                lock.unlock()
                #if DEBUG
                stage = "receipt"
                #endif
                defer {
                    if child.isRunning { close() }
                    if child.isRunning { child.waitUntilExit() }
                    try? input.fileHandleForWriting.close(); try? errors.fileHandleForReading.close()
                }
                var length = UInt32(payload.count).bigEndian
                let frame = withUnsafeBytes(of: &length) { Data($0) } + payload
                try input.fileHandleForWriting.write(contentsOf: frame)
                let seconds: Int
                if case .integer(let value) = config["timeoutSeconds"] { seconds = Int(value) } else { seconds = 30 }
                DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(seconds + 5)) {
                    if child.isRunning { child.terminate() }
                    DispatchQueue.global().asyncAfter(deadline: .now() + 2) { if child.isRunning { Darwin.kill(child.processIdentifier, SIGKILL) } }
                }
                var receipt = Data()
                while receipt.count <= 65_536 {
                    let block = try errors.fileHandleForReading.read(upToCount: min(4096, 65_537 - receipt.count)) ?? Data()
                    if block.isEmpty { break }
                    receipt.append(block)
                }
                if receipt.count > 65_536, child.isRunning { close() }
                child.waitUntilExit()
                guard child.terminationStatus == 0, receipt.count <= 65_536,
                      case .object(let value) = try JSONDecoder().decode(JSONValue.self, from: receipt),
                      value["schemaVersion"] == .integer(1), value["kind"] == .string("databaseCapture"), value["format"] == .string("ipc"),
                      case .integer(let bytes) = value["bytes"], (1...10_000_000).contains(bytes),
                      case .integer(let rows) = value["rows"], rows > 0 || config["kind"] == .string("api") && rows == 0,
                      case .integer(let columns) = value["columns"], (1...128).contains(columns), rows <= 5_000_000 / columns,
                      case .string(let hash) = value["sha256"], hash.utf8.count == 64,
                      value["sampleOnly"] == .bool(preview), !preview || rows <= 8 else { throw DatabaseSourceBrokerFailure.captureFailed }
                receiptResult = receipt
            } catch {
                #if DEBUG
                // Stage names are fixed constants; never emit driver errors,
                // endpoint settings, the payload, or credentials.
                NSLog("DBBroker.capture_failed_stage_%@", stage)
                #endif
                close(); errorResult = "capture_failed"
            }
        }
    }
}
private final class Listener: NSObject, NSXPCListenerDelegate {
    let budget = CaptureBudget()
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        guard let requirement = try? DatabaseSourceServiceIdentity.clientRequirement() else {
            NSLog("DBBroker.listener_requirement_unavailable")
            return false
        }
        connection.setCodeSigningRequirement(requirement)
        let service = DatabaseService(budget: budget)
        connection.exportedInterface = NSXPCInterface(with: KwipMLDatabaseSourceProtocol.self)
        connection.exportedObject = service
        connection.invalidationHandler = { service.close() }; connection.interruptionHandler = { service.close() }
        connection.resume(); return true
    }
}
private let delegate = Listener()
private let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
