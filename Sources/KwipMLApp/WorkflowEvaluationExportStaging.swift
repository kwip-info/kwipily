import Foundation
import Darwin
import KwipMLCore

protocol WorkflowEvaluationExportIO: Actor {
    func stage(reference: WorkflowEvaluationHistoryReference, root: URL, runtime: RuntimeLocation,
               expected: WorkflowEvaluationExportIdentity, includesTestRows: Bool, executionBudget: ResolvedExecutionBudget?) async throws -> WorkflowEvaluationExportPackage
    func publish(_ package: WorkflowEvaluationExportPackage, to destination: URL) async throws
}

actor WorkflowEvaluationExportWriter: WorkflowEvaluationExportIO {
    private struct CapturedExport: Encodable {
        let exportID: String
        let projectID: String
        let expected: WorkflowEvaluationExportIdentity
        let includesTestRows: Bool
        let runtimeEvidence: WorkflowRuntimeEvidence
        let executionBudget: ResolvedExecutionBudget?
    }
    private let launch: WorkflowWorkerLauncher
    private let accountingRoot: URL?
    private let storageLimitBytes: Int64?
    init(accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil, launch: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker) {
        self.accountingRoot = accountingRoot; self.storageLimitBytes = storageLimitBytes; self.launch = launch
    }
    func stage(reference: WorkflowEvaluationHistoryReference, root: URL, runtime: RuntimeLocation,
               expected: WorkflowEvaluationExportIdentity, includesTestRows: Bool, executionBudget: ResolvedExecutionBudget? = nil) async throws -> WorkflowEvaluationExportPackage {
        try Task.checkCancellation()
        let budgetWindow = try executionBudget.map { try ExecutionBudgetWindow(policy: $0, startedAtUptime: ProcessInfo.processInfo.systemUptime) }
        guard root.isFileURL, !root.path.contains("\0"), let path = realpath(root.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let workspace = URL(fileURLWithPath: String(cString: path), isDirectory: true)
        free(path)
        let source = workspace.appendingPathComponent("workflow-evaluations")
            .appendingPathComponent(reference.projectID).appendingPathComponent(reference.evaluationID.uuidString)
        let inspection = try WorkflowEvaluationOwnership.acquireInspection(directory: source,
            projectID: reference.projectID, evaluationID: reference.evaluationID.uuidString)
        defer { inspection.close() }
        let disclosed = try WorkflowEvaluationDisclosure.load(reference, root: workspace, exposureRoot: accountingRoot, inspection: inspection)
        let value = disclosed.value
        guard case .string(let state) = value.manifest.frozenLineage["stateDigest"],
              expected == WorkflowEvaluationExportIdentity(evaluationID: reference.evaluationID.uuidString,
                evaluationManifestSHA256: value.manifestSHA256, checkpointSHA256: value.manifest.parent.checkpointSHA256,
                stateDigest: state, snapshotReceiptSHA256: value.manifest.snapshotReceiptSHA256) else {
            throw WorkflowArchiveError("The saved evaluation changed. Reopen its results before exporting.")
        }
        let evidence = try WorkflowRuntimeEvidence.capture(runtime)
        let exportID = UUID().uuidString
        let capture = CapturedExport(exportID: exportID, projectID: reference.projectID, expected: expected,
            includesTestRows: includesTestRows, runtimeEvidence: evidence, executionBudget: executionBudget)
        let request = WorkflowEvaluationExportRequest(schemaVersion: executionBudget == nil ? 1 : 2, exportID: exportID, projectID: reference.projectID,
            evaluationID: reference.evaluationID.uuidString, requestIdentity: WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(capture)),
            evaluationDirectory: source.path, evaluationManifestSHA256: value.manifestSHA256, includeTestRows: includesTestRows, executionBudget: executionBudget)
        let storage = try storageLimitBytes.map { limit in
            try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot).reserve(
                scopes: [.init(namespace: .workflowEvaluationExports, components: [reference.projectID, exportID])],
                peakBytes: 623_000_000, limitBytes: limit)
        }
        var storageTransferred = false
        defer { if !storageTransferred { storage?.close() } }
        let owner = try WorkflowEvaluationExportOwnership.create(workspace: workspace, request: request)
        defer { owner.close() }
        do {
            if let storage {
                let metadata = owner.metadata
                try storage.bind(to: .init(scope: .init(namespace: .workflowEvaluationExports,
                    components: [metadata.projectID, metadata.exportID]),
                    directoryIdentity: metadata.directoryIdentity, ownerLockIdentity: metadata.ownerLockIdentity,
                    activityLockIdentity: metadata.activityLockIdentity))
            }
            try owner.publishRequest(request)
            try Task.checkCancellation()
            guard try evidence.matchesCurrent(runtime) else { throw WorkflowArchiveError("Worker runtime changed before export. Retry with the current runtime.") }
            try inspection.check()
            var budgetArguments: [String] = []
            if let budgetWindow {
                let remaining = try budgetWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                budgetArguments = ["--kwip-launch-budget-sha256", try owner.publishLaunchBudget(remainingSeconds: remaining)]
            }
            let ownershipSHA256 = try owner.prepareLaunch(), requestSHA256 = try request.requestSHA256()
            let process = WorkflowEvaluationExportProcess(exportID: request.exportID, requestIdentity: request.requestIdentity,
                requestSHA256: requestSHA256, ownershipSHA256: ownershipSHA256, includesTestRows: includesTestRows, executionBudget: executionBudget)
            let completion = try await process.run(executable: runtime.executableURL,
                arguments: runtime.argumentPrefix + ["--kwip-export-supervise", owner.directory.path, requestSHA256, ownershipSHA256] + budgetArguments,
                directory: runtime.workingDirectoryURL, launch: launch)
            try Task.checkCancellation()
            try inspection.check(); try owner.check()
            var package = try WorkflowEvaluationExportPackage.verify(directory: owner.directory.appendingPathComponent("package"),
                expected: expected, includesTestRows: includesTestRows, manifestSHA256: completion.manifestSHA256,
                cleanupDirectory: owner.directory)
            guard package.fileCount == completion.fileCount, package.totalBytes == completion.totalBytes else {
                throw WorkflowArchiveError("Export completion counts differ from its verified package.")
            }
            try Task.checkCancellation()
            try owner.check(); try inspection.check()
            package.storageLifetime = try WorkflowExportStorageLifetime(directory: owner.directory, reservation: storage)
            storageTransferred = true
            return package
        } catch {
            // run() does not return on cancellation until the actual child exit callback.
            // Only this still-pinned export directory can be removed; never the source.
            let removable = (try? owner.check()) != nil
            owner.close()
            if removable, let cleanup = try? WorkflowExportStorageLifetime(directory: owner.directory, reservation: storage) {
                cleanup.remove()
            }
            throw error
        }
    }
    func publish(_ package: WorkflowEvaluationExportPackage, to destination: URL) async throws {
        try package.publish(to: destination)
    }
}

/// Used under the process relay's lock. Completion is provisional until actual exit.
final class WorkflowEvaluationExportTranscript {
    struct Completion: Decodable {
        let manifestSHA256: String
        let fileCount: Int
        let totalBytes: Int
        let includesTestRows: Bool
    }
    private let exportID: String, requestIdentity: String, requestSHA256: String, ownershipSHA256: String
    private let includesTestRows: Bool
    private let executionBudget: ResolvedExecutionBudget?
    private var ready = false, started = false
    private var completion: Completion?
    init(exportID: String, requestIdentity: String, requestSHA256: String, ownershipSHA256: String, includesTestRows: Bool, executionBudget: ResolvedExecutionBudget? = nil) {
        self.exportID = exportID; self.requestIdentity = requestIdentity; self.requestSHA256 = requestSHA256
        self.ownershipSHA256 = ownershipSHA256; self.includesTestRows = includesTestRows; self.executionBudget = executionBudget
    }
    func consume(_ line: String) throws {
        guard line.utf8.count <= 1_000_000, let bytes = line.data(using: .utf8),
              let event = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              let name = event["event"] as? String, completion == nil else { throw failure("Malformed or late export event.") }
        func keys(_ expected: Set<String>) throws {
            guard Set(event.keys) == expected else { throw failure("Unexpected export event fields.") }
        }
        if name == "error" {
            throw failure(String((event["message"] as? String ?? "Export worker failed.").prefix(4000)))
        }
        guard event["exportID"] as? String == exportID else { throw failure("Export event belongs to another request.") }
        switch name {
        case "exportReady":
            try keys(["event", "exportID", "requestSHA256", "ownershipSHA256"])
            guard !ready, event["requestSHA256"] as? String == requestSHA256,
                  event["ownershipSHA256"] as? String == ownershipSHA256 else { throw failure("Export authority acknowledgement differs from its launch.") }
            ready = true
        case "exportStarted":
            try keys(["event", "exportID", "requestIdentity"])
            guard ready, !started, event["requestIdentity"] as? String == requestIdentity else { throw failure("Export started out of order or with another identity.") }
            started = true
        case "exportCompleted":
            try keys(["event", "exportID", "requestIdentity", "manifestSHA256", "fileCount", "totalBytes", "includesTestRows"])
            let value = try JSONDecoder().decode(Completion.self, from: bytes)
            let minimumCount = WorkflowEvaluationExportPackage.requiredLimits.count + 1 + (includesTestRows ? WorkflowEvaluationExportPackage.optionalLimits.count : 0)
            let maximumCount = minimumCount + WorkflowEvaluationExportPackage.dependencyLimits.count
            // Provisional count admission only: stage() subsequently verifies the exact
            // reviewed names, inventory, bytes and hashes before returning a package.
            guard ready, started, event["requestIdentity"] as? String == requestIdentity,
                  WorkflowArchiveCodec.isHash(value.manifestSHA256), value.includesTestRows == includesTestRows,
                  (minimumCount...maximumCount).contains(value.fileCount), (1...282_000_000).contains(value.totalBytes) else { throw failure("Export completion differs from the requested package.") }
            completion = value
        default: throw failure("Unexpected export worker event.")
        }
    }
    func finish(exitCode: Int32, stderr: String) throws -> Completion {
        if exitCode == 124, let executionBudget { throw ExecutionBudgetError.wallTimeLimitExceeded(seconds: executionBudget.maximumRunSeconds) }
        guard exitCode == 0 else { throw failure("Export worker exited with code \(exitCode). " + String(stderr.prefix(4000))) }
        guard let completion else { throw failure("Export worker exited without a verified completion sequence.") }
        return completion
    }
    private func failure(_ message: String) -> WorkflowArchiveError { WorkflowArchiveError(message) }
}

/// Callback order and handle assignment are synchronized even for synchronous test launchers.
/// A cancellation request never resumes the waiter before actual process exit.
final class WorkflowEvaluationExportProcess: @unchecked Sendable {
    private let lock = NSLock()
    private let transcript: WorkflowEvaluationExportTranscript
    private var continuation: CheckedContinuation<WorkflowEvaluationExportTranscript.Completion, Error>?
    private var handle: WorkflowWorkerHandle?
    private var cancelled = false, finished = false, exited = false, cancellationDelivered = false
    private var failure: Error?
    init(exportID: String, requestIdentity: String, requestSHA256: String, ownershipSHA256: String, includesTestRows: Bool, executionBudget: ResolvedExecutionBudget? = nil) {
        transcript = .init(exportID: exportID, requestIdentity: requestIdentity, requestSHA256: requestSHA256,
            ownershipSHA256: ownershipSHA256, includesTestRows: includesTestRows, executionBudget: executionBudget)
    }
    var hasExited: Bool { lock.lock(); defer { lock.unlock() }; return exited }
    func run(executable: URL, arguments: [String], directory: URL, launch: @escaping WorkflowWorkerLauncher) async throws -> WorkflowEvaluationExportTranscript.Completion {
        try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                start(continuation, executable: executable, arguments: arguments, directory: directory, launch: launch)
            }
        }, onCancel: { self.cancel() })
    }
    private func start(_ waiter: CheckedContinuation<WorkflowEvaluationExportTranscript.Completion, Error>,
                       executable: URL, arguments: [String], directory: URL, launch: WorkflowWorkerLauncher) {
        lock.lock()
        guard !finished, continuation == nil else { lock.unlock(); waiter.resume(throwing: WorkflowArchiveError("Export process already started.")); return }
        if cancelled { finished = true; lock.unlock(); waiter.resume(throwing: CancellationError()); return }
        continuation = waiter
        lock.unlock()
        do {
            let value = try launch(executable, arguments, directory, { self.receive($0) }, { self.exit($0, $1) })
            lock.lock()
            if !finished { handle = value }
            let cancel = pendingCancellation()
            lock.unlock()
            cancel?()
        } catch { end(.failure(error), actualExit: false) }
    }
    func cancel() {
        lock.lock(); cancelled = true
        let action = pendingCancellation()
        lock.unlock(); action?()
    }
    private func pendingCancellation() -> (() -> Void)? {
        guard !finished, !cancellationDelivered, cancelled || failure != nil, let handle else { return nil }
        cancellationDelivered = true
        return handle.cancel
    }
    private func receive(_ line: String) {
        lock.lock()
        if !finished, failure == nil {
            do { try transcript.consume(line) } catch { failure = error }
        }
        let action = pendingCancellation()
        lock.unlock(); action?()
    }
    private func exit(_ code: Int32, _ stderr: String) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        let result: Result<WorkflowEvaluationExportTranscript.Completion, Error>
        if cancelled { result = .failure(CancellationError()) }
        else if code == 125 { result = Result { try transcript.finish(exitCode: code, stderr: stderr) } }
        else if let failure { result = .failure(failure) }
        else { result = Result { try transcript.finish(exitCode: code, stderr: stderr) } }
        finished = true; exited = true
        let waiter = continuation; continuation = nil; handle = nil
        lock.unlock(); waiter?.resume(with: result)
    }
    private func end(_ result: Result<WorkflowEvaluationExportTranscript.Completion, Error>, actualExit: Bool) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true; exited = actualExit
        let waiter = continuation; continuation = nil; handle = nil
        lock.unlock(); waiter?.resume(with: result)
    }
}
