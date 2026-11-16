import Foundation
import Darwin
import KwipMLCore

/// Typed tools share operation ownership without borrowing experiment or run identities.
/// Callers retain their own workspace/lesson reservation until the exit callback.
@MainActor final class WorkflowToolExecution {
    private let owner: WorkflowOperationOwner
    private let request: WorkflowToolRequest
    private let storageReservation: WorkflowStorageAdmission.Reservation?
    private let onLine: (String) -> Void
    private let onExit: (Int32, String) -> Void
    private var worker: WorkflowWorkerHandle?
    private var ready = false, cancelled = false, finished = false, terminalReceived = false, cancellationDelivered = false
    private let finalize: ((Int32, String) async throws -> Void)?
    private let retainInputsOnSuccess:Bool
    private let retainModelRecoveryInputs:Bool
    private var protocolError: String?

    private init(owner: WorkflowOperationOwner, request: WorkflowToolRequest,
                 storageReservation: WorkflowStorageAdmission.Reservation? = nil,
                 finalize: ((Int32, String) async throws -> Void)? = nil, retainInputsOnSuccess:Bool = false,retainModelRecoveryInputs:Bool = false,
                 onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String) -> Void) {
        self.storageReservation = storageReservation; self.retainInputsOnSuccess=retainInputsOnSuccess;self.retainModelRecoveryInputs=retainModelRecoveryInputs
        self.owner = owner; self.request = request; self.onLine = onLine; self.onExit = onExit; self.finalize = finalize
    }
    static func launch(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, toolMode: WorkflowToolRequest.Mode,
                       inputs: [String: Data], maximumSeconds: Double, storageLimitBytes: Int64? = nil,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle {
        try launchAdmitted(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID, runtime: runtime, window: window,
            toolMode: toolMode, inputs: inputs, maximumSeconds: maximumSeconds, storageLimitBytes: storageLimitBytes,
            operationID: UUID().uuidString, admittedStorage: nil, launcher: launcher, onLine: onLine, onExit: onExit)
    }
    static func launchAsync(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, toolMode: WorkflowToolRequest.Mode,
                       inputs: [String: Data], maximumSeconds: Double, storageLimitBytes: Int64? = nil,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String) -> Void) async throws -> WorkflowWorkerHandle {
        try Task.checkCancellation()
        let operationID = UUID().uuidString
        var reservation: WorkflowStorageAdmission.Reservation?
        if let storageLimitBytes {
            guard inputs.count <= (toolMode == .learningAssessment ? 3 : 2), inputs.values.allSatisfy({ $0.count <= 2_000_000 }) else { throw WorkflowFileLeaseError.unsafePath }
            let peak = inputs.values.reduce(Int64(1_000_000 + (toolMode == .learningAssessment ? 65_536 : 0))) { $0 + Int64($1.count) }
            let admission = Task.detached(priority: .userInitiated) {
                try Task.checkCancellation()
                return try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot).reserve(
                    scopes: [.init(namespace: .workflowOperations, components: [projectID, operationID])],
                    peakBytes: peak, limitBytes: storageLimitBytes)
            }
            reservation = try await withTaskCancellationHandler(operation: { try await admission.value }, onCancel: { admission.cancel() })
        }
        do {
            try Task.checkCancellation()
            return try launchAdmitted(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID, runtime: runtime, window: window,
                toolMode: toolMode, inputs: inputs, maximumSeconds: maximumSeconds, storageLimitBytes: nil,
                operationID: operationID, admittedStorage: reservation, launcher: launcher, onLine: onLine, onExit: onExit)
        } catch { reservation?.close(); throw error }
    }
    private static func launchAdmitted(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, toolMode: WorkflowToolRequest.Mode,
                       inputs: [String: Data], maximumSeconds: Double, storageLimitBytes: Int64? = nil,
                       operationID: String, admittedStorage: WorkflowStorageAdmission.Reservation?,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle {
        guard UUID(uuidString: projectID)?.uuidString == projectID, maximumSeconds.isFinite, maximumSeconds > 0 else {
            throw WorkflowFileLeaseError.unsafePath
        }
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        guard let canonical = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let root = URL(fileURLWithPath: String(cString: canonical)); free(canonical)
        let directory = root.appendingPathComponent("workflow-operations/\(projectID)/\(operationID)")
        let files = inputs.sorted { $0.key < $1.key }.map { WorkflowArchiveFile(path: $0.key, bytes: $0.value.count, sha256: WorkflowArchiveCodec.hash($0.value)) }
        var arguments: [String]
        switch toolMode {
        case .selfPlay, .gameSession, .modelAsset, .textGeneration, .programExport, .backendFit, .modelPreview, .modelTraining, .dataPreparation, .sourceInspection, .preparationInspection, .tableViewQuery: throw WorkflowArchiveError("This operation requires streamed input capture before tool launch.")
        case .mixtureValidation: arguments = ["--validate-mixture",directory.appendingPathComponent("input/mixture.json").path]
        case .runtimeCapabilities: arguments = ["--capabilities"]
        case .learningRuntime: arguments = ["--learning-runtime"]
        case .codeExport:
            let workflow = Set(inputs.keys) == ["input/workflow-code.json"]
            arguments = workflow
                ? ["--export-workflow-code", "--workflow-code-plan", directory.appendingPathComponent("input/workflow-code.json").path]
                : ["--export-code", "--graph", directory.appendingPathComponent("input/graph.json").path]
        case .learningAssessment:
            guard let submission = inputs["input/submission.json"] else { throw WorkflowFileLeaseError.unsafePath }
            arguments = ["--learning-evidence-request", directory.appendingPathComponent("input/assessment.json").path,
                         "--learning-submission", directory.appendingPathComponent("input/submission.json").path,
                         "--learning-submission-sha256", WorkflowArchiveCodec.hash(submission)]
            if inputs["input/model-registry.json"] != nil {
                arguments += ["--registry", directory.appendingPathComponent("input/model-registry.json").path]
            }
        }
        let runtimeEvidence = try WorkflowRuntimeEvidence.capture(runtime)
        struct Identity: Encodable {
            let operationID: String
            let projectID: String
            let toolMode: WorkflowToolRequest.Mode
            let files: [WorkflowArchiveFile]
            let budget: ResolvedExecutionBudget
            let runtime: WorkflowRuntimeEvidence
        }
        let identity = try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(Identity(operationID: operationID,
            projectID: projectID, toolMode: toolMode, files: files, budget: window.policy, runtime: runtimeEvidence)))
        let request = WorkflowToolRequest(operationID: operationID, projectID: projectID, requestIdentity: identity,
            toolMode: toolMode, arguments: arguments, inputFiles: files, executionBudget: window.policy,
            remainingSeconds: min(maximumSeconds, try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)))
        try request.validate(directory: directory, verifyInputs: false)
        let storageReservation: WorkflowStorageAdmission.Reservation?
        if let admittedStorage { storageReservation = admittedStorage }
        else if let storageLimitBytes {
            // Small tools return their result on the bounded event stream. Reserve exact
            // input bytes plus the bounded request/ownership/claim/cancellation metadata.
            // Standardization assessments serialize one CSV fixture at a time under
            // this operation's input directory; runner.learning_evidence enforces 64 KiB.
            let fixtureBytes: Int64 = toolMode == .learningAssessment ? 65_536 : 0
            let peakBytes = files.reduce(Int64(1_000_000) + fixtureBytes) { $0 + Int64($1.bytes) }
            storageReservation = try WorkflowStorageAdmission(workspace: root, accountingRoot: accountingRoot).reserve(
                scopes: [.init(namespace: .workflowOperations, components: [projectID, operationID])],
                peakBytes: peakBytes, limitBytes: storageLimitBytes)
        } else { storageReservation = nil }
        var transferred = false
        defer { if !transferred { storageReservation?.close() } }
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        do {
            if let storageReservation {
                try storageReservation.bind(to: .init(
                    scope: .init(namespace: .workflowOperations, components: [projectID, operationID]),
                    directoryIdentity: owner.metadata.directoryIdentity,
                    ownerLockIdentity: owner.metadata.ownerLockIdentity,
                    activityLockIdentity: owner.metadata.activityLockIdentity))
            }
            if !inputs.isEmpty {
                try DiagnosticFileIO.createExclusiveDirectory(root: owner.directory, name: "input")
                for file in files {
                    try DiagnosticFileIO.writeImmutable(try inputs[file.path].unwrapToolInput(), root: owner.directory,
                        parts: WorkflowArchiveCodec.pathParts(file.path), limit: 2_000_000)
                }
            }
            try owner.publishRequest(request)
            guard try runtimeEvidence.matchesCurrent(runtime) else { throw WorkflowArchiveError("Worker runtime changed before tool launch.") }
            _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
            let pin = try owner.prepareLaunch()
            let handle = try launchPrepared(owner: owner, request: request, runtime: runtime, ownershipSHA256: pin,
                storageReservation: storageReservation, launcher: launcher, onLine: onLine, onExit: onExit)
            transferred = true
            return handle
        } catch {
            if (try? owner.check()) != nil { try? FileManager.default.removeItem(at: owner.directory.appendingPathComponent("input")) }
            owner.close(); throw error
        }
    }
    /// A prepared owner remains held while an optional asynchronous finalizer verifies output.
    /// Small tools omit the hook and preserve their synchronous terminal callback ordering.
    static func launchPrepared(owner: WorkflowOperationOwner, request: WorkflowToolRequest, runtime: RuntimeLocation,
                               ownershipSHA256: String, storageReservation: WorkflowStorageAdmission.Reservation? = nil,
                               launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                               finalize: ((Int32, String) async throws -> Void)? = nil, retainInputsOnSuccess:Bool = false,retainModelRecoveryInputs:Bool = false,
                               onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle {
        do {
            try owner.check()
            guard !retainInputsOnSuccess || [.selfPlay,.gameSession,.dataPreparation,.textGeneration,.modelAsset].contains(request.toolMode) else {throw WorkflowFileLeaseError.unsafePath}
            guard ownershipSHA256 == owner.ownershipSHA256, try request.requestSHA256() == owner.metadata.requestSHA256 else { throw WorkflowFileLeaseError.unsafePath }
            let relay = WorkflowToolExecution(owner: owner, request: request, storageReservation: storageReservation, finalize: finalize, retainInputsOnSuccess:retainInputsOnSuccess,retainModelRecoveryInputs:retainModelRecoveryInputs, onLine: onLine, onExit: onExit)
            let worker = try launcher(runtime.executableURL,
                runtime.argumentPrefix + ["--kwip-operation-supervise", owner.directory.path, owner.metadata.requestSHA256, ownershipSHA256],
                runtime.workingDirectoryURL, { relay.consume($0) }, { relay.exit($0, $1) })
            if !relay.terminalReceived { relay.worker = worker; relay.deliverCancellation() }
            return .init(cancel: { guard !relay.finished else { return }; relay.cancelled = true; relay.deliverCancellation() })
        } catch {
            if (try? owner.check()) != nil { try? FileManager.default.removeItem(at: owner.directory.appendingPathComponent("input")) }
            owner.close(); storageReservation?.close(); throw error
        }
    }
    private func consume(_ line: String) {
        guard !finished, !terminalReceived, !cancelled, protocolError == nil else { return }
        guard let data = line.data(using: .utf8), data.count <= 1_000_000,
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let name = event["event"] as? String else {
            reject("Malformed tool event."); return
        }
        if name == "operationReady" {
            guard !ready, Set(event.keys) == ["event", "operationID", "toolMode", "requestSHA256", "ownershipSHA256"],
                  event["operationID"] as? String == request.operationID, event["toolMode"] as? String == request.toolMode.rawValue,
                  event["requestSHA256"] as? String == owner.metadata.requestSHA256,
                  event["ownershipSHA256"] as? String == owner.ownershipSHA256 else { reject("Tool readiness differs from captured authority."); return }
            ready = true; return
        }
        guard ready || name == "error" else { reject("Tool output preceded its ready acknowledgement."); return }
        onLine(line)
    }
    private func reject(_ message: String) { protocolError = message; deliverCancellation() }
    private func deliverCancellation() {
        guard !finished, !cancellationDelivered, cancelled || protocolError != nil, let worker else { return }
        cancellationDelivered = true; worker.cancel()
    }
    private func exit(_ code: Int32, _ stderr: String) {
        guard !terminalReceived, !finished else { return }
        terminalReceived = true; worker = nil
        let message = protocolError ?? (!ready && code == 0 ? "Tool exited without readiness." : stderr)
        let result: Int32 = cancelled ? 130 : code == 124 ? 124 : protocolError != nil || (!ready && code == 0) ? 1 : code
        if let finalize {
            Task { [self] in
                do { try await finalize(result, message); complete(result, message) }
                catch { complete(result == 0 ? 1 : result, error.localizedDescription) }
            }
        } else { complete(result, message) }
    }
    private func complete(_ code: Int32, _ message: String) {
        guard !finished else { return }
        finished = true
        if request.toolMode != .selfPlay && !(retainInputsOnSuccess && code == 0 && !cancelled) && !(retainModelRecoveryInputs && request.toolMode == .modelAsset), (try? owner.check()) != nil { try? FileManager.default.removeItem(at: owner.directory.appendingPathComponent("input")) }
        owner.close()
        storageReservation?.close()
        onExit(cancelled ? 130 : code, message)
    }

}

private extension Optional where Wrapped == Data {
    func unwrapToolInput() throws -> Data {
        guard let self else { throw WorkflowFileLeaseError.unsafePath }; return self
    }
}
