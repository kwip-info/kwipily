import Foundation
import Darwin
import KwipMLCore

/// Shared owned lifecycle for bounded multi-file tools; each mode keeps its own result verifier.
@MainActor enum WorkflowCapturedToolExecution {
    enum Mode {
        case selfPlay, gameSession, backendFit, preview, preparation, sourceInspection, preparationInspection, modelTraining, tableViewQuery
        var toolMode: WorkflowToolRequest.Mode {
            switch self { case .selfPlay:return .selfPlay; case .gameSession:return .gameSession; case .tableViewQuery: return .tableViewQuery; case .backendFit: return .backendFit; case .preview: return .modelPreview; case .preparation: return .dataPreparation; case .sourceInspection: return .sourceInspection; case .preparationInspection: return .preparationInspection; case .modelTraining: return .modelTraining }
        }
        var purpose: WorkflowPreviewCapture.Purpose {
            switch self { case .selfPlay:return .selfPlay; case .gameSession:return .gameSession; case .tableViewQuery: return .tableViewQuery; case .backendFit: return .backendFit; case .preview: return .preview; case .preparation: return .preparation; case .sourceInspection: return .sourceInspection; case .preparationInspection: return .preparationInspection; case .modelTraining: return .modelTraining }
        }
        var completionEvent: String {
            switch self { case .selfPlay:return "selfPlayCompleted"; case .gameSession:return "gameSessionCompleted"; case .tableViewQuery: return "tableViewPage"; case .backendFit: return "backendCompleted"; case .preview: return "previewCompleted"; case .preparation: return "dataflowCompleted"; case .sourceInspection: return "dataPreview"; case .preparationInspection: return "dataflowInspectionCompleted"; case .modelTraining: return "completed" }
        }
        var title: String {
            switch self { case .selfPlay:return "Self-play"; case .gameSession:return "Game session"; case .tableViewQuery: return "Table View"; case .backendFit: return "Backend fit"; case .preview: return "Preview"; case .preparation: return "Preparation"; case .sourceInspection: return "Source inspection"; case .preparationInspection: return "Saved-output inspection"; case .modelTraining: return "Model training" }
        }
    }
    static func launch(mode: Mode, workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, operationID: String = UUID().uuidString, storageLimitBytes: Int64? = nil, storageAdmission: WorkflowCaptureStorageReservation? = nil, inputs: @escaping (URL) throws -> [WorkflowPreviewCapture.Input],
                       arguments: (URL) -> [String], retainInputsOnSuccess:Bool = false,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void,
                       onExit: @escaping (Int32, String, URL?) -> Void) async throws -> WorkflowWorkerHandle {
        guard UUID(uuidString: projectID)?.uuidString == projectID, UUID(uuidString: operationID)?.uuidString == operationID else { throw WorkflowFileLeaseError.unsafePath }
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        guard let pointer = realpath(workspace.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let root = URL(fileURLWithPath: String(cString: pointer)); free(pointer)
        let directory = root.appendingPathComponent("workflow-operations/\(projectID)/\(operationID)")
        let capturedArguments = arguments(directory)
        let preparation = Task.detached(priority: .userInitiated) {
            try prepare(mode: mode, workspace: root, accountingRoot: accountingRoot, projectID: projectID, operationID: operationID, runtime: runtime,
                        window: window, storageLimitBytes: storageLimitBytes, storageAdmission: storageAdmission, inputs: inputs, arguments: capturedArguments)
        }
        let prepared = try await withTaskCancellationHandler(operation: { try await preparation.value }, onCancel: { preparation.cancel() })
        do {
            try Task.checkCancellation()
            _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
            let state = State(mode: mode, prepared: prepared, window: window, onLine: onLine, onExit: onExit)
            let worker = try WorkflowToolExecution.launchPrepared(owner: prepared.owner, request: prepared.request,
                runtime: runtime, ownershipSHA256: prepared.ownershipSHA256, storageReservation: prepared.storageReservation, launcher: launcher,
                finalize: { code, message in try await state.finalize(code: code, message: message) },retainInputsOnSuccess:retainInputsOnSuccess,
                onLine: { state.consume($0) }, onExit: { state.finish(code: $0, message: $1) })
            state.attach(worker)
            return .init(cancel: { state.cancel() })
        } catch {
            cleanup(prepared.owner)
            prepared.storageReservation?.close()
            throw error
        }
    }
    private struct Prepared {
        let owner: WorkflowOperationOwner
        let request: WorkflowToolRequest
        let ownershipSHA256: String
        let storageReservation: WorkflowStorageAdmission.Reservation?
    }
    private nonisolated static func prepare(mode: Mode, workspace: URL, accountingRoot: URL? = nil, projectID: String, operationID: String, runtime: RuntimeLocation,
                                            window: ExecutionBudgetWindow, storageLimitBytes: Int64?, storageAdmission: WorkflowCaptureStorageReservation?, inputs: @escaping (URL) throws -> [WorkflowPreviewCapture.Input], arguments: [String]) throws -> Prepared {
        func check() throws {
            try Task.checkCancellation()
            _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        }
        try check()
        let evidence = try WorkflowRuntimeEvidence.capture(runtime)
        let directory = workspace.appendingPathComponent("workflow-operations/\(projectID)/\(operationID)")
        guard storageLimitBytes == nil || storageAdmission == nil else {
            throw WorkflowArchiveError("Provide one storage admission for the captured operation.")
        }
        if storageLimitBytes != nil {
            guard mode == .selfPlay || mode == .gameSession || mode == .preview || mode == .modelTraining || mode == .backendFit else {
                throw WorkflowArchiveError("This tool's earlier source staging must participate in storage admission before enabling its limit.")
            }
        }
        var storageReservation: WorkflowStorageAdmission.Reservation?
        var transferred = false
        defer { if !transferred { storageReservation?.close() } }
        if let storageAdmission {
            storageReservation = try storageAdmission.take(workspace: workspace, projectID: projectID,
                operationID: operationID, purpose: mode.purpose, workload: window.policy.resources?.workloadPolicy)
        }
        let capturedInputs = try inputs(directory)
        try check()
        let stageID = storageAdmission?.stageID ?? UUID()
        let captureBytes: Int?
        if let storageAdmission { captureBytes = storageAdmission.maximumCaptureBytes }
        else if let storageLimitBytes {
            let byteCount = try WorkflowPreviewCapture.byteCountBound(inputs: capturedInputs, purpose: mode.purpose, workload: window.policy.resources?.workloadPolicy)
            captureBytes = byteCount
            let admission = try WorkflowCaptureStorageReservation(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
                operationID: operationID, purpose: mode.purpose, stageID: stageID,
                maximumCaptureBytes: byteCount, limitBytes: storageLimitBytes, workload: window.policy.resources?.workloadPolicy)
            storageReservation = try admission.take(workspace: workspace, projectID: projectID,
                operationID: operationID, purpose: mode.purpose, workload: window.policy.resources?.workloadPolicy)
        } else { captureBytes = nil }
        try check()
        let capture = try WorkflowPreviewCapture.stage(workspace: workspace, inputs: capturedInputs, purpose: mode.purpose, workload: window.policy.resources?.workloadPolicy,
            stageID: stageID, maximumBytes: captureBytes, checkCancellation: check)
        defer { capture.remove() }
        try check()
        if mode == .modelTraining, let capturedFile = capture.files.first(where: { $0.path == "input/model-registry.json" }) {
            let captured = try DiagnosticFileIO.read(root: capture.directory, parts: ["input","model-registry.json"], limit: 2_000_000)
            let actual = try DiagnosticFileIO.read(root: runtime.registryURL.deletingLastPathComponent(), parts: [runtime.registryURL.lastPathComponent], limit: 2_000_000)
            guard captured.count == capturedFile.bytes, WorkflowArchiveCodec.hash(captured) == capturedFile.sha256,
                  WorkflowArchiveCodec.hash(actual) == evidence.workerFiles["components.json"],
                  try WorkflowArchiveCodec.registryHash(JSONDecoder().decode(ComponentRegistry.self, from: captured)) == WorkflowArchiveCodec.registryHash(JSONDecoder().decode(ComponentRegistry.self, from: actual)) else {
                throw WorkflowArchiveError("Captured source3 registry differs from the selected worker's pinned contracts.")
            }
        }
        struct Identity: Encodable {
            let operationID: String; let projectID: String; let toolMode: WorkflowToolRequest.Mode; let arguments: [String]
            let files: [WorkflowArchiveFile]; let budget: ResolvedExecutionBudget; let runtime: WorkflowRuntimeEvidence
        }
        let identity = try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(Identity(operationID: operationID,
            projectID: projectID, toolMode: mode.toolMode, arguments: arguments, files: capture.files, budget: window.policy, runtime: evidence)))
        let request = WorkflowToolRequest(operationID: operationID, projectID: projectID, requestIdentity: identity,
            toolMode: mode.toolMode, arguments: arguments, inputFiles: capture.files, executionBudget: window.policy,
            remainingSeconds: try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime))
        let owner = try WorkflowOperationOwnership.create(workspace: workspace, request: request)
        do {
            try owner.check()
            if let storageReservation {
                try storageReservation.bind(to: .init(
                    scope: .init(namespace: .workflowOperations, components: [projectID, operationID]),
                    directoryIdentity: owner.metadata.directoryIdentity,
                    ownerLockIdentity: owner.metadata.ownerLockIdentity,
                    activityLockIdentity: owner.metadata.activityLockIdentity))
            }
            try FileManager.default.moveItem(at: capture.directory.appendingPathComponent("input"), to: owner.directory.appendingPathComponent("input"))
            try owner.publishRequest(request)
            try check()
            guard try evidence.matchesCurrent(runtime) else { throw WorkflowArchiveError("Worker runtime changed before captured tool launch.") }
            let pin = try owner.prepareLaunch()
            try check()
            transferred = true
            return Prepared(owner: owner, request: request, ownershipSHA256: pin, storageReservation: storageReservation)
        } catch { cleanup(owner); throw error }
    }
    private nonisolated static func cleanup(_ owner: WorkflowOperationOwner) {
        if (try? owner.check()) != nil { try? FileManager.default.removeItem(at: owner.directory.appendingPathComponent("input")) }
        owner.close()
    }
    @MainActor private final class State {
        let mode: Mode
        let prepared: Prepared
        let window: ExecutionBudgetWindow
        let onLine: (String) -> Void
        let onExit: (Int32, String, URL?) -> Void
        var worker: WorkflowWorkerHandle?
        var timer: DispatchWorkItem?
        var finalization: Task<URL, Error>?
        var completion: Data?
        var completionError: String?
        var output: URL?
        var userCancelled = false, timedOut = false, finished = false
        init(mode: Mode, prepared: Prepared, window: ExecutionBudgetWindow,
             onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String, URL?) -> Void) {
            self.mode = mode; self.prepared = prepared; self.window = window; self.onLine = onLine; self.onExit = onExit
        }
        func attach(_ handle: WorkflowWorkerHandle) {
            guard !finished else { return }
            worker = handle
            do {
                let remaining = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                let timer = DispatchWorkItem { [weak self] in
                    guard let self, !self.finished, !self.userCancelled else { return }
                    self.timedOut = true; self.worker?.cancel(); self.finalization?.cancel()
                }
                self.timer = timer
                DispatchQueue.main.asyncAfter(deadline: .now() + remaining, execute: timer)
            } catch { timedOut = true; worker?.cancel(); finalization?.cancel() }
        }
        func cancel() {
            guard !finished, !userCancelled else { return }
            userCancelled = true; worker?.cancel(); finalization?.cancel()
        }
        func consume(_ line: String) {
            guard !finished, !userCancelled, !timedOut else { return }
            if let data = line.data(using: .utf8), let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               event["event"] as? String == mode.completionEvent {
                if completion != nil { completionError = mode.title + " published duplicate completion events." }
                else { completion = data }
            }
            onLine(line)
        }
        func finalize(code: Int32, message: String) async throws {
            guard code == 0, !userCancelled, !timedOut else { return }
            if let completionError { throw WorkflowArchiveError(completionError) }
            guard let completion else { throw WorkflowArchiveError(mode.title + " exited without a completed result.") }
            do { _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime) }
            catch { timedOut = true; throw error }
            try prepared.owner.check()
            let directory = prepared.owner.directory, request = prepared.request, window = self.window, mode = self.mode
            let verification = Task.detached(priority: .userInitiated) {
                switch mode {
                case .selfPlay:return try SelfPlayToolContract.verifyResult(directory:directory,request:request,completion:completion,window:window)
                case .gameSession:return try GameSessionToolContract.verifyResult(directory:directory,request:request,completion:completion,window:window)
                case .tableViewQuery: return try TableViewQueryExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                case .backendFit: return try BackendFitWorkflowExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                case .modelTraining: return try WorkflowModelTrainingExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                case .preparationInspection: return try PreparationInspectionExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                case .sourceInspection: return try SourceInspectionExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                case .preview: return try WorkflowPreviewExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                case .preparation: return try WorkflowPreparationExecution.verifyResult(directory: directory, request: request, completion: completion, window: window)
                }
            }
            finalization = verification
            defer { finalization = nil }
            do { output = try await verification.value; try prepared.owner.check() }
            catch {
                if case ExecutionBudgetError.wallTimeLimitExceeded = error { timedOut = true }
                throw error
            }
        }
        func finish(code: Int32, message: String) {
            guard !finished else { return }
            finished = true; timer?.cancel(); timer = nil; worker = nil; finalization = nil
            let result: Int32 = userCancelled ? 130 : timedOut || code == 124 ? 124 : code
            onExit(result, timedOut ? mode.title + " exceeded its captured execution deadline." : message, result == 0 ? output : nil)
        }
    }
}
