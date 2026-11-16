import Foundation
import KwipMLCore

typealias PreparationInspectionExecutor = @MainActor (URL, String, RuntimeLocation, ExecutionBudgetWindow, URL, String, @escaping (Int32, String, PreparationInspectionResult?) -> Void) async throws -> WorkflowWorkerHandle

/// Both historical views use the same captured inputs, strict stream and joined result acceptance.
@MainActor enum PreparationInspectionExecution {
    static func launch(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, manifestURL: URL, manifestSHA256: String,
                       storageLimitBytes: Int64? = nil, storageAdmission: WorkflowCaptureStorageReservation? = nil,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onExit: @escaping (Int32, String, PreparationInspectionResult?) -> Void) async throws -> WorkflowWorkerHandle {
        guard storageLimitBytes == nil || storageAdmission == nil else {
            throw WorkflowArchiveError("Provide one storage admission for saved-output inspection.")
        }
        let captureTask = Task.detached(priority: .userInitiated) {
            try check(window)
            let expectation = try PreparationInspectionExpectation.read(manifestURL: manifestURL, manifestSHA256: manifestSHA256)
            let raw = try DiagnosticFileIO.read(root: manifestURL.deletingLastPathComponent(), parts: [manifestURL.lastPathComponent], limit: 2_000_000)
            guard WorkflowArchiveCodec.hash(raw) == manifestSHA256 else { throw WorkflowFileLeaseError.unsafePath }
            let manifest = try JSONDecoder().decode(Inventory.self, from: raw)
            let files = manifest.artifacts.map { WorkflowArchiveFile(path: "input/preparation/" + $0.path, bytes: $0.bytes, sha256: $0.sha256) }
                + [.init(path: "input/preparation/manifest.json", bytes: raw.count, sha256: manifestSHA256)]
            try WorkflowCompletedPreparationInventory.verify(data: raw, files: files, prefix: "input/preparation/")
            var inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/preparation/manifest.json", data: raw)]
            for file in manifest.artifacts {
                try check(window)
                inputs.append(.file(path: "input/preparation/" + file.path,
                    source: manifestURL.deletingLastPathComponent().appendingPathComponent(file.path), expectedSHA256: file.sha256))
            }
            let admission: WorkflowCaptureStorageReservation?
            if let storageLimitBytes {
                let count = try WorkflowPreviewCapture.byteCountBound(inputs: inputs, purpose: .preparationInspection, workload: window.policy.resources?.workloadPolicy)
                admission = try WorkflowCaptureStorageReservation(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
                    purpose: .preparationInspection, maximumCaptureBytes: count, limitBytes: storageLimitBytes, workload: window.policy.resources?.workloadPolicy)
            } else { admission = storageAdmission }
            return (expectation, inputs, admission)
        }
        let (expectation, inputs, admission) = try await withTaskCancellationHandler(operation: { try await captureTask.value }, onCancel: { captureTask.cancel() })
        try check(window)
        var stream = PreparationInspectionStream(expectation: expectation)
        var streamFailure: String?
        return try await WorkflowCapturedToolExecution.launch(mode: .preparationInspection, workspace: workspace, accountingRoot: accountingRoot,
            projectID: projectID, runtime: runtime, window: window,
            operationID: admission?.operationID ?? UUID().uuidString, storageAdmission: admission, inputs: { _ in inputs },
            arguments: { ["--inspect-dataflow", $0.appendingPathComponent("input/preparation/manifest.json").path,
                          "--dataflow-manifest-sha256", manifestSHA256] }, launcher: launcher,
            onLine: { line in
                guard streamFailure == nil else { return }
                do { try stream.receive(line) } catch { streamFailure = error.localizedDescription }
            }, onExit: { code, message, verified in
                guard code == 0, verified != nil else { onExit(code == 0 ? 1 : code, message, nil); return }
                if let streamFailure { onExit(1, streamFailure, nil); return }
                do { onExit(0, message, try stream.finish(exitCode: code)) }
                catch { onExit(1, error.localizedDescription, nil) }
            })
    }
    private struct Inventory: Decodable { let artifacts: [WorkflowArchiveFile] }
    private nonisolated static func check(_ window: ExecutionBudgetWindow) throws {
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
    }
    nonisolated static func verifyResult(directory: URL, request: WorkflowToolRequest, completion: Data,
                                         window: ExecutionBudgetWindow) throws -> URL {
        try check(window)
        guard request.toolMode == .preparationInspection else { throw WorkflowFileLeaseError.unsafePath }
        try request.validate(directory: directory, verifyInputs: true)
        let input = directory.appendingPathComponent("input/preparation")
        let raw = try WorkflowEvaluationOwnership.readProtocol(input, name: "manifest.json", limit: 2_000_000)
        let manifest = try JSONDecoder().decode(Inventory.self, from: raw)
        guard let event = try JSONSerialization.jsonObject(with: completion) as? [String: Any],
              event["event"] as? String == "dataflowInspectionCompleted",
              event["outputDir"] as? String == input.path,
              event["manifestPath"] as? String == input.appendingPathComponent("manifest.json").path,
              event["manifestSHA256"] as? String == request.arguments[3],
              event["artifacts"] as? [String] == manifest.artifacts.map(\.path) else {
            throw WorkflowArchiveError("Saved-output completion differs from its captured archive.")
        }
        try check(window)
        return directory
    }
}
