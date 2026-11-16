import Foundation
import KwipMLCore

/// One cancellable, supervised page request from a pinned immutable View artifact.
@MainActor enum TableViewQueryExecution {
    static func launch(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, indexURL: URL, query: TableViewQuery, retainedSegments: [WorkflowArchiveFile] = [],
                       storageLimitBytes: Int64? = nil,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onMaterializedIndex: ((TableViewQuerySession.Artifact) -> Void)? = nil,
                       onExit: @escaping (Int32, String, TableViewPage?) -> Void) async throws -> WorkflowWorkerHandle {
        try query.validate(workload: window.policy.resources?.workloadPolicy)
        guard indexURL.isFileURL else { throw WorkflowFileLeaseError.unsafePath }
        let raw = try WorkflowArchiveCodec.encode(query)
        var inputs: [WorkflowPreviewCapture.Input] = [
            .bytes(path: "input/view-query.json", data: raw),
            .file(path: query.inputKind == "table" ? "input/view.table.json" : "input/view.sqlite3", source: indexURL, expectedSHA256: query.indexSHA256)
        ]
        guard retainedSegments.isEmpty || query.inputKind == "table" && window.policy.resources?.workloadPolicy != nil else { throw WorkflowFileLeaseError.unsafePath }
        for file in retainedSegments {
            guard PreparationArtifactPolicy.isSegmentName(file.path) else { throw WorkflowFileLeaseError.unsafePath }
            inputs.append(.file(path:"input/"+file.path,source:indexURL.deletingLastPathComponent().appendingPathComponent(file.path),expectedSHA256:file.sha256))
        }
        let capturedInputs = inputs
        let task = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
            guard let storageLimitBytes else { return Optional<WorkflowCaptureStorageReservation>.none }
            let count = try WorkflowPreviewCapture.byteCountBound(inputs: capturedInputs, purpose: .tableViewQuery, workload: window.policy.resources?.workloadPolicy)
            return try WorkflowCaptureStorageReservation(workspace: workspace, accountingRoot: accountingRoot,
                projectID: projectID, purpose: .tableViewQuery, maximumCaptureBytes: count, limitBytes: storageLimitBytes, workload: window.policy.resources?.workloadPolicy)
        }
        let admission = try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
        var result: TableViewPage?
        var streamFailed = false
        var workerError: String?
        return try await WorkflowCapturedToolExecution.launch(mode: .tableViewQuery, workspace: workspace,
            accountingRoot: accountingRoot, projectID: projectID, runtime: runtime, window: window,
            operationID: admission?.operationID ?? UUID().uuidString, storageAdmission: admission,
            inputs: { _ in capturedInputs }, arguments: { ["--query-table-view", $0.appendingPathComponent("input/view-query.json").path] },
            launcher: launcher, onLine: { line in
                guard let data = line.data(using: .utf8),
                      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
                if object["event"] as? String == "error" {
                    streamFailed = true
                    if workerError == nil { workerError = boundedWorkerError(object) }
                    return
                }
                guard object["event"] as? String == "tableViewPage" else { return }
                guard result == nil, let page = try? TableViewPage.decode(data, query: query, workload: window.policy.resources?.workloadPolicy) else { streamFailed = true; return }
                result = page
            }, onExit: { code, message, token in
                guard code == 0, token != nil, !streamFailed, let result else {
                    onExit(code == 0 ? 1 : code, message.isEmpty ? (workerError ?? "Table View did not return a verified page.") : message, nil); return
                }
                if let index=result.materializedIndex, let directory=token {
                    onMaterializedIndex?(.init(indexURL:directory.appendingPathComponent(index.path),indexSHA256:index.sha256,sourceSHA256:query.sourceSHA256,workloadPolicy:window.policy.resources?.workloadPolicy))
                }
                onExit(0, message, result)
            })
    }
    nonisolated static func boundedWorkerError(_ event: [String: Any]) -> String? {
        guard event["event"] as? String == "error", let message = event["message"] as? String else { return nil }
        let bounded = String(message.unicodeScalars.prefix(2000)).trimmingCharacters(in: .whitespacesAndNewlines)
        return bounded.isEmpty ? nil : bounded
    }
    nonisolated static func verifyResult(directory: URL, request: WorkflowToolRequest, completion: Data,
                                         window: ExecutionBudgetWindow) throws -> URL {
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        guard request.toolMode == .tableViewQuery else { throw WorkflowFileLeaseError.unsafePath }
        try request.validate(directory: directory, verifyInputs: true)
        let query = try TableViewQuery.decode(DiagnosticFileIO.read(root: directory, parts: ["input", "view-query.json"], limit: 256_000), workload: window.policy.resources?.workloadPolicy)
        let page = try TableViewPage.decode(completion, query: query, workload: window.policy.resources?.workloadPolicy)
        if let index=page.materializedIndex {
            try WorkflowPreviewToolContract.verifyFile(file:.init(path:index.path,bytes:index.bytes,sha256:index.sha256),directory:directory,limit:TableTransferCapacity(workload: window.policy.resources?.workloadPolicy).artifactBytes)
        }
        return directory
    }
}
