import Foundation
import KwipMLCore

extension WorkspaceStore {
    func inspectData() {
        guard !isBusy else { return }
        guard let path = execution.datasetCSV else { dataPreviewSourcePath = nil; dataColumns = []; dataRows = []; dataRowCount = 0; dataError = nil; return }
        guard let runtime else { dataError = runtimeError ?? "Runtime unavailable."; return }
        let token = UUID(), project = activeProjectID
        let scope = project ?? transientDataInspectionProjectID
        let accountingRoot = applicationDirectory
        let workspace = workspaceDirectory, permissions = sourceAccess.directory
        let window: ExecutionBudgetWindow
        let policy: ResolvedExecutionPreferences
        do {
            policy = try executionPolicy(for: project)
            window = try ExecutionBudgetWindow(policy: policy.budget,
                startedAtUptime: ProcessInfo.processInfo.systemUptime)
        } catch { dataError = error.localizedDescription; return }
        dataInspectionGeneration = token; activeDataInspectionToken = token; dataInspectionTerminalToken = nil
        isInspectingData = true
        if dataPreviewSourcePath != path {
            dataPreviewSourcePath = nil; dataColumns = []; dataRows = []; dataRowCount = 0
        }
        let configuredExecute = dataInspectionExecute
        dataInspectionTask = Task { [self] in
            do {
                let captureTask = Task.detached(priority: .userInitiated) {
                    try SourceInspectionCapture.capture(path: path, workspace: workspace, accountingRoot: accountingRoot, sourcePermissions: permissions, window: window,
                        projectID: scope, storageLimitBytes: policy.storageLimitBytes)
                }
                let capture = try await withTaskCancellationHandler(operation: { try await captureTask.value }, onCancel: { captureTask.cancel() })
                defer { capture.remove() }
                try Task.checkCancellation()
                guard dataInspectionGeneration == token, activeProjectID == project, execution.datasetCSV == path else {
                    finishDataInspection(token: token, path: path, project: project, result: nil, failure: nil); return
                }
                let execute: SourceInspectionExecutor = configuredExecute ?? { workspace, project, runtime, window, source, hash, separator, exit in
                    try await SourceInspectionExecution.launch(workspace: workspace, projectID: project, runtime: runtime,
                        window: window, sourceURL: source, sourceSHA256: hash, separator: separator,
                        storageAdmission: capture.storageReservation, onExit: exit)
                }
                let handle = try await execute(workspace, scope, runtime, window, capture.sourceURL, capture.sha256, ",") { [self] code, message, result in
                    guard activeDataInspectionToken == token, dataInspectionTerminalToken != token else { return }
                    dataInspectionTerminalToken = token
                    dataInspectionHandle = nil
                    guard dataInspectionGeneration == token, activeProjectID == project, execution.datasetCSV == path else {
                        finishDataInspection(token: token, path: path, project: project, result: nil, failure: nil); return
                    }
                    guard code == 0, let result, result.sourceSHA256 == capture.sha256 else {
                        finishDataInspection(token: token, path: path, project: project, result: nil,
                            failure: message.isEmpty ? "Source inspection did not finish. Refresh the preview." : message); return
                    }
                    dataInspectionValidationTask = Task { [self] in
                        do {
                            let verification = Task.detached(priority: .userInitiated) {
                                try SourceInspectionCapture.verifyCurrent(path: path, sourcePermissions: permissions,
                                    window: window, expectedSHA256: result.sourceSHA256)
                            }
                            try await withTaskCancellationHandler(operation: { try await verification.value }, onCancel: { verification.cancel() })
                            try Task.checkCancellation()
                            finishDataInspection(token: token, path: path, project: project, result: result, failure: nil)
                        } catch {
                            finishDataInspection(token: token, path: path, project: project, result: nil, failure: error.localizedDescription)
                        }
                    }
                }
                if dataInspectionTerminalToken == token { return }
                if activeDataInspectionToken == token {
                    dataInspectionHandle = handle
                    if dataInspectionGeneration != token { handle.cancel() }
                } else { handle.cancel() }
            } catch {
                finishDataInspection(token: token, path: path, project: project, result: nil, failure: error.localizedDescription)
            }
        }
    }
    private func finishDataInspection(token: UUID, path: String, project: String?, result: SourceInspectionResult?, failure: String?) {
        guard activeDataInspectionToken == token else { return }
        dataInspectionHandle = nil; dataInspectionTask = nil; dataInspectionValidationTask = nil
        activeDataInspectionToken = nil; isInspectingData = false
        guard dataInspectionGeneration == token else { statusMessage = "Source inspection cancelled; prior preview retained."; return }
        guard activeProjectID == project, execution.datasetCSV == path else { return }
        if let result {
            dataColumns = result.columns; dataRows = result.rows; dataRowCount = result.rowCount
            dataPreviewSourcePath = path; dataError = nil
        } else if let failure { dataError = failure }
    }
    func cancelDataInspection() {
        guard isInspectingData else { return }
        dataInspectionGeneration = UUID()
        dataInspectionTask?.cancel(); dataInspectionValidationTask?.cancel(); dataInspectionHandle?.cancel()
        statusMessage = "Cancelling source inspection…"
    }
}
