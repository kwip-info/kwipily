import Foundation
import KwipMLCore

struct SourceHeaderEvidence {
    var path: String
    var config: [String: JSONValue]
    var fingerprint: String
    var columns: [String]
    var result: SourceInspectionResult? = nil
}


@MainActor final class DataFlowSourceInspectionOperation {
    let token: UUID
    let nodeID: String
    let path: String
    let config: [String: JSONValue]
    let window: ExecutionBudgetWindow
    var fingerprint: String?
    var sourceSHA256: String?
    var reservationID: UUID?
    var worker: WorkflowWorkerHandle?
    var launchTask: Task<Void, Never>?
    var verificationTask: Task<Void, Never>?
    var terminalReceived = false
    var cancelled = false
    init(token: UUID, nodeID: String, path: String, config: [String: JSONValue], window: ExecutionBudgetWindow) {
        self.token = token; self.nodeID = nodeID; self.path = path; self.config = config; self.window = window
    }
}

extension DataFlowWorkspaceStore {
    var sourceSeparatorValues: [String] { [",", "\t", ";", "|"] }
    func configureSourceSeparator(_ value:String,nodeID:String) {
        guard requirePreparationOperation() else {return}
        guard selectedNodeID==nodeID, CSVParsingConfiguration.isSeparator(value), let node=project.graph.nodes.first(where:{$0.id==nodeID && $0.operatorID=="kwip.data.sourceCSV"}) else { return }
        var config=node.config;config["separator"] = .string(value)
        configure(config,nodeID:nodeID)
        inspectSource(id:nodeID)
    }
    var sourceColumns: [String] {
        if let id = selectedNodeID { return availableColumns(for: id) }
        let sources = project.graph.nodes.filter { $0.operatorID == "kwip.data.sourceCSV" }
        return sources.count == 1 ? availableColumns(for: sources[0].id) : []
    }
    func availableColumns(for id: String) -> [String] { availableColumnSchema(for: id).map(\.name) }
    /// Cached verified sample. Reading this value never acquires a grant or reads a file.
    func sourceInspectionPreview(for id: String) -> SourceInspectionResult? {
        guard allowsPreparationOperations, projectID == workspace.activeProjectID, let evidence = sourceHeaders[id],
              project.sourceBindings[id] == evidence.path,
              project.graph.nodes.first(where: { $0.id == id && $0.operatorID == "kwip.data.sourceCSV" })?.config == evidence.config,
              let result = evidence.result, result.columns == evidence.columns else { return nil }
        return result
    }
    func sourceHeaderColumns(_ id: String) -> [String] {
        if policy == .normalizationLesson { return project.graph.nodes.contains(where:{$0.id == id && $0.operatorID == "kwip.data.sourceCSV"}) ? policy.fixtureColumns.map(\.name) : [] }
        guard allowsPreparationOperations else {return []}
        guard let evidence = sourceHeaders[id], project.sourceBindings[id] == evidence.path,
              project.graph.nodes.first(where: { $0.id == id })?.config == evidence.config,
              sourceFingerprint(evidence.path) == evidence.fingerprint else { return [] }
        return evidence.columns
    }
    func sourceFingerprint(_ path: String) -> String? {
        guard allowsPreparationOperations else {return nil}
        return try? Self.inspectionFingerprint(path, permissions: workspace.sourceAccess.directory)
    }
    nonisolated private static func inspectionFingerprint(_ path: String, permissions: URL) throws -> String {
        try SourceAccess(directory: permissions).withAccess(to: path) { url in
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .fileResourceIdentifierKey])
            return "\(values.fileSize ?? -1):\(values.contentModificationDate?.timeIntervalSince1970 ?? -1):\(String(describing: values.fileResourceIdentifier))"
        }
    }
    func inspectSource(id: String) {
        guard requirePreparationOperation() else {return}
        guard !inspectionClosed, !isPreparing, !isRestoringResults, projectID == workspace.activeProjectID,
              project.graph.nodes.contains(where: { $0.id == id && $0.operatorID == "kwip.data.sourceCSV" }), project.sourceBindings[id] != nil else { return }
        inspectionGeneration = UUID(); sourceHeaders[id] = nil
        if isInspectingSource {
            queuedInspectionSourceID = id
            cancelActiveSourceInspection()
            statusMessage = "Replacing the previous source inspection…"
            return
        }
        guard !workspace.isBusy else { statusMessage = "Wait for the current worker before inspecting a source."; return }
        startSourceInspection(id: id)
    }
    private func startSourceInspection(id: String) {
        guard requirePreparationOperation() else { return }
        guard !inspectionClosed, let path = project.sourceBindings[id],
              let node = project.graph.nodes.first(where: { $0.id == id }), let runtime = workspace.runtime else { return }
        do {
            let parsingConfiguration = try CSVParsingConfiguration.validate(node.config)
            guard case .string(let separator) = parsingConfiguration["separator"] else { return }
            let policy = try workspace.executionPolicy(for: projectID)
            let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let operation = DataFlowSourceInspectionOperation(token: inspectionGeneration, nodeID: id, path: path, config: node.config, window: window)
            let accountingRoot = workspace.applicationDirectory
            let storage = workspace.workspaceDirectory, permissions = workspace.sourceAccess.directory
            let operationProjectID = sourceInspectionProjectID
            sourceInspectionOperation = operation; isInspectingSource = true; workspace.isPreparingData = true
            // Retain the editor through capture, actual worker exit and final live-source verification.
            workspace.cancelDataPreparation = { [self] in cancelSourceInspection() }
            operation.reservationID = workspace.dataPreparationReservationID
            statusMessage = "Capturing source columns for inspection…"
            let configuredExecute = sourceInspectionExecute
            operation.launchTask = Task { [self] in
                do {
                    try Task.checkCancellation()
                    let capture = Task.detached(priority: .userInitiated) {
                        let fingerprint = try Self.inspectionFingerprint(path, permissions: permissions)
                        let snapshot = try SourceInspectionCapture.capture(path: path, workspace: storage, accountingRoot: accountingRoot, sourcePermissions: permissions, window: window,
                            projectID: operationProjectID, storageLimitBytes: policy.storageLimitBytes)
                        do {
                            guard try Self.inspectionFingerprint(path, permissions: permissions) == fingerprint else {
                                throw DataFlowContractError("Source changed during capture. Refresh its columns.")
                            }
                            return (snapshot, fingerprint)
                        } catch { snapshot.remove(); throw error }
                    }
                    let (snapshot, fingerprint) = try await withTaskCancellationHandler(operation: { try await capture.value }, onCancel: { capture.cancel() })
                    defer { snapshot.remove() }
                    try Task.checkCancellation()
                    guard inspectionContextCurrent(operation) else { throw CancellationError() }
                    operation.fingerprint = fingerprint; operation.sourceSHA256 = snapshot.sha256
                    statusMessage = "Inspecting source columns…"
                    let execute: SourceInspectionExecutor = configuredExecute ?? { workspace, project, runtime, window, source, hash, separator, exit in
                        try await SourceInspectionExecution.launch(workspace: workspace, projectID: project, runtime: runtime,
                            window: window, sourceURL: source, sourceSHA256: hash, separator: separator,
                            configuration: parsingConfiguration, storageAdmission: snapshot.storageReservation, onExit: exit)
                    }
                    let handle = try await execute(storage, operationProjectID, runtime, window,
                        snapshot.sourceURL, snapshot.sha256, separator, { [self] code, stderr, result in
                            receiveSourceInspectionExit(operation, code: code, stderr: stderr, result: result)
                        })
                    if sourceInspectionOperation === operation && !operation.terminalReceived {
                        operation.worker = handle
                        if operation.cancelled { handle.cancel() }
                    }
                    operation.launchTask = nil
                } catch {
                    operation.launchTask = nil
                    if !operation.terminalReceived {
                        operation.terminalReceived = true
                        finishSourceInspection(operation, message: error.localizedDescription)
                    }
                }
            }
        } catch { statusMessage = error.localizedDescription }
    }
    private func inspectionContextCurrent(_ operation: DataFlowSourceInspectionOperation) -> Bool {
        sourceInspectionOperation === operation && !operation.cancelled && !inspectionClosed
            && inspectionGeneration == operation.token && projectID == workspace.activeProjectID
            && operation.reservationID == workspace.dataPreparationReservationID
            && project.sourceBindings[operation.nodeID] == operation.path
            && project.graph.nodes.first(where: { $0.id == operation.nodeID })?.config == operation.config
    }
    private func receiveSourceInspectionExit(_ operation: DataFlowSourceInspectionOperation, code: Int32,
                                              stderr: String, result: SourceInspectionResult?) {
        guard sourceInspectionOperation === operation, !operation.terminalReceived else { return }
        operation.terminalReceived = true; operation.worker = nil
        guard inspectionContextCurrent(operation) else { finishSourceInspection(operation, message: nil); return }
        guard code == 0, let result, result.sourceSHA256 == operation.sourceSHA256 else {
            let message = code == 124 ? "Source inspection exceeded its time allowance. Reduce the input size or adjust the execution budget."
                : (CSVSourceDiagnostic.decodeTransport(stderr)?.detail
                    ?? (stderr.isEmpty ? "Source inspection did not finish with verified columns. Refresh its columns." : "Source inspection failed: \(stderr.prefix(500))"))
            finishSourceInspection(operation, message: message); return
        }
        let path = operation.path, window = operation.window
        let permissions = workspace.sourceAccess.directory
        operation.verificationTask = Task { [self] in
            do {
                try Task.checkCancellation()
                let verification = Task.detached(priority: .userInitiated) {
                    try SourceInspectionCapture.verifyCurrent(path: path, sourcePermissions: permissions,
                        window: window, expectedSHA256: result.sourceSHA256)
                    return try Self.inspectionFingerprint(path, permissions: permissions)
                }
                let fingerprint = try await withTaskCancellationHandler(operation: { try await verification.value }, onCancel: { verification.cancel() })
                try Task.checkCancellation()
                guard fingerprint == operation.fingerprint else {
                    throw DataFlowContractError("Source changed during inspection. Refresh its columns.")
                }
                guard inspectionContextCurrent(operation) else { throw CancellationError() }
                _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                sourceHeaders[operation.nodeID] = SourceHeaderEvidence(path: path, config: operation.config,
                    fingerprint: fingerprint, columns: result.columns, result: result)
                finishSourceInspection(operation, message: "Source columns inspected from a verified snapshot.")
            } catch { finishSourceInspection(operation, message: error.localizedDescription) }
        }
    }
    private func finishSourceInspection(_ operation: DataFlowSourceInspectionOperation, message: String?) {
        guard sourceInspectionOperation === operation else { return }
        sourceInspectionOperation = nil; operation.worker = nil; operation.verificationTask = nil
        isInspectingSource = false
        let ownsReservation = operation.reservationID == workspace.dataPreparationReservationID
        if ownsReservation { workspace.isPreparingData = false; workspace.cancelDataPreparation = nil }
        let next = queuedInspectionSourceID; queuedInspectionSourceID = nil
        let prepare = prepareAfterInspection; prepareAfterInspection = false
        guard !inspectionClosed, projectID == workspace.activeProjectID, ownsReservation else { return }
        if operation.cancelled || inspectionGeneration != operation.token {
            if next == nil, !prepare { statusMessage = "Source inspection cancelled." }
        } else if let message { statusMessage = message }
        if prepare { runPreparation() }
        else if let next { inspectSource(id: next) }
    }
    private func cancelActiveSourceInspection() {
        guard let operation = sourceInspectionOperation else { return }
        operation.cancelled = true
        operation.launchTask?.cancel(); operation.verificationTask?.cancel(); operation.worker?.cancel()
    }
    func cancelSourceInspection(cancelQueuedWork: Bool = true) {
        inspectionGeneration = UUID()
        if cancelQueuedWork { queuedInspectionSourceID = nil; prepareAfterInspection = false }
        cancelActiveSourceInspection()
        if isInspectingSource { statusMessage = "Cancelling source inspection…" }
    }
    func closeSourceInspection() { inspectionClosed = true; cancelSourceInspection(); cancelResultRestoration() }
}
