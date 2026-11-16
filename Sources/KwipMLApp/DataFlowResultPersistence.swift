import Foundation
import KwipMLCore

/// Private per-project pointer; immutable output bytes are never rewritten on restoration.
struct DataFlowCompletedReference: Codable, Equatable {
    var schemaVersion = 1
    var projectID: String
    var manifestRelativePath: String
    var manifestSHA256: String
    var graphDigest: String
    var graph: DataFlowDocument
    var bindings: [String: String]
    var sourceDigests: [String: String]
    var completedAt: Date

    /// New operation-owned outputs are project-scoped; historical run paths retain schema 1.
    static func version(for path: String, projectID: String) -> Int? {
        let pieces = path.split(separator: "/", omittingEmptySubsequences: false)
        guard pieces.count == 5, !pieces.contains(".."), !pieces.contains("."), !pieces.contains(""),
              pieces[3] == "artifacts", pieces[4] == "manifest.json" else { return nil }
        if pieces[0] == "dataflow", pieces[1] == "runs" { return 1 }
        guard pieces[0] == "workflow-operations", pieces[1] == Substring(projectID),
              UUID(uuidString: projectID)?.uuidString == projectID,
              UUID(uuidString: String(pieces[2]))?.uuidString == String(pieces[2]) else { return nil }
        return 2
    }

    static func canonicalPath(_ url: URL) throws -> String {
        guard let pointer = realpath(url.path, nil) else {
            throw DataFlowContractError("Preparation output path is unavailable.")
        }
        defer { free(pointer) }
        return String(cString: pointer)
    }
}

extension DataFlowWorkspaceStore {
    var completedResultURL: URL { saveURL.deletingLastPathComponent().appendingPathComponent("last-result.json") }
    var resultMatchesEditor: Bool { resultGraph == project.graph && resultBindings == project.sourceBindings }
    var resultProvenance: String? {
        guard policy == .preparation else {return nil}
        guard manifestURL != nil else { return nil }
        let when = completedResult.map { " · " + $0.completedAt.formatted(date: .abbreviated, time: .shortened) } ?? ""
        if restoredResult {
            return resultMatchesEditor ? "Restored historical outputs\(when) · artifacts verified · current sources unchecked. Prepare again before training." : "Restored historical outputs\(when) · flow or bindings changed · current sources unchecked."
        }
        return resultMatchesEditor ? "Last completed preparation\(when) · verified at execution" : "Last completed preparation\(when) · flow or bindings changed"
    }
    func persistCompletedResult(manifest: URL, graph: DataFlowDocument, bindings: [String: String], digests: [String: String]) throws {
        try requirePreparationAccess()
        guard let projectID, projectID == workspace.activeProjectID else { throw DataFlowContractError("The preparation project changed before results were saved.") }
        let root = try DataFlowCompletedReference.canonicalPath(workspace.workspaceDirectory) + "/"
        let path = try DataFlowCompletedReference.canonicalPath(manifest)
        guard path.hasPrefix(root) else { throw DataFlowContractError("Preparation output is outside managed storage.") }
        let relative = String(path.dropFirst(root.count))
        guard let version = DataFlowCompletedReference.version(for: relative, projectID: projectID) else {
            throw DataFlowContractError("Preparation output has an invalid project or storage path.")
        }
        let bytes = try WorkspaceDocumentReader.read(manifest)
        guard let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any], let graphDigest = object["graphDigest"] as? String else { throw DataFlowContractError("Preparation manifest has no graph identity.") }
        let reference = DataFlowCompletedReference(schemaVersion: version, projectID: projectID, manifestRelativePath: relative, manifestSHA256: Self.cacheDigest(bytes), graphDigest: graphDigest, graph: graph, bindings: bindings, sourceDigests: digests, completedAt: Date())
        try FileManager.default.createDirectory(at: completedResultURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try DataFlowCodec.encodeProject(project).write(to: saveURL, options: .atomic)
        let data = try JSONEncoder().encode(reference)
        guard data.count <= 2_000_000 else { throw DataFlowContractError("Completed-result reference exceeds its size bound.") }
        try data.write(to: completedResultURL, options: .atomic)
        completedResult = reference
    }
    func readCompletedReference() throws -> (DataFlowCompletedReference, URL) {
        try requirePreparationAccess()
        return try Self.readCompletedReference(referenceURL: completedResultURL, workspace: workspace.workspaceDirectory, projectID: projectID)
    }
    nonisolated private static func readCompletedReference(referenceURL: URL, workspace: URL, projectID: String?) throws -> (DataFlowCompletedReference, URL) {
        let reference = try JSONDecoder().decode(DataFlowCompletedReference.self, from: WorkspaceDocumentReader.read(referenceURL))
        func hash(_ value: String) -> Bool { value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) } }
        guard reference.projectID == projectID,
              DataFlowCompletedReference.version(for: reference.manifestRelativePath, projectID: reference.projectID) == reference.schemaVersion,
              hash(reference.manifestSHA256), hash(reference.graphDigest),
              reference.bindings.count <= 128, reference.sourceDigests.count <= 128,
              reference.bindings.values.allSatisfy({ $0.count <= 4096 }), reference.sourceDigests.values.allSatisfy(hash) else {
            throw DataFlowContractError("Completed-result reference has invalid project, path or identity fields.")
        }
        _ = try DataFlowCodec.decodeGraph(DataFlowCodec.encodeGraph(reference.graph))
        let manifest = workspace.appendingPathComponent(reference.manifestRelativePath)
        let root = try DataFlowCompletedReference.canonicalPath(workspace) + "/"
        let path = try DataFlowCompletedReference.canonicalPath(manifest)
        guard path.hasPrefix(root), reference.schemaVersion == 1 || path == root + reference.manifestRelativePath else {
            throw DataFlowContractError("Completed-result path escapes its managed storage location.")
        }
        guard WorkflowArchiveCodec.hash(try WorkspaceDocumentReader.read(manifest)) == reference.manifestSHA256 else { throw DataFlowContractError("Completed preparation manifest changed; historical outputs were not restored.") }
        let archived = try DataFlowCodec.decodeGraph(WorkspaceDocumentReader.read(manifest.deletingLastPathComponent().appendingPathComponent("dataflow.json")))
        guard archived == reference.graph else { throw DataFlowContractError("Completed preparation graph does not match its saved reference.") }
        return (reference, reference.schemaVersion == 2 ? URL(fileURLWithPath: path) : manifest)
    }
}

@MainActor final class DataFlowRestorationOperation {
    let token = UUID()
    let window: ExecutionBudgetWindow
    var reservationID: UUID?
    var launchTask: Task<Void, Never>?
    var verificationTask: Task<Void, Never>?
    var worker: WorkflowWorkerHandle?
    var terminalReceived = false
    var cancelled = false
    init(window: ExecutionBudgetWindow) { self.window = window }
}

extension DataFlowWorkspaceStore {
    /// Verifies only archived artifacts; never opens a current source binding.
    func restoreCompletedOutputs() {
        guard requirePreparationOperation() else { return }
        guard !isBusy, !workspace.isBusy, !inspectionClosed, let projectID, projectID == workspace.activeProjectID,
              let runtime = workspace.runtime else { return }
        guard FileManager.default.fileExists(atPath: completedResultURL.path) else { return }
        do {
            let policy = try workspace.executionPolicy(for: projectID)
            let window = try ExecutionBudgetWindow(policy: policy.budget,
                startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let operation = DataFlowRestorationOperation(window: window)
            restorationOperation = operation; isRestoringResults = true; workspace.isPreparingData = true
            workspace.cancelDataPreparation = { [weak self] in self?.cancelResultRestoration() }
            operation.reservationID = workspace.dataPreparationReservationID
            statusMessage = "Verifying saved preparation artifacts; current sources will not be read…"
            let referenceURL = completedResultURL, root = workspace.workspaceDirectory, accountingRoot = workspace.applicationDirectory
            let execute: PreparationInspectionExecutor = resultRestorationExecute ?? { root, project, runtime, window, manifest, pin, exit in
                try await PreparationInspectionExecution.launch(workspace: root, accountingRoot: accountingRoot, projectID: project, runtime: runtime,
                    window: window, manifestURL: manifest, manifestSHA256: pin,
                    storageLimitBytes: policy.storageLimitBytes, onExit: exit)
            }
            operation.launchTask = Task { [self] in
                do {
                    let reader = Task.detached(priority: .userInitiated) {
                        try Task.checkCancellation()
                        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                        let captured = try Self.readCompletedReference(referenceURL: referenceURL, workspace: root, projectID: projectID)
                        try Task.checkCancellation()
                        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                        return captured
                    }
                    let (reference, manifest) = try await withTaskCancellationHandler(operation: { try await reader.value }, onCancel: { reader.cancel() })
                    try Task.checkCancellation()
                    guard restorationContextCurrent(operation) else { throw CancellationError() }
                    let handle = try await execute(root, projectID, runtime, window, manifest, reference.manifestSHA256) { [self] code, message, result in
                        guard restorationOperation === operation, !operation.terminalReceived else { return }
                        operation.terminalReceived = true; operation.worker = nil
                        guard restorationContextCurrent(operation), code == 0, let result else {
                            finishResultRestoration(operation, message: message.isEmpty ? "verification did not complete." : message); return
                        }
                        operation.verificationTask = Task { [self] in
                            do {
                                let conversion = Task.detached(priority: .userInitiated) { () throws -> [String: [String: Any]] in
                                    try Task.checkCancellation()
                                    _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                                    guard Set(result.outputs.keys) == Set(reference.graph.nodes.map { WorkflowNodeAddress(domain: .data, localID: $0.id) }) else {
                                        throw DataFlowContractError("Saved outputs do not match the archived graph.")
                                    }
                                    let values = Dictionary(uniqueKeysWithValues: result.outputs.map { ($0.key.localID, $0.value) })
                                    let bytes = try JSONEncoder().encode(values)
                                    guard bytes.count <= 32_000_000, let previews = try JSONSerialization.jsonObject(with: bytes) as? [String: [String: Any]] else {
                                        throw DataFlowContractError("Saved outputs exceed their display bounds.")
                                    }
                                    try Task.checkCancellation()
                                    return previews
                                }
                                let staged = try await withTaskCancellationHandler(operation: { try await conversion.value }, onCancel: { conversion.cancel() })
                                try Task.checkCancellation()
                                _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                                guard restorationContextCurrent(operation) else { throw CancellationError() }
                                completedResult = reference; manifestURL = manifest; previews = staged
                                resultGraph = reference.graph; resultBindings = reference.bindings; resultSourceDigests = reference.sourceDigests
                                restoredResult = true; cacheReport = result.cacheReport
                                finishResultRestoration(operation, message: nil)
                            } catch { finishResultRestoration(operation, message: error.localizedDescription) }
                        }
                    }
                    guard !operation.terminalReceived else { return }
                    if restorationOperation === operation {
                        operation.worker = handle
                        if operation.cancelled { handle.cancel() }
                    } else { handle.cancel() }
                } catch { finishResultRestoration(operation, message: error.localizedDescription) }
            }
        } catch { statusMessage = "Saved outputs were not restored: \(error.localizedDescription)" }
    }
    private func restorationContextCurrent(_ operation: DataFlowRestorationOperation) -> Bool {
        restorationOperation === operation && !operation.cancelled && !inspectionClosed
            && projectID == workspace.activeProjectID && operation.reservationID == workspace.dataPreparationReservationID
    }
    private func finishResultRestoration(_ operation: DataFlowRestorationOperation, message: String?) {
        guard restorationOperation === operation else { return }
        restorationOperation = nil; operation.worker = nil; operation.launchTask = nil; operation.verificationTask = nil
        isRestoringResults = false
        let ownsReservation = operation.reservationID == workspace.dataPreparationReservationID
        if ownsReservation { workspace.isPreparingData = false; workspace.cancelDataPreparation = nil }
        guard ownsReservation, projectID == workspace.activeProjectID else { return }
        if operation.cancelled || inspectionClosed { statusMessage = "Saved-output verification cancelled; no new outputs accepted." }
        else if let message { statusMessage = "Saved outputs were not restored: " + String(message.prefix(500)) }
        else { statusMessage = "Historical outputs restored from verified artifacts. Current source files were not read." }
    }
    func cancelResultRestoration() {
        guard let operation = restorationOperation else { return }
        operation.cancelled = true
        operation.launchTask?.cancel(); operation.verificationTask?.cancel(); operation.worker?.cancel()
        statusMessage = "Cancelling saved-output verification…"
    }
}
