import Foundation
import CryptoKit
import KwipMLCore

extension WorkspaceStore {
    /// Resolve a primitive by its canonical authoring path, never by a Block's synthetic canvas ID.
    func expandedPreviewNodeID(instancePath: [String], localNodeID: String) throws -> String {
        guard let owner = modelAuthoring, owner.projectID == activeProjectID else {
            throw previewInputError("Open the model's current authoring session before inspecting it.")
        }
        return try previewNodeID(in: ModelExecutionRequest.capture(from: owner, registry: registry), instancePath: instancePath, localNodeID: localNodeID)
    }
    private func previewNodeID(in request: ModelExecutionRequest, instancePath: [String], localNodeID: String) throws -> String {
        let matches = request.snapshot.origins.filter { $0.value.instancePath == instancePath && $0.value.localNodeID == localNodeID }
        guard matches.count == 1, let id = matches.first?.key else {
            throw previewInputError("Select a primitive inside this Block. A Block boundary is not a single executable tensor.")
        }
        return id
    }
    func canPreviewModelNode(instancePath: [String], localNodeID: String) -> Bool {
        !isBusy && !isLearningLessonActive && runtime != nil && capabilityBoundaryError == nil &&
            (try? expandedPreviewNodeID(instancePath: instancePath, localNodeID: localNodeID)) != nil
    }
    private func retainedModelPreviewNodeID(instancePath: [String], localNodeID: String) -> String? {
        guard let request = previewModelRequest, request.projectID == activeProjectID,
              request.isCurrent(in: modelAuthoring), previewSignature == request.executionSignature,
              previewCachePath != nil else { return nil }
        return try? previewNodeID(in: request, instancePath: instancePath, localNodeID: localNodeID)
    }
    func modelPreview(instancePath: [String], localNodeID: String) -> [String: Any]? {
        guard let id = retainedModelPreviewNodeID(instancePath: instancePath, localNodeID: localNodeID), cacheValid else { return nil }
        return previews[id]
    }
    func modelPreviewStatus(instancePath: [String], localNodeID: String) -> String {
        guard let id = retainedModelPreviewNodeID(instancePath: instancePath, localNodeID: localNodeID) else { return "Not evaluated" }
        return previewStatus(for: id)
    }
    func canResumeModelNode(instancePath: [String], localNodeID: String) -> Bool {
        guard canPreviewModelNode(instancePath: instancePath, localNodeID: localNodeID),
              let id = retainedModelPreviewNodeID(instancePath: instancePath, localNodeID: localNodeID) else { return false }
        return canResumePreviewNode(id, validation: validation)
    }
    func previewModelNode(instancePath: [String], localNodeID: String, resume: Bool = false) {
        guard allowExecution() else {return}
        guard allowOrdinaryExecution(), !isBusy else { return }
        do {
            guard let owner = modelAuthoring, owner.projectID == activeProjectID else { throw previewInputError("Open the model's current authoring session before inspecting it.") }
            let request = try ModelExecutionRequest.capture(from: owner, registry: registry)
            let id = try previewNodeID(in: request, instancePath: instancePath, localNodeID: localNodeID)
            guard !resume || canResumeModelNode(instancePath: instancePath, localNodeID: localNodeID) else { throw previewInputError("Cache is stale. Preview this flow again first.") }
            launchPreview(resume: resume, selectedID: id, modelRequest: request)
        } catch { statusMessage = "Preview unavailable: \(error.localizedDescription)" }
    }
    private func canResumePreviewNode(_ id: String, validation: ValidationResult) -> Bool {
        guard cacheValid, validation.isValid, let index = validation.orderedNodeIDs.firstIndex(of: id) else { return false }
        let captured = Set(previewExecuted + previewReused)
        return validation.orderedNodeIDs.prefix(index).allSatisfy { captured.contains($0) }
    }
    func capturePreviewArtifactHashes() {
        previewArtifactHashes = [:]
        guard let cache = previewCachePath else { return }
        var paths = [cache, URL(fileURLWithPath: cache).deletingPathExtension().appendingPathExtension("json").path]
        if let checkpoint = previewCheckpoint { paths += [checkpoint, URL(fileURLWithPath: checkpoint).deletingLastPathComponent().appendingPathComponent("graph.json").path] }
        for path in paths { if previewFileIdentity(path) != nil, let hash = sourceDigest(path, maximumBytes: 100_000_000) { previewArtifactHashes[path] = hash } }
    }
    var canResumeSelected: Bool {
        guard let id = selectedNodeID else { return false }
        return canResumePreviewNode(id, validation: validation)
    }

    var cacheValid: Bool {
        if modelAuthoring != nil {
            guard let request = previewModelRequest, request.projectID == activeProjectID,
                  request.isCurrent(in: modelAuthoring), previewSignature == request.executionSignature else { return false }
        }
        guard let path = previewCachePath, FileManager.default.fileExists(atPath: path), previewSignature == semanticSignature(graph) else { return false }
        let metadata = URL(fileURLWithPath: path).deletingPathExtension().appendingPathExtension("json").path
        guard previewArtifactHashes[path] != nil, previewArtifactHashes[metadata] != nil,
              previewArtifactHashes.allSatisfy({ previewFileIdentity($0.key) != nil && ($0.value.hasPrefix("metadata:") ? previewFileIdentity($0.key) : sourceDigest($0.key, maximumBytes: 100_000_000)) == $0.value }) else { return false }
        if execution.preparedTraining != nil { return previewArtifactHashes.keys.contains(execution.preparedTraining!.manifestPath) }
        if let source = execution.datasetCSV { return previewSourceHash != nil && previewSourceHash == sourceDigest(source) }
        return true
    }
    func previewStatus(for id: String) -> String {
        if previewCachePath != nil {
            guard cacheValid else { return previews[id] == nil ? "Not evaluated" : "Stale" }
            if previewReused.contains(id) { return "Cached" }
            if previewExecuted.contains(id) { return "Computed" }
        }
        if previews[id] != nil { return resultsStale ? "Stale" : "Computed" }
        return "Not evaluated"
    }
    func previewSelected() { launchPreview(resume: false) }
    func resumeSelected() { launchPreview(resume: true) }
    private func launchPreview(resume: Bool, selectedID: String? = nil, modelRequest: ModelExecutionRequest? = nil) {
        guard allowExecution() else {return}
        guard allowOrdinaryExecution() else { return }
        guard !isBusy, validation.isValid, let selected = selectedID ?? selectedNodeID else { statusMessage = "Select a block in a valid flow first."; return }
        guard !resume || canResumePreviewNode(selected, validation: validation) else { statusMessage = "Cache is stale. Preview this flow again first."; return }
        guard let runtime else { statusMessage = runtimeError ?? "Runtime unavailable."; return }
        if let error = capabilityBoundaryError { statusMessage = error; return }
        do {
            if activeProjectID == nil, !saveManagedProject() { throw previewInputError(saveError ?? "Save the project before previewing.") }
            guard let projectID = activeProjectID else { throw previewInputError("Open a project before previewing.") }
            let capturedRequest = try modelRequest ?? modelAuthoring.map { try ModelExecutionRequest.capture(from: $0, registry: registry) }
            let capturedGraph = capturedRequest?.graph ?? graph
            let settings = capturedRequest?.settings ?? execution
            let signature = capturedRequest?.executionSignature ?? semanticSignature(capturedGraph)
            let session = historySessionRevision, token = UUID()
            let preparedVersions = try preparedPreviewVersions(settings: settings)
            let selectedCheckpoint = resume ? previewCheckpoint : (!resultsStale && lastCompleted ? outputDirectory?.appendingPathComponent("checkpoint.pt").path : nil)
            let cache = previewCachePath, hashes = previewArtifactHashes, permissions = sourceAccess.directory
            let policy = try executionPolicy(for: projectID)
            let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let contextCurrent: () -> Bool = { [weak self] in
                guard let self, self.activeProjectID == projectID, self.historySessionRevision == session else { return false }
                if let capturedRequest { return capturedRequest.isCurrent(in: self.modelAuthoring) }
                return self.modelAuthoring == nil
            }
            previewRequestToken = token; previewCancelled = false; isPreviewing = true; showPreview = true
            statusMessage = "Capturing inference inputs…"
            previewLaunchTask = Task { [weak self] in
                guard let self else { return }
                do {
                    let capture = Task.detached(priority: .userInitiated) {
                        try PreviewLaunchPlan.capture(graph: capturedGraph, settings: settings, selected: selected, resume: resume,
                            cache: cache, checkpoint: selectedCheckpoint, hashes: hashes, sourcePermissions: permissions)
                    }
                    let plan = try await withTaskCancellationHandler(operation: { try await capture.value }, onCancel: { capture.cancel() })
                    try Task.checkCancellation()
                    guard contextCurrent(), self.previewRequestToken == token else { throw CancellationError() }
                    self.statusMessage = resume ? "Executing from selected block using cached upstream data…" : "Computing inference preview through selected block…"
                    var completion: [String: Any]?, errorMessage: String?
                    let handle = try await WorkflowPreviewExecution.launch(workspace: self.workspaceDirectory, accountingRoot: self.applicationDirectory, projectID: projectID,
                        runtime: runtime, window: window, storageLimitBytes: policy.storageLimitBytes, inputs: plan.inputs, arguments: plan.arguments,
                        launcher: self.previewWorkerLauncher, onLine: { [weak self] line in
                            guard let self, self.previewRequestToken == token, contextCurrent(),
                                  let data = line.data(using: .utf8), let event = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return }
                            if event["event"] as? String == "previewCompleted" { completion = event }
                            if event["event"] as? String == "error" { errorMessage = event["message"] as? String }
                            if event["event"] as? String == "previewStarted" { self.resolvedDevice = event["device"] as? String ?? self.resolvedDevice }
                        }, onExit: { [weak self] code, stderr, artifacts in
                            guard let self, self.previewRequestToken == token else { return }
                            self.previewRequestToken = nil; self.isPreviewing = false; self.previewWorker = nil; self.previewLaunchTask = nil
                            guard contextCurrent() else { return }
                            guard !self.previewCancelled, code != 130 else { self.statusMessage = "Inference preview cancelled."; return }
                            if code == 124 { self.statusMessage = "Inference preview exceeded its time allowance. Reduce the input size or adjust the execution budget."; return }
                            guard code == 0, let result = completion, let artifacts else {
                                self.statusMessage = "Preview failed: \(errorMessage ?? (stderr.isEmpty ? "No verified preview artifact." : stderr))"; return
                            }
                            guard self.semanticSignature(self.graph) == signature,
                                  (try? self.preparedPreviewVersions(settings: settings)) == preparedVersions else {
                                self.statusMessage = "Preview inputs changed during execution. Preview again."; return
                            }
                            self.previewModelRequest = capturedRequest
                            self.previews = result["previews"] as? [String: [String: Any]] ?? [:]
                            self.previewCachePath = artifacts.appendingPathComponent("activation-cache.pt").path
                            self.previewSignature = signature; self.previewSourceHash = result["sourceHash"] as? String
                            self.previewCheckpoint = plan.hasCheckpoint ? artifacts.appendingPathComponent("weights/checkpoint.pt").path : nil
                            self.capturePreviewArtifactHashes()
                            self.previewArtifactHashes.merge(preparedVersions) { _, value in value }
                            self.previewExecuted = result["executedNodeIDs"] as? [String] ?? []
                            self.previewReused = result["reusedNodeIDs"] as? [String] ?? []
                            self.previewWeightLabel = result["weights"] as? String == "checkpoint" ? "Inference · trained checkpoint" : "Inference · initialized weights"
                            self.resolvedDevice = result["device"] as? String ?? self.resolvedDevice
                            let count = (result["inputProvenance"] as? [String: Any])?["count"] as? Int ?? 0
                            let rows = settings.preparedTraining == nil ? "" : " · \(count) frozen validation rows (maximum 8) · inspect row IDs in Data inspector"
                            self.statusMessage = "Preview ready · \(self.previewExecuted.count) computed · \(self.previewReused.count) cached · inference only; no metric estimate.\(rows)"
                        })
                    if self.previewRequestToken == token, self.isPreviewing {
                        self.previewWorker = handle; self.previewLaunchTask = nil
                        if self.previewCancelled || Task.isCancelled { handle.cancel() }
                    }
                } catch {
                    guard self.previewRequestToken == token else { return }
                    self.previewRequestToken = nil; self.isPreviewing = false; self.previewWorker = nil; self.previewLaunchTask = nil
                    guard contextCurrent() else { return }
                    self.statusMessage = self.previewCancelled || error is CancellationError ? "Inference preview cancelled." : "Preview failed: \(error.localizedDescription)"
                }
            }
        } catch { statusMessage = "Preview failed: \(error.localizedDescription)" }
    }
    func cancelPreviewExecution() {
        guard isPreviewing else { return }
        previewCancelled = true; previewLaunchTask?.cancel(); previewWorker?.cancel()
    }
    /// Cheap UI invalidation only; every worker launch re-reads and cryptographically verifies all prepared artifacts.
    private func previewFileIdentity(_ path: String) -> String? {
        guard let values=try? URL(fileURLWithPath:path).resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey,.contentModificationDateKey,.fileResourceIdentifierKey]),values.isRegularFile==true,values.isSymbolicLink != true,let size=values.fileSize,let modified=values.contentModificationDate else{return nil}
        return "metadata:\(size):\(modified.timeIntervalSince1970):\(String(describing:values.fileResourceIdentifier))"
    }
    private func preparedPreviewVersions(settings: ExecutionSettings) throws -> [String:String] {
        guard let selection=settings.preparedTraining else{return [:]}
        let manifest=try verifiedPreparationManifest(selection)
        guard let artifacts=manifest["artifacts"] as? [[String:Any]],artifacts.count<=300 else{throw previewInputError("Prepared artifact inventory is invalid.")}
        var result=[selection.manifestPath:selection.manifestSHA256]
        let root=URL(fileURLWithPath:selection.manifestPath).deletingLastPathComponent()
        for artifact in artifacts {
            guard let name=artifact["path"] as? String,!name.isEmpty,name != ".",name != "..",URL(fileURLWithPath:name).lastPathComponent==name,!name.contains("/"),let size=artifact["bytes"] as? Int,size>=0,size<=50_000_000 else{throw previewInputError("Prepared artifact path or size is invalid.")}
            let path=root.appendingPathComponent(name).path
            guard let identity=previewFileIdentity(path) else{throw previewInputError("Prepared artifact is missing or not a regular file.")}
            result[path]=identity
        }
        return result
    }
    private func previewInputError(_ message:String)->NSError {NSError(domain:"KwipML.Preview",code:1,userInfo:[NSLocalizedDescriptionKey:message])}

}
