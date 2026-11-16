import AppKit
import Foundation
import KwipMLCore

extension DataFlowWorkspaceStore {
    func chooseSource(ifStillCurrent: () -> Bool = { true }) {
        guard requirePreparationOperation() else {return}
        guard ifStillCurrent(), !isBusy, let id = selectedNodeID, project.graph.nodes.contains(where: { $0.id == id && $0.operatorID == "kwip.data.sourceCSV" }) else { return }
        let panel = NSOpenPanel(); panel.title = "Choose Table Input file"; panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url, ifStillCurrent(), !isBusy, selectedNodeID == id, requirePreparationOperation() else { return }
        do {
            try workspace.sourceAccess.remember(url)
            bindSource(url.path, nodeID: id); statusMessage = "Source linked on this Mac. Map its numeric columns before preparing."
            inspectSource(id: id)
        } catch { statusMessage = error.localizedDescription }
    }
    func runPreparation() {
        guard workspace.allowExecution() else {statusMessage=workspace.statusMessage;return}
        guard requirePreparationOperation() else {return}
        guard !hasPendingNumericEdits else {
            statusMessage = "Apply or correct pending numeric edits before preparing data."
            return
        }
        guard projectID == workspace.activeProjectID else { statusMessage = "Return to this preparation's project before running it."; return }
        if isInspectingSource {
            prepareAfterInspection = true; queuedInspectionSourceID = nil
            cancelSourceInspection(cancelQueuedWork: false)
            statusMessage = "Preparation will start after source inspection stops."
            return
        }
        guard !isBusy, !workspace.isBusy else { statusMessage = "Wait for the current worker to finish."; return }
        guard let runtime = workspace.runtime, operators != nil else { statusMessage = "A runtime with data operators is required."; return }
        guard validation.isValid else { statusMessage = validation.issues.map(\.message).joined(separator: " · "); return }
        do {
            guard let projectID else { throw DataFlowContractError("Save the project before preparing data.") }
            let snapshotGraph = project.graph, sourceBindings = project.sourceBindings, nodeLabels = requestNodeLabels
            let componentTitles = Dictionary(uniqueKeysWithValues: (operators?.operators ?? []).map { ($0.id, $0.title) })
            let cachePolicy = self.cachePolicy, cacheReference = lastCacheReference
            let accountingRoot = workspace.applicationDirectory
            let storage = workspace.workspaceDirectory, permissions = workspace.sourceAccess.directory
            let policy = try workspace.executionPolicy(for: projectID)
            let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let token = UUID()
            let contextCurrent: () -> Bool = { [weak self] in
                guard let self else { return false }
                return self.projectID == self.workspace.activeProjectID && self.project.graph == snapshotGraph && self.project.sourceBindings == sourceBindings
            }
            preparationToken = token; cancelled = false; isPreparing = true; workspace.isPreparingData = true
            workspace.cancelDataPreparation = { [weak self] in self?.cancelPreparation() }
            statusMessage = "Capturing preparation inputs…"
            preparationLaunchTask = Task { [self] in
                do {
                    let capture = Task.detached(priority: .userInitiated) {
                        try PreparationLaunchPlan.capture(graph: snapshotGraph, bindings: sourceBindings, labels: nodeLabels,
                            cachePolicy: cachePolicy, cacheReference: cacheReference, workspace: storage, accountingRoot: accountingRoot,
                            sourcePermissions: permissions, projectID: projectID, storageLimitBytes: policy.storageLimitBytes, preparationCacheLimitBytes: policy.preparationCacheLimitBytes, workload: policy.budget.resources?.workloadPolicy, checkCancellation: {
                                try Task.checkCancellation()
                                _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                            })
                    }
                    let plan = try await withTaskCancellationHandler(operation: { try await capture.value }, onCancel: { capture.cancel() })
                    defer { plan.removeSourceStage() }
                    try Task.checkCancellation()
                    guard contextCurrent(), self.preparationToken == token else { throw CancellationError() }
                    var stagedPreviews: [String: [String: Any]] = [:], previewBytes = 0
                    var stagedCache: DataFlowCacheReport?, failure: String?, failureDetail: DataFlowFailureDetail?
                    var completed = false, invalidEvents = false
                    let nodeIDs = Set(snapshotGraph.nodes.map(\.id))
                    let handle = try await WorkflowPreparationExecution.launch(workspace: storage, projectID: projectID,
                        runtime: runtime, window: window, storageAdmission: plan.storageReservation, inputs: plan.inputs, arguments: plan.arguments,
                        launcher: self.preparationWorkerLauncher, onLine: { [weak self] line in
                            guard let self, self.preparationToken == token, !self.cancelled, contextCurrent(),
                                  let data = line.data(using: .utf8), data.count <= 1_000_000,
                                  let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
                            switch event["event"] as? String {
                            case "dataflowStarted": self.statusMessage = "Executing preparation graph…"
                            case "dataflowPreview":
                                previewBytes += data.count
                                guard previewBytes <= 32_000_000, let id = event["nodeID"] as? String,
                                      nodeIDs.contains(id), stagedPreviews[id] == nil,
                                      let outputs = event["outputs"] as? [String: Any] else { invalidEvents = true; return }
                                stagedPreviews[id] = outputs; self.statusMessage = "Prepared \(nodeLabels[id] ?? id)"
                            case "dataflowCompleted":
                                completed = true
                                if let value = event["cache"], let bytes = try? JSONSerialization.data(withJSONObject: value) {
                                    stagedCache = try? JSONDecoder().decode(DataFlowCacheReport.self, from: bytes)
                                }
                            case "error":
                                failure = (event["message"] as? String).map { String($0.prefix(2000)) }
                                if let object = event["dataflowError"], let bytes = try? JSONSerialization.data(withJSONObject: object) {
                                    failureDetail = try? DataFlowFailureDetail.decode(bytes, graph: snapshotGraph, labels: nodeLabels, titles: componentTitles)
                                }
                            default: break
                            }
                        }, onExit: { [self] code, stderr, artifacts in
                            guard self.preparationToken == token else { return }
                            self.finishPreparationLifetime()
                            guard contextCurrent() else { return }
                            guard !self.cancelled, code != 130 else { self.statusMessage = "Preparation cancelled; previous results retained."; return }
                            if code == 124 { self.statusMessage = "Preparation exceeded its time allowance. Reduce the input size or adjust the execution budget."; return }
                            guard code == 0, completed, !invalidEvents, Set(stagedPreviews.keys) == nodeIDs, let artifacts else {
                                self.preparationFailure = code != 0 ? failureDetail.map { detail in
                                    DataFlowFailure(detail: detail, graph: snapshotGraph, labels: nodeLabels, sourceBindings: sourceBindings, sourceDigests: plan.sourceDigests)
                                } : nil
                                self.statusMessage = failure ?? (stderr.isEmpty ? "Preparation did not publish a verified result." : String(stderr.prefix(500))); return
                            }
                            let manifest = artifacts.appendingPathComponent("manifest.json")
                            let previousReport = self.cacheReport, previousReference = self.lastCacheReference
                            do {
                                try self.acceptCacheReport(manifestURL: manifest, report: stagedCache)
                                try self.persistCompletedResult(manifest: manifest, graph: snapshotGraph, bindings: sourceBindings, digests: plan.sourceDigests)
                            } catch {
                                self.cacheReport = previousReport; self.lastCacheReference = previousReference; self.persistCachePreferences()
                                self.statusMessage = error.localizedDescription; return
                            }
                            self.preparationFailure = nil; self.restoredResult = false
                            self.previews = stagedPreviews; self.resultGraph = snapshotGraph; self.resultBindings = sourceBindings
                            self.resultSourceDigests = plan.sourceDigests; self.manifestURL = manifest
                            self.statusMessage = "Preparation complete · " + self.cacheSummary
                        })
                    if self.preparationToken == token, self.isPreparing {
                        self.preparationWorker = handle; self.preparationLaunchTask = nil
                        if self.cancelled || Task.isCancelled { handle.cancel() }
                    }
                } catch {
                    guard self.preparationToken == token else { return }
                    self.finishPreparationLifetime()
                    guard contextCurrent() else { return }
                    self.statusMessage = self.cancelled || error is CancellationError ? "Preparation cancelled; previous results retained." : error.localizedDescription
                }
            }
        } catch { statusMessage = error.localizedDescription }
    }
    private func finishPreparationLifetime() {
        preparationToken = nil; preparationLaunchTask = nil; preparationWorker = nil
        isPreparing = false; workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
    }
    func cancelPreparation() {
        guard requirePreparationOperation() else { return }
        if isRestoringResults { cancelResultRestoration(); return }
        if isInspectingSource { cancelSourceInspection(); return }
        guard isPreparing else { return }
        cancelled = true; preparationLaunchTask?.cancel(); preparationWorker?.cancel()
        statusMessage = "Cancelling preparation…"
    }
    func exportPortableFlow() {
        guard requirePreparationOperation() else {return}
        let panel = NSSavePanel(); panel.nameFieldStringValue = "preparation.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try DataFlowCodec.encodeGraph(project.graph).write(to: url, options: .atomic); statusMessage = "Portable graph exported; source permissions remain on this Mac." }
        catch { statusMessage = error.localizedDescription }
    }
}
