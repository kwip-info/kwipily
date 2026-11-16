import AppKit
import Foundation
import KwipMLCore

extension WorkspaceStore {
    /// Invalidate UI identity before cancelling: even synchronous exit cannot publish old code.
    func invalidateDraftCode() {
        codeGenerationToken = UUID()
        codeGenerationTask?.cancel(); codeGenerationTask = nil
        let previous = codeGenerationWorker; codeGenerationWorker = nil
        isGeneratingCode = false; draftCode = ""; draftCodeFiles = [:]; draftCodeError = nil
        codeSignature = nil; codeContentIdentity = nil; codeSourceContext = nil
        codeRuntimeEvidence = nil; codePackageMessage = nil
        previous?.cancel()
    }
    func refreshDraftCode() {
        guard allowOrdinaryExecution() else { draftCodeError = statusMessage; return }
        let snapshot: CodeGenerationSnapshot
        do { snapshot = try codeGenerationSnapshot() }
        catch { invalidateDraftCode(); draftCodeError = error.localizedDescription; return }
        let signature = snapshot.signature
        let contentIdentity = snapshot.identity
        if codeSignature == signature && codeContentIdentity == contentIdentity && codeSourceContext == codeGenerationContextID {
            if isGeneratingCode { return }
            if !draftCode.isEmpty, let runtime, let evidence = codeRuntimeEvidence,
               (try? evidence.matchesCurrent(runtime)) == true { return }
        }
        invalidateDraftCode()
        guard let runtime else { draftCodeError = runtimeError ?? "Runtime unavailable."; return }
        if let model = snapshot.model, let error = capabilityBoundaryError(for: model, context: workflowSession == nil ? .bundledDataset : .preparedWorkflow) { draftCodeError = error; return }
        // Unsaved legacy drafts become real managed projects before acquiring operation ownership.
        if activeProjectID == nil {
            guard saveManagedProject() else { draftCodeError = saveError ?? "Save this project before generating source."; return }
            // Saving can migrate a legacy draft. Capture the resulting canonical graph,
            // rather than launching source whose identity is already stale.
            refreshDraftCode()
            return
        }
        guard let projectID = activeProjectID else { draftCodeError = "Open a project before generating source."; return }
        let context = codeGenerationContextID
        let token = UUID(); codeGenerationToken = token
        do {
            let policy = try executionPolicy(for: projectID, capabilityMaximumSeconds: 30)
            let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let bytes = snapshot.bytes
            let runtimeEvidence = try WorkflowRuntimeEvidence.capture(runtime)
            isGeneratingCode = true; codeSignature = signature; codeContentIdentity = contentIdentity; codeSourceContext = context
            var exported: String?, files: [String: String] = [:], failure: String?
            codeGenerationTask = Task { [self] in
              defer { if codeGenerationToken == token { codeGenerationTask = nil } }
              do {
                try Task.checkCancellation()
                let handle = try await WorkflowToolExecution.launchAsync(workspace: workspaceDirectory, accountingRoot: applicationDirectory, projectID: projectID,
                runtime: runtime, window: window, toolMode: .codeExport, inputs: [snapshot.inputName: bytes],
                maximumSeconds: 30, storageLimitBytes: policy.storageLimitBytes, launcher: codeWorkerLauncher, onLine: { [weak self] line in
                    guard let self, self.codeGenerationToken == token,
                          let data = line.data(using: .utf8), let event = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return }
                    if event["event"] as? String == "codeExport" {
                        exported = event["code"] as? String
                        files = (event["files"] as? [String: String]) ?? [:]
                    }
                    if event["event"] as? String == "error" { failure = event["message"] as? String }
                }, onExit: { [weak self] code, stderr in
                    guard let self, self.codeGenerationToken == token else { return }
                    self.codeGenerationWorker = nil; self.isGeneratingCode = false
                    guard self.activeProjectID == projectID, self.codeGenerationContextID == context,
                          self.currentCodeContentIdentity == contentIdentity, let current = try? self.codeGenerationSnapshot(),
                          current.signature == signature else {
                        self.codeSignature = nil; self.codeContentIdentity = nil; self.codeSourceContext = nil; return
                    }
                    guard code == 0, let exported, !exported.isEmpty else {
                        self.codeSignature = nil; self.codeContentIdentity = nil; self.codeSourceContext = nil
                        if code == 124 { self.draftCodeError = "Code generation exceeded its time allowance. Retry or reduce the model’s size." }
                        else if code == 130 { self.draftCodeError = "Code generation was cancelled. Generate again when ready." }
                        else { self.draftCodeError = failure ?? (stderr.isEmpty ? "The source generator did not produce code." : stderr) }
                        return
                    }
                    self.draftCode = exported; self.draftCodeFiles = files.isEmpty ? ["model.py": exported] : files; self.codeRuntimeEvidence = runtimeEvidence; self.draftCodeError = nil
                })
                if codeGenerationToken != token { handle.cancel() }
                else if isGeneratingCode { codeGenerationWorker = handle }
              } catch { failDraftCode(error, token: token) }
            }
        } catch { failDraftCode(error, token: token) }
    }
    private func failDraftCode(_ error: Error, token: UUID) {
        guard codeGenerationToken == token else { return }
        isGeneratingCode = false; codeGenerationWorker = nil
        codeSignature = nil; codeContentIdentity = nil; codeSourceContext = nil; draftCodeError = error.localizedDescription
    }
    func copyPythonCode(_ source:String) {
        guard allowOrdinaryExecution() else { return }
        guard !source.isEmpty else{return}
        NSPasteboard.general.clearContents();NSPasteboard.general.setString(source,forType:.string)
        statusMessage="Python source copied."
    }
    func savePythonCode(_ source:String, filename: String = "model.py") {
        guard allowOrdinaryExecution() else { return }
        guard !source.isEmpty else{return}
        let panel=NSSavePanel();panel.title="Export source file";panel.nameFieldStringValue=URL(fileURLWithPath: filename).lastPathComponent
        guard panel.runModal() == .OK,let url=panel.url else{return}
        do {try source.write(to:url,atomically:true,encoding:.utf8);statusMessage="Exported \(url.lastPathComponent)."}
        catch {statusMessage="Code export failed: \(error.localizedDescription)"}
    }
    func exportWorkflowCodePackage() {
        guard allowOrdinaryExecution(), currentCodeReady, draftCodeFiles["workflow.py"] != nil, let runtime, let evidence = codeRuntimeEvidence else { return }
        let panel = WorkflowExportParentChooser.panel(title: "Choose a folder for the Code package", message: "Create a new source package inside this folder.")
        guard panel.runModal() == .OK, let parent = panel.url else { return }
        let accessingParent = parent.startAccessingSecurityScopedResource()
        let destination = parent.appendingPathComponent("\(ProductBrand.name) Code " + UUID().uuidString.prefix(8))
        let files = draftCodeFiles, identity = currentCodeContentIdentity, context = codeGenerationContextID
        codePackageMessage = "Exporting source package…"
        Task {
            defer { if accessingParent { parent.stopAccessingSecurityScopedResource() } }
            do {
                let contents = try await Task.detached(priority: .userInitiated) {
                    try WorkflowCodePackage.contents(files: files, runtime: runtime, expectedEvidence: evidence)
                }.value
                guard currentCodeContentIdentity == identity, codeGenerationContextID == context, currentCodeReady else {
                    throw WorkflowArchiveError("The workflow changed before export. Generate its current source again.")
                }
                try await Task.detached(priority: .userInitiated) { try WorkflowCodePackage.write(contents, to: destination) }.value
                codePackageMessage = "Exported source package. Install its declared dependencies before running."
                NSWorkspace.shared.activateFileViewerSelecting([destination])
            } catch { codePackageMessage = "Code package export failed: \(error.localizedDescription)" }
        }
    }
}
