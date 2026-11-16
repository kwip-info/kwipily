import AppKit
import UniformTypeIdentifiers
import KwipMLCore

@MainActor final class FlowApplicationDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification:Notification) { DirectUpdateStore.shared.start() }
    func applicationWillTerminate(_ notification:Notification){MCPController.stopAll()}
    @objc func newWindowForTab(_ sender: Any?) { ProjectWindowRegistry.shared.createNewFromMenu() }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard ExpandedConfigurationWindowController.prepareToTerminate()else{return .terminateCancel}
        let prepare = { ProjectWindowRegistry.shared.prepareToTerminate() && FlowCanvasWindowRegistry.shared.prepareToTerminate() }
        do {
            let accepted = try ProjectWindowRegistry.shared.restoration?.captureAfterPreparation(prepare) ?? prepare()
            return accepted ? .terminateNow : .terminateCancel
        } catch {
            let alert = NSAlert(); alert.messageText = "Window layout could not be saved"
            alert.informativeText = error.localizedDescription + " Your previous saved session is retained. Project save decisions have already completed."
            alert.addButton(withTitle: "Quit keeping saved session"); alert.addButton(withTitle: "Cancel")
            return alert.runModal() == .alertFirstButtonReturn ? .terminateNow : .terminateCancel
        }
    }
}

@MainActor enum FlowDocumentCommands {
    static func open() {
        let panel = NSOpenPanel()
        panel.title = "Open subflow document"
        panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "kwipflow", conformingTo: .json) ?? .data]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try ProjectWindowRegistry.shared.openStandaloneSubflow(url) }
        catch { show(error) }
    }
    static func export(workspace: WorkspaceStore) {
        guard !workspace.isBusy, !workspace.isLearningLessonActive, let session = workspace.workflowSession else { return }
        do {
            let plan = try FlowDocumentMigration.plan(session.owner.document)
            let panel = NSSavePanel()
            panel.title = "Export independent flow documents"
            panel.message = "Creates Main.kwipflow and separate subflow files. Your current project stays unchanged. Open a subflow file to edit it independently."
            panel.nameFieldStringValue = "Flow Documents"
            panel.canCreateDirectories = true
            guard panel.runModal() == .OK, let destination = panel.url else { return }
            let root = try FlowDocumentStore.commitMigration(plan, to: destination)
            NSWorkspace.shared.activateFileViewerSelecting([root])
            workspace.statusMessage = "Exported independent flow documents. The current project still uses its existing definitions."
        } catch { workspace.statusMessage = error.localizedDescription; show(error) }
    }
    private static func show(_ error: Error) {
        let alert = NSAlert(); alert.messageText = "Flow document could not be opened or saved"
        alert.informativeText = error.localizedDescription; alert.addButton(withTitle: "OK"); alert.runModal()
    }
}


extension WorkspaceStore {
    @discardableResult func convertActiveWorkflowToFiles() -> Bool {
        guard !isBusy, !isLearningLessonActive, let current = workflowSession else { return false }
        if current.owner.flowRootURL != nil { return true }
        guard saveManagedProject(), flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition() else { return false }
        do {
            let record = try projectLibrary.activateFileBackedWorkflow(id: current.owner.projectID)
            guard beginWorkflowProject(id: record.id, forceReload: true) else { return false }
            activeProjectID = record.id; projectURL = URL(fileURLWithPath: record.path)
            refreshProjects(); persistWorkspaceView()
            statusMessage = "This project now saves to Main.kwipflow. Subflows open as independent documents."
            return true
        } catch { statusMessage = "Conversion failed: " + error.localizedDescription; return false }
    }
}
