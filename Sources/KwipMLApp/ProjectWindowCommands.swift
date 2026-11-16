import SwiftUI
import AppKit

struct ProjectWindowCommands: Commands {
    @ObservedObject var registry: ProjectWindowRegistry
    @ObservedObject private var updates=DirectUpdateStore.shared
    @ObservedObject var launcher: ProjectWindowSession
    /// Resolve the exact owner of a transient dropdown; never borrow a last-active file.
    private var commandWindow: NSWindow? { AttachedDropdownFocus.commandOwner(for: NSApp.keyWindow) }
    private var active: ProjectWindowSession? {
        guard commandWindow?.windowController is ProjectWindowController else { return nil }
        return registry.session(for: commandWindow)
    }
    private var store: WorkspaceStore { (active ?? launcher).workspace }
    private var canSave: Bool {
        if commandWindow?.windowController is FlowCanvasWindowController { return true }
        return active != nil && store.lessonContext?.restoreState != .failed
    }
    private var canRun: Bool {
        guard let active else { return false }
        guard !isHistoricalConstruction(active) else {return false}
        if active.tensor.isActive { return active.tensor.canSubmit }
        if active.normalization.isActive { return active.normalization.canSubmit }
        if active.affine.isActive { return active.affine.canSubmit }
        return store.canRunCurrentFlow
    }
    var body: some Commands {
        CommandGroup(after:.appInfo) {
            Button("Check for Updates…") {updates.check()}.disabled(!updates.canCheck)
        }
        CommandGroup(replacing: .appSettings) {
            Button("Settings…") { ApplicationNavigationWindows.shared.showSettings() }
                .keyboardShortcut(",", modifiers: .command)
        }
        CommandGroup(replacing: .newItem) {
            Button("New flow") { registry.createNewFromMenu() }
                .keyboardShortcut(store.shortcut(for: .newProject).keyboardShortcut)
            Button("Start from template…") { ApplicationNavigationWindows.shared.showTemplates() }
            Button("Open project…") { registry.openFromMenu() }
                .keyboardShortcut(store.shortcut(for: .openProject).keyboardShortcut)
            Button("Open Block file…") { FlowDocumentCommands.open() }
            Menu("Open recent") {
                ForEach(RecentProjectGroup.make(Array(launcher.workspace.recentProjects.prefix(20)))) { group in
                    if group.entries.count==1,let entry=group.entries.first {
                        Button(entry.title) {launcher.workspace.openRecentProject(entry)}.help(entry.url.path)
                    } else {
                        Menu(group.title+" · "+String(group.entries.count)+" projects") {
                            ForEach(group.entries) {entry in
                                Button(entry.detail) {launcher.workspace.openRecentProject(entry)}.help(entry.url.path)
                            }
                        }
                    }
                }
                if launcher.workspace.recentProjects.isEmpty {Text("No recent projects")}

            }
            Button("Show welcome") { registry.showLibrary(section: "Projects") }
                .keyboardShortcut(store.shortcut(for: .projectLibrary).keyboardShortcut)
            Divider()
            Button("Save") { save() }
                .keyboardShortcut(store.shortcut(for: .saveProject).keyboardShortcut).disabled(!canSave)
            Menu("Export") {
                Button(store.workflowSession == nil ? "Model graph…":"Workflow source…") { active?.workspace.saveProject() }
                    .keyboardShortcut(store.shortcut(for: .exportProject).keyboardShortcut)
                    .disabled(active == nil || store.isLearningLessonActive)
                Button("Flow documents…") { if let active { FlowDocumentCommands.export(workspace: active.workspace) } }
                    .disabled(active == nil || store.workflowSession == nil || store.isBusy || store.isLearningLessonActive)
            }
            Button("Convert to file-backed flow") { _ = active?.workspace.convertActiveWorkflowToFiles() }
                .disabled(active == nil || store.workflowSession == nil || store.workflowSession?.owner.flowRootURL != nil || store.isBusy || store.isLearningLessonActive)
        }
        EditorHistoryCommands()
#if KWIPML_RECOVERY_UAT
        RecoveryUATCommands()
#endif
        CommandGroup(after: .toolbar) {
            Divider()
            Button("Build canvas") { store.activeSection = "Build" }.disabled(active == nil || store.isLearningLessonActive)
            Button("History") { store.activeSection = store.workflowSession == nil ? "Run" : "History" }.disabled(active == nil || store.isLearningLessonActive)
            Button("Generated code") { store.activeSection = "Code" }.disabled(active == nil || store.isLearningLessonActive)
            Button("Configure") { store.activeSection = "Settings" }.disabled(active == nil || store.isLearningLessonActive)
            Divider()
            Button("Toggle components") { store.showLibrary.toggle() }
                .keyboardShortcut(store.shortcut(for: .toggleComponents).keyboardShortcut).disabled(active == nil)
            Button("Toggle configuration") { store.showInspector.toggle() }
                .keyboardShortcut(store.shortcut(for: .toggleInspector).keyboardShortcut).disabled(active == nil)
            Button("Toggle results") { store.showPreview.toggle() }
                .keyboardShortcut(store.shortcut(for: .toggleResults).keyboardShortcut).disabled(active == nil || store.isLearningLessonActive)
        }
        CommandMenu("Learn") {
            Button("Courses & examples…") { ApplicationNavigationWindows.shared.showLearn() }
                .keyboardShortcut(store.shortcut(for: .learningLibrary).keyboardShortcut)
            Divider()
            Button("Open lesson canvas") {
                if !registry.focusLessonCanvas(preferred:active) {ApplicationNavigationWindows.shared.showLearn(search:"")}
            }

        }
        CommandMenu("Run") {
            Button(store.isLearningLessonActive ? "Check construction" : "Run flow") { run() }
                .keyboardShortcut(store.shortcut(for: .runFlow).keyboardShortcut).disabled(!canRun)
            Button("Cancel active operation") {
                if let host = active?.workspace.workflowSession { host.cancel() }
                else { active?.workspace.cancelRun() }
            }.disabled(active == nil || !store.isBusy)
            Divider()
            RunFeedbackMenu(store: launcher.workspace)
        }
    }
    private func save() {
        if FlowCanvasWindowRegistry.shared.save(window: commandWindow) { return }
        _ = registry.save(window: commandWindow)
    }
    private func isHistoricalConstruction(_ session:ProjectWindowSession)->Bool {
        session.workspace.lessonContext?.kind == .normalization ||
        (session.workspace.lessonContext?.kind == .tensor && RetiredTrainingLessonDestination.contains(session.tensor.draft?.operation))
    }
    private func run() {
        guard let active, !isHistoricalConstruction(active) else { return }
        if active.tensor.isActive { active.tensor.submit() }
        else if active.normalization.isActive { active.normalization.submit() }
        else if active.affine.isActive { active.affine.submit() }
        else { active.workspace.run() }
    }
}
