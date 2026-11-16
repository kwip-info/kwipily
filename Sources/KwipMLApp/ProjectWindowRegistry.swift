import AppKit
import SwiftUI
import Combine
import KwipMLCore

/// Maps a managed project identity to exactly one root editor, independent of tab grouping.
@MainActor final class ProjectWindowRegistry: ObservableObject {
    static let shared = ProjectWindowRegistry()
    private let makeWorkspace: @MainActor () -> WorkspaceStore
    private let makeExternalWorkspace: @MainActor (ExternalProjectLocation) -> WorkspaceStore
    private var externalCatalog: ExternalFlowCatalog?
    private let presentsWindows: Bool
    private var preferencesObservations: [ObjectIdentifier: AnyCancellable] = [:]
    private var propagatingPreferences = false
    private var importedFiles: [URL: String] = [:]
    private var projects: [String: ProjectWindowController] = [:]
    var reopenLibrary: ((String) -> Void)?
    private weak var launcherWindow: NSWindow?
    private weak var launcher: ProjectWindowSession?
    @Published var restoration: CanvasTabSessionCoordinator?
    var restorationCatalog: ExternalFlowCatalog { flowCatalog() }
    var restorationPresentsWindows: Bool { presentsWindows }
    var restorationWorkspace: WorkspaceStore { launcher?.workspace ?? makeWorkspace() }
    var restorationLauncherWindow: NSWindow? { launcherWindow }
    @Published private(set) var activeSession: ProjectWindowSession?
    @Published private(set) var navigationError:String?
    private var observers: [NSObjectProtocol] = []
    private var activeObservation: AnyCancellable?
    var hasActiveOperations:Bool {
        if ScheduledRunAdmission.hasOwner {return true}
        let sessions=projects.values.map(\.session)+(launcher.map{[$0]} ?? [])
        return sessions.contains{$0.workspace.isBusy || $0.preparation.current?.isPreparing == true}
    }
    var projectCount: Int { projects.count }
    var openControllers: [ProjectWindowController] { Array(projects.values) }
    init(presentsWindows: Bool = true, makeWorkspace: @escaping @MainActor () -> WorkspaceStore = { WorkspaceStore() },
         externalCatalog: ExternalFlowCatalog? = nil,
         makeExternalWorkspace: @escaping @MainActor (ExternalProjectLocation) -> WorkspaceStore = { WorkspaceStore(externalLocation: $0) }) {
        self.presentsWindows = presentsWindows
        self.makeWorkspace = makeWorkspace
        self.externalCatalog = externalCatalog
        self.makeExternalWorkspace = makeExternalWorkspace
        observers.append(NotificationCenter.default.addObserver(forName: NSWindow.didBecomeKeyNotification, object: nil, queue: .main) { [weak self] notification in
            MainActor.assumeIsolated { self?.select(window: notification.object as? NSWindow) }
        })
        observers.append(NotificationCenter.default.addObserver(forName: NSWindow.didResignKeyNotification, object: nil, queue: .main) { [weak self] notification in
            MainActor.assumeIsolated {
                guard let self, let window = notification.object as? NSWindow,
                      let resigningSession = self.session(for: window), self.activeSession === resigningSession else { return }
                // A child panel may resign after its document has regained focus.
                // Only the actual active owner can clear its own command routing.
                self.select(window: nil)
            }
        })
    }
    deinit { for observer in observers { NotificationCenter.default.removeObserver(observer) } }
    func session(for window: NSWindow?) -> ProjectWindowSession? {
        guard let window else { return nil }
        if window === launcherWindow { return launcher }
        return projects.values.first { $0.window === window }?.session
    }
    func select(window: NSWindow?) {
        let next = session(for: AttachedDropdownFocus.commandOwner(for: window))
        // SwiftUI attachment updates may select an already-key launcher again.
        // Publishing an unchanged owner feeds back into the observing launcher scene.
        guard activeSession !== next else { return }
        activeObservation = nil
        activeSession = next
        activeObservation = next?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    func registerLauncher(_ window: NSWindow, session: ProjectWindowSession) {
        guard launcherWindow !== window || launcher !== session else { return }
        launcherWindow = window; launcher = session
        installRouting(session)
        window.tabGroup?.removeWindow(window)
        window.tabbingIdentifier = "info.kwip.KwipML.welcome"
        window.tabbingMode = .disallowed
        if window.isKeyWindow { select(window: window) }
        let needsLaunchRestore = restoration == nil
        if needsLaunchRestore { restoration = makeRestorationCoordinator() }
        if needsLaunchRestore { Task { @MainActor [weak self, weak session] in
            guard let self, let session else { return }
            self.restoration?.restoreOnce(enabled: session.workspace.appPreferences.restoreCanvasSessionAtLaunch != false)
            self.dismissWelcomeAfterRestoration()
        } }
    }
    func showLibrary(section: String) {
        if section == "Examples", presentsWindows { ApplicationNavigationWindows.shared.showLearn(); return }
        launcher?.workspace.refreshProjects(); launcher?.workspace.activeSection = section
        if let reopenLibrary { reopenLibrary(section); return }
        launcherWindow?.makeKeyAndOrderFront(nil)
    }
    private func installRouting(_ session: ProjectWindowSession) {
        let workspace = session.workspace
        workspace.projectCatalogChanged = { [weak self,weak workspace] in
            guard let self else{return}
            for peer in self.projects.values where peer.session.workspace !== workspace {peer.session.workspace.refreshProjects()}
            if self.launcher?.workspace !== workspace {self.launcher?.workspace.refreshProjects()}
        }
        session.affine.openProjectBrowser = { [weak self] in self?.showLibrary(section: "Projects") }
        session.normalization.openProjectBrowser = { [weak self] in self?.showLibrary(section: "Projects") }
        workspace.focusOwningProject = { [weak self, weak workspace] in
            guard let self, let workspace,
                  let controller = self.projects.values.first(where: { $0.session.workspace === workspace }),
                  let window = controller.window else { return }
            // Focus the existing owner in its current group; never move or recreate a project.
            window.tabGroup?.selectedWindow = window
            self.select(window: window)
            if self.presentsWindows { window.makeKeyAndOrderFront(nil) }
        }
        preferencesObservations[ObjectIdentifier(session)] = workspace.$appPreferences.dropFirst().sink { [weak self] preferences in
            guard let self, !self.propagatingPreferences else { return }
            self.propagatingPreferences = true
            defer { self.propagatingPreferences = false }
            let sessions = self.projects.values.map(\.session) + (self.launcher.map { [$0] } ?? [])
            for peer in sessions where peer.workspace !== workspace && peer.workspace.appPreferences != preferences {
                peer.workspace.appPreferences = preferences
            }
        }
        workspace.exampleInWindow = { [weak self, weak workspace] device, kind in
            do { _ = try self?.launch { session in
                guard session.workspace.loadPreparedWorkflowExample(device:device,kind:kind) else {return false}
                session.workspace.inheritLesson(from:workspace);return true
            }; return self != nil }
            catch { workspace?.statusMessage = error.localizedDescription; return false }
        }
        session.tensor.startInWindow = { [weak self, weak workspace] operation, mode in
            do { _ = try self?.launch { session in
                guard session.tensor.start(operation:operation,mode:mode) else {return false}
                session.workspace.inheritLesson(from:workspace);return true
            }; return self != nil }
            catch { workspace?.statusMessage = error.localizedDescription; return false }
        }
        session.affine.startInWindow = { [weak self, weak workspace] identity, mode in
            do { _ = try self?.launch { session in
                session.affine.start(identity:identity,mode:mode)
                guard session.affine.isActive else {return false}
                session.workspace.inheritLesson(from:workspace);return true
            } }
            catch { workspace?.statusMessage = error.localizedDescription;self?.show(error) }
        }
        session.normalization.startInWindow = { [weak self, weak workspace] mode in
            do { _ = try self?.launch { session in
                session.normalization.start(mode:mode)
                guard session.normalization.isActive else {return false}
                session.workspace.inheritLesson(from:workspace);return true
            } }
            catch { workspace?.statusMessage = error.localizedDescription;self?.show(error) }
        }
        workspace.openExternalProjectInWindow = { [weak self, weak workspace] record in
            do { _ = try self?.openExternal(record) }
            catch { workspace?.statusMessage = error.localizedDescription; self?.show(error) }
        }
        workspace.renewExternalProjectAccess = { [weak self] record in self?.chooseFlowFolder(for: record.rootURL) }
        workspace.locateExternalProject = { [weak self, weak workspace] record in
            guard let workspace else { return }; self?.locateExternalFlow(record, workspace: workspace)
        }
        workspace.openProjectInWindow = { [weak self, weak workspace] record, section in
            do { _ = try self?.open(record, section: section) }
            catch { workspace?.statusMessage = error.localizedDescription;self?.show(error) }
        }
        workspace.newProjectInWindow = { [weak self, weak workspace] in
            do { _ = try self?.newFlow() } catch { workspace?.statusMessage = error.localizedDescription;self?.show(error) }
        }
        workspace.openFileInWindow = { [weak self, weak workspace] url in
            do { _ = try self?.openSelectedFile(url) } catch { workspace?.statusMessage = error.localizedDescription;self?.show(error) }
        }
    }
    private func key(_ id: String, workspace: WorkspaceStore) -> String {
        workspace.workspaceDirectory.resolvingSymlinksInPath().path + ":" + id.lowercased()
    }
    @discardableResult func newFlow(present: Bool = true) throws -> ProjectWindowController {
        let workspace = makeWorkspace()
        let source = WorkflowDocument(projectID: UUID().uuidString, name: "Untitled flow")
        let record = try workspace.projectLibrary.saveSource(.workflow(source))
        return try open(record, workspace: workspace, section: "Build", present: present)
    }
    @discardableResult func newFlow(templateID: String, present: Bool = true) throws -> ProjectWindowController {
        let source = try WorkflowTemplates.instantiate(id: templateID)
        let workspace = makeWorkspace()
        let record = try workspace.projectLibrary.saveSource(.workflow(source))
        let controller = try open(record, workspace: workspace, section: "Build", present: present)
        if let owner = controller.session.workspace.workflowSession?.owner,
           let first = owner.document.nodes.first(where: {
               if case .dataOperator(let id, _) = $0.payload { return ["kwip.data.sourceCSV", "kwip.data.sourceFiles"].contains(id) }
               return false
           }) {
            _ = owner.select(.init(nodeAddresses: [first.address]), lease: owner.lease())
            controller.session.workspace.showInspector = true
        }
        if present && presentsWindows {ApplicationNavigationWindows.shared.showTemplateSetup(workspace:controller.session.workspace)}
        return controller
    }
    @discardableResult func open(_ record: ProjectRecord, section: String? = nil, present: Bool = true) throws -> ProjectWindowController {
        if let existing = projects.values.first(where: { $0.session.workspace.externalLocation == nil && $0.session.workspace.activeProjectID?.lowercased() == record.id.lowercased() }) {
            if let section { existing.session.workspace.activeSection = section }
            if present {noteRecent(existing.session.workspace)}
            if present && presentsWindows { focusController(existing) }
            return existing
        }
        return try open(record, workspace: makeWorkspace(), section: section, present: present)
    }
    private func open(_ record: ProjectRecord, workspace: WorkspaceStore, section: String?, present: Bool) throws -> ProjectWindowController {
        let identity = key(record.id, workspace: workspace)
        if let existing = projects[identity] {
            if let section { existing.session.workspace.activeSection = section }
            if present {noteRecent(existing.session.workspace);focusController(existing) }
            return existing
        }
        let session = ProjectWindowSession(workspace: workspace)
        workspace.openManagedProject(record, section: section)
        guard workspace.activeProjectID?.lowercased() == record.id.lowercased() else {
            throw ProjectWindowError.openFailed(workspace.statusMessage)
        }
        return register(session, identity: identity, present: present)
    }
    @discardableResult func focusLessonCanvas(preferred:ProjectWindowSession?)->Bool {
        let controller=preferred.flatMap {preferred in projects.values.first {$0.session === preferred && ($0.session.workspace.lessonRailEntryID != nil || $0.session.workspace.isLearningLessonActive)}}
            ?? projects.values.first {$0.session.workspace.lessonRailEntryID != nil || $0.session.workspace.isLearningLessonActive}
        guard let controller else {return false}
        let workspace=controller.session.workspace
        workspace.showLessonRail=true
        workspace.activeSection=workspace.isLearningLessonActive ? "Learn":"Build"
        if presentsWindows {focusController(controller)} else {select(window:controller.window)}
        return true
    }
    @discardableResult func startFreshLesson(entryID:String,operation:String?=nil,workspace:WorkspaceStore)throws->ProjectWindowController {
        guard !workspace.isBusy,let source=projects.values.first(where:{$0.session.workspace === workspace})?.session else {throw ProjectWindowError.openFailed("Use an idle open lesson canvas to start a fresh copy.")}
        let activeTask=entryID == "affine" ? source.affine.lessonTask?.identity.assessmentID : (workspace.lessonRailEntryID == entryID && source.tensor.isActive ? source.tensor.draft?.operation:nil)
        return try openLesson(entryID:entryID,operation:operation ?? activeTask,source:source,freshCopy:true)
    }
    @discardableResult func openLesson(entryID:String,operation:String?=nil,source:ProjectWindowSession,freshCopy:Bool=false) throws -> ProjectWindowController {
        guard ComponentLessonDestination.isAvailable(entryID:entryID) else {throw ProjectWindowError.openFailed("This lesson is unavailable.")}
        let workedOperation:String?
        if entryID=="module-composition-examples" {workedOperation=operation.flatMap {ModuleCompositionLearningTask(rawValue:$0)}.map {"prepared-"+$0.rawValue}}
        else if entryID=="checkpoint-examples" {workedOperation=operation.flatMap {task in CheckpointLearningExamples.all.first {$0.task.rawValue==task}?.id}}
        else {workedOperation=nil}
        if let operation,workedOperation == nil,ConstructionLessonCatalog.groups[entryID]?.contains(where:{$0.id==operation}) != true {
            throw ProjectWindowError.openFailed("This construction task is unavailable.")
        }
        source.workspace.refreshProjects()
        if !freshCopy,operation == nil,let saved=source.workspace.lessonReading.record(entryID).savedProjectID,
           let record=source.workspace.projects.first(where:{$0.id==saved}) {
            let controller=try open(record,section:nil)
            controller.session.workspace.showLesson(entryID:entryID)
            return controller
        }
        return try launch { session in
            let workspace=session.workspace
            let opened:Bool
            if let workedOperation {opened=session.tensor.start(operation:workedOperation,mode:.guided)}
            else {
            switch entryID {
            case "guided":opened=workspace.loadGuidedWorkflowCourse()
            case "model-output":opened=workspace.loadComponentWorkflowExample(ModelOutputPracticeExample.exampleID,device:"cpu")
            case "mixture-experts":opened=workspace.loadComponentWorkflowExample(MixtureExpertsPractice.lessonID,device:"cpu")
            case "fine-tuning":opened=workspace.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu")
            case "load-model","text-generation":opened=workspace.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu")
            case "model-input":opened=workspace.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu")
            case "union","summarize","sort","unique","reshapeTable","table-output","text-encoding","sequence-windows","data02-expression","data02-regex-match":
                opened=workspace.loadComponentWorkflowExample(entryID,device:"cpu")
            case "splitText":opened=workspace.loadComponentWorkflowExample(SplitTextPracticeExample.exampleID,device:"cpu")
            case "columns":opened=workspace.loadComponentWorkflowExample(ColumnsPracticeExample.exampleID,device:"cpu")
            case "filter":opened=workspace.loadComponentWorkflowExample(FilterPracticeExample.exampleID,device:"cpu")
            case "document-classification":opened=workspace.loadComponentWorkflowExample(DocumentClassificationPracticeExample.exampleID,device:"cpu")
            case "self-play":opened=workspace.loadComponentWorkflowExample(SelfPlayPractice.lessonID,device:"cpu")
            case "game-episodes":opened=workspace.loadComponentWorkflowExample(GameEpisodePractice.lessonID,device:"cpu")
            case "game-rules":opened=workspace.loadComponentWorkflowExample(GameRuleConfiguration.lessonID,device:"cpu")
            case "prebuilt-models":opened=workspace.loadComponentWorkflowExample(PrebuiltModelsPractice.lessonID,device:"cpu")
            case "typed-decisions":opened=workspace.loadComponentWorkflowExample(TypedDecisionPracticeExample.exampleID,device:"cpu")
            case "classify-audio":opened=workspace.loadComponentWorkflowExample(AudioClassificationPracticeExample.exampleID,device:"cpu")
            case "transcribe-audio":opened=workspace.loadComponentWorkflowExample(SpeechTranscriptionPracticeExample.exampleID,device:"cpu")
            case "extract-text":opened=workspace.loadComponentWorkflowExample(OCRPracticeExample.exampleID,device:"cpu")
            case "prepare-media":opened=workspace.loadComponentWorkflowExample(PrepareMediaPracticeExample.exampleID,device:"cpu")
            case "file-input":opened=workspace.loadComponentWorkflowExample(FileInputViewPracticeExample.exampleID,device:"cpu")
            case "preprocess", "normalization":opened=workspace.loadComponentWorkflowExample(PreprocessPracticeExample.exampleID,device:"cpu")
            case "formula":opened=workspace.loadComponentWorkflowExample(FormulaPracticeExample.exampleID,device:"cpu")
            case "partition":opened=workspace.loadComponentWorkflowExample(PartitionPracticeExample.exampleID,device:"cpu")
            case "affine":
                var identity=ConstructionLessonCatalog.affineIdentities.first {$0.assessmentID == operation} ?? .composition
                if identity.assessmentID == ModelConstructionLessonIdentity.recovery(taskSeed:0).assessmentID {
                    identity = .recovery(taskSeed:freshCopy ? (source.affine.lessonTask?.identity.taskSeed ?? UInt64.random(in:UInt64.min...UInt64.max)):UInt64.random(in:UInt64.min...UInt64.max))
                }
                session.affine.start(identity:identity,mode:.assessment);opened=session.affine.isActive
            case "sigmoid-lesson":session.affine.start(identity:.sigmoid,mode:.assessment);opened=session.affine.isActive
            case "tanh-lesson":session.affine.start(identity:.tanh,mode:.assessment);opened=session.affine.isActive
            case "relu-lesson":session.affine.start(identity:.relu,mode:.assessment);opened=session.affine.isActive
            case "parallel":opened=workspace.loadPreparedWorkflowExample()
            case "tensor-example":opened=workspace.loadTensorLayoutWorkflowExample()
            default:
                if let operation=operation ?? ConstructionLessonCatalog.groups[entryID]?.first?.id {
                    opened=session.tensor.start(operation:operation,mode:.assessment)
                } else {
                // Reference/family entries keep their existing exercise choices in the rail.
                do {
                    let document=WorkflowDocument(projectID:UUID().uuidString,name:LearningLibraryPresentation.items.first(where:{$0.id==entryID})?.title ?? "Lesson")
                    let record=try workspace.projectLibrary.saveSource(.workflow(document))
                    opened=workspace.beginWorkflowProject(id:record.id)
                } catch {workspace.statusMessage=error.localizedDescription;return false}
                }
            }
            }
            guard opened else {return false}
            workspace.showLesson(entryID:entryID)
            return true
        }
    }
    @discardableResult private func launch(_ start: (ProjectWindowSession) -> Bool) throws -> ProjectWindowController {
        let session = ProjectWindowSession(workspace: makeWorkspace())
        session.workspace.startsIndependentLesson = true
        defer { session.workspace.startsIndependentLesson = false }
        guard start(session), let id = session.workspace.activeProjectID else { throw ProjectWindowError.openFailed(session.workspace.statusMessage) }
        return register(session, identity: key(id, workspace: session.workspace), present: true)
    }
    private func register(_ session: ProjectWindowSession, identity: String, present: Bool) -> ProjectWindowController {
        let controller = ProjectWindowController(session: session)
        installRouting(session)
        controller.newTab = { [weak self] in self?.createNewFromMenu() }
        controller.didClose = { [weak self, weak controller] in
            guard let self, let controller, self.projects[identity] === controller else { return }
            self.projects.removeValue(forKey: identity)
            self.importedFiles = self.importedFiles.filter { $0.value != identity }
            self.preferencesObservations.removeValue(forKey: ObjectIdentifier(controller.session))
            if self.activeSession === controller.session { self.select(window: nil) }
        }
        projects[identity] = controller
        if present {noteRecent(session.workspace)}
        if !RestorationCatalogSnapshot.shared.isActive {
            for peer in projects.values { peer.session.workspace.refreshProjects() }
            launcher?.workspace.refreshProjects()
        }
        if present && presentsWindows { presentController(controller) }
        return controller
    }
    private func flowCatalog() -> ExternalFlowCatalog {
        if let externalCatalog { return externalCatalog }
        let catalog = (launcher?.workspace ?? makeWorkspace()).externalFlowCatalog
        externalCatalog = catalog
        return catalog
    }
    @discardableResult func openExternal(_ record: ExternalFlowCatalog.Record, present: Bool = true) throws -> ProjectWindowController {
        if let existing = projects.values.first(where: { $0.session.workspace.externalLocation?.id == record.id }) {
            guard existing.session.workspace.externalLocation?.rootURL == record.rootURL else { throw ProjectWindowError.openFailed("This external location changed. Close its current tab before reopening it.") }
            if present {noteRecent(existing.session.workspace)}
            if present && presentsWindows { focusController(existing) }
            return existing
        }
        let lease = try flowCatalog().open(record)
        let location = ExternalProjectLocation(id: record.id, projectID: record.projectID, rootURL: lease.rootURL)
        let workspace = makeExternalWorkspace(location)
        workspace.externalFolderLease = lease
        let source = try workspace.projectLibrary.loadSource(id: record.projectID)
        let project = ProjectRecord(id: record.projectID, name: source.name, path: lease.rootURL.path, updatedAt: Date())
        return try open(project, workspace: workspace, section: "Build", present: present)
    }
    /// File selection authorizes only that file; editing the closure needs an explicit enclosing-folder grant.
    @discardableResult private func openSelectedFile(_ url: URL) throws -> ProjectWindowController? {
        do { return try openFile(url) }
        catch ProjectWindowError.externalRoot { chooseFlowFolder(for: url); return nil }
    }
    private func chooseFlowFolder(for root: URL) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.allowsMultipleSelection = false
        panel.title = "Allow access to flow folder"
        panel.message = "Choose the folder containing this flow to edit its canvas, subflows and saved revisions."
        panel.directoryURL = root.deletingLastPathComponent()
        panel.prompt = "Open flow folder"
        guard panel.runModal() == .OK, let folder = panel.url else { return }
        do {
            let record = try flowCatalog().attach(folder: folder, rootURL: root)
            _ = try openExternal(record)
            launcher?.workspace.refreshProjects()
        } catch { show(error) }
    }
    /// Managed Main files stay authoritative. Unsupported external roots are never flattened into imports.
    @discardableResult func openFile(_ url: URL, present: Bool = true) throws -> ProjectWindowController {
        let canonical = url.resolvingSymlinksInPath()
        if let identity = importedFiles[canonical], let existing = projects[identity] {
            if present && presentsWindows { focusController(existing) }
            return existing
        }
        if let existing = projects.values.first(where: {
            let workspace = $0.session.workspace
            return (workspace.workflowSession?.owner.flowRootURL ?? workspace.projectURL)?.resolvingSymlinksInPath() == canonical
        }) {
            if present && presentsWindows { focusController(existing) }
            return existing
        }
        let workspace = makeWorkspace()
        if let record = try workspace.projectLibrary.listSources().first(where: { URL(fileURLWithPath: $0.path).resolvingSymlinksInPath() == canonical }) {
            return try open(record, workspace: workspace, section: nil, present: present)
        }
        if url.pathExtension.lowercased() == "kwipflow" {
            if let record = try flowCatalog().list().first(where: { $0.rootURL == canonical }) { return try openExternal(record, present: present) }
            throw ProjectWindowError.externalRoot
        }
        let record = try workspace.projectLibrary.importSource(from: url)
        let controller = try open(record, workspace: workspace, section: nil, present: present)
        importedFiles[canonical] = key(record.id, workspace: workspace)
        return controller
    }
    private func presentController(_ controller: ProjectWindowController) {
        guard let window = controller.window else { return }
        let active = NSApp.keyWindow
        let host = active?.tabbingIdentifier == window.tabbingIdentifier ? active : NSApp.orderedWindows.first { $0 !== window && $0 !== launcherWindow && $0.tabbingIdentifier == window.tabbingIdentifier }
        if let host, host !== window { host.addTabbedWindow(window, ordered: .above); host.tabGroup?.selectedWindow = window }
        controller.showWindow(nil); window.makeKeyAndOrderFront(nil)
        launcherWindow?.close()
    }
    private func focusController(_ controller: ProjectWindowController) {
        controller.showWindow(nil); controller.window?.makeKeyAndOrderFront(nil)
        launcherWindow?.close()
    }
    func dismissWelcomeForOpenedCanvas() { launcherWindow?.close() }
    func restoreSavedCanvasLayout() {
        restoration?.retry()
        dismissWelcomeAfterRestoration()
    }
    private func dismissWelcomeAfterRestoration() {
        guard presentsWindows, restoration?.status == nil,
              openControllers.contains(where: { $0.window?.isVisible == true }) ||
              FlowCanvasWindowRegistry.shared.openControllers.contains(where: { $0.window?.isVisible == true }) else { return }
        launcherWindow?.close()
    }
    private func noteRecent(_ workspace:WorkspaceStore) {
        guard let url=workspace.workflowSession?.owner.flowRootURL ?? workspace.projectURL else{return}
        do {try RecentProjectHistory(directory:workspace.applicationDirectory).record(url)}
        catch {workspace.statusMessage="Opened project; recent-project metadata could not be updated. "+error.localizedDescription}
        launcher?.workspace.refreshProjects()
    }
    func openFromMenu(_ record: ProjectRecord) { do { _ = try open(record) } catch { show(error) } }
    func createNewFromMenu() { do { _ = try newFlow() } catch { show(error) } }
    func openFromMenu() {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = false; panel.canChooseDirectories = false
        panel.title = "Open project"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { _ = try openSelectedFile(url) } catch { show(error) }
    }
    private func show(_ error: Error) {
        navigationError=error.localizedDescription
        guard presentsWindows else{return}
        let alert=NSAlert(error:error);alert.runModal()
    }
    @discardableResult func save(window: NSWindow?) -> Bool {
        guard let session = session(for: window) else { return false }
        return session.save()
    }
    @discardableResult func close(_ controller: ProjectWindowController) -> Bool {
        guard controller.session.prepareToClose() else { return false }
        controller.window?.close(); return true
    }
    func prepareToTerminate() -> Bool {
        guard !hasActiveOperations else {
            let sessions=projects.values.map(\.session)+(launcher.map{[$0]} ?? [])
            for session in sessions where session.workspace.isBusy || session.preparation.current?.isPreparing == true {
                session.workspace.statusMessage="Finish or cancel active work before quitting or installing an update."
            }
            return false
        }
        for controller in projects.values { if !controller.session.prepareToClose() { return false } }
        return launcher?.prepareToClose() ?? true
    }
}

enum ProjectWindowError: LocalizedError {
    case openFailed(String), externalRoot
    var errorDescription: String? {
        switch self {
        case .openFailed(let message): return message
        case .externalRoot: return "Choose the containing flow folder to allow editing this external workflow and its subflows."
        }
    }
}

@MainActor final class ProjectWindowController: NSWindowController, NSWindowDelegate {
    let session: ProjectWindowSession
    var didClose: (() -> Void)?
    var newTab: (() -> Void)?
    private var observation: AnyCancellable?
    private var documentIdentity: DocumentWindowIdentity?
    init(session: ProjectWindowSession) {
        self.session = session
        let window = NSWindow(contentRect: .init(x: 0, y: 0, width: 1440, height: 900), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        super.init(window: window)
        window.isReleasedWhenClosed = false; window.delegate = self
        window.minSize = .init(width: 1050, height: 700)
        window.tabbingIdentifier = "info.kwip.KwipML.canvas-documents"; window.tabbingMode = .preferred
        window.contentView = NSHostingView(rootView: ProjectWorkspaceRootView(session: session))
        updateTitle(); window.center()
        observation = session.workspace.objectWillChange.receive(on: RunLoop.main).sink { [weak self] _ in self?.updateTitle() }
    }
    required init?(coder: NSCoder) { fatalError("Project sessions require a workspace") }
    private func updateTitle() {
        window?.title = session.workspace.modelSourceName
        window?.tab.title = session.workspace.modelSourceName
        if let window {
            let workspace = session.workspace
            let identity = DocumentWindowIdentity.project(
                url: workspace.workflowSession?.owner.flowRootURL ?? workspace.projectURL,
                isLesson: workspace.isLearningLessonActive || workspace.lessonRailEntryID != nil,
                lessonKind: workspace.lessonRailEntryID.flatMap {id in LearningLibraryPresentation.items.first(where:{$0.id==id})?.kind},
                isExternal: workspace.externalLocation != nil,
                isManaged: workspace.activeProjectID != nil && workspace.externalLocation == nil
            )
            if identity != documentIdentity { identity.apply(to: window); documentIdentity = identity }
        }
    }
    @objc override func newWindowForTab(_ sender: Any?) { newTab?() }
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        guard session.prepareToClose() else {
            let alert = NSAlert(); alert.messageText = "Project cannot close yet"; alert.informativeText = session.workspace.statusMessage; alert.runModal(); return false
        }
        return true
    }
    func windowWillClose(_ notification: Notification) { session.retire(); didClose?() }
}
