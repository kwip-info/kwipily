import AppKit
import Foundation
import SwiftUI
import KwipMLCore
import CryptoKit
import Combine

struct EpochMetric: Identifiable, Equatable {
    let epoch: Int
    let trainLoss: Double
    let validationLoss: Double
    let accuracy: Double?
    var id: Int { epoch }
}

@MainActor final class WorkspaceStore: ObservableObject {
    @Published var preparationSampling: PreparationSampling = .all {
        didSet {
            guard preparationSampling != oldValue else { return }
            workflowSession?.objectWillChange.send()
            workflowSession?.dataPreparation?.objectWillChange.send()
            workflowSession?.canvas.objectWillChange.send()
        }
    }
    var executionAdmission:()throws->Void = {}
    var startsIndependentLesson = false
    var exampleInWindow: ((String, PreparedWorkflowExample) -> Bool)?
    @Published var externalProjects: [ExternalFlowCatalog.Record] = []
    @Published var recentProjects:[RecentProjectEntry]=[]
    var openExternalProjectInWindow: ((ExternalFlowCatalog.Record) -> Void)?
    var renewExternalProjectAccess: ((ExternalFlowCatalog.Record) -> Void)?
    var locateExternalProject: ((ExternalFlowCatalog.Record) -> Void)?
    @Published var lessonRailEntryID: String?
    @Published var showLessonRail = true
    @Published var lessonNodeFocus = CanvasLessonFocus()
    var launchLessonEntryID: String?
    lazy var lessonReading = LessonReadingStore.shared(directory: workspaceDirectory)
    var focusOwningProject: (() -> Void)?
    var openProjectInWindow: ((ProjectRecord, String?) -> Void)?
    var projectCatalogChanged:(()->Void)?
    var newProjectInWindow: (() -> Void)?
    var openFileInWindow: ((URL) -> Void)?
    let gradientCapture = GradientCaptureStore()
    @Published var hasUnsavedChanges = false
    @Published var saveError: String?
    @Published var lastRunSourceHash: String?
    @Published var legacyGraph: GraphDocument
    @Published var workflowSession: WorkflowWorkspaceSession?
    @Published var workflowHistory: WorkflowHistoryStore?
    var workflowOwnerObservation: AnyCancellable?
    var workflowCanvasObservation: AnyCancellable?
    var workflowChildren: WorkflowChildWorkspace?
    @Published var modelAuthoring: ModelAuthoringStore?
    @Published var modelCanvas: ModelCanvasAdapter?
    var modelChildWindows: ModelChildWindowController?
    var modelOwnerObservation: AnyCancellable?
    var graph: GraphDocument {
        get {
            guard let owner = modelAuthoring else { return legacyGraph }
            return (try? owner.executionSnapshot().graph) ?? GraphDocument(name: modelSourceName, execution: modelSourceExecution)
        }
        set {
            guard modelAuthoring == nil, workflowSession == nil else { statusMessage = "Edit the canonical model canvas; an expanded execution snapshot cannot replace its source."; return }
            legacyGraph = newValue
        }
    }
    let registry: ComponentRegistry
    @Published var selectedNodeIDs: Set<String> = []
    @Published var selectedEdgeIDs: Set<String> = []
    @Published var selectedNodeID: String?
    @Published var selectedEdgeID: String?
    @Published var pendingSourceID: String?
    @Published var statusMessage = "Ready · Open the guided example or drag a component onto the canvas."
    @Published var canvasViewRevision = UUID()
    @Published var historySessionRevision = UUID()
    let appearanceState = AppAppearanceState()
    @Published var appPreferences = AppPreferences.factory {
        didSet { appearanceState.preferences = appPreferences.appearance; appearanceState.canvasZoomSensitivity=appPreferences.effectiveCanvasZoomSensitivity }
    }
    @Published var appPreferencesError: String?
    @Published var compactNavigation = true
    var legacyCompactNavigationPreference: () -> Bool = {
        UserDefaults.standard.object(forKey: "compactProjectNavigation") == nil ? true : UserDefaults.standard.bool(forKey: "compactProjectNavigation")
    }
    @Published var panelLayouts: [String: PanelLayoutPreference] = [:]
    @Published var modelCanvasViewState = CanvasViewState()
    @Published var dataCanvasViewState = CanvasViewState()
    @Published var compositeCanvasStates: [String: CanvasViewState] = [:]
    @Published var blockCanvasStates: [String: CanvasViewState] = [:]
    @Published var workspaceArrangements: [WorkspaceArrangement] = []
    @Published var arrangementError: String?
    var arrangementObservation: AnyCancellable?
    var arrangementTerminationObservation: AnyCancellable?
    var arrangementSaveTask: DispatchWorkItem?
    var isRestoringArrangement = false
    @Published var showLibrary = true
    @Published var showInspector = true
    @Published var showPreview = true
    @Published var activeSection = "Projects"
    @Published var showTutorial = false
    @Published var wasCancelled = false
    @Published var isRunning = false { didSet { synchronizeModelBusy() } }
    @Published var isPreviewing = false { didSet { synchronizeModelBusy() } }
    @Published var isPreparingData = false { didSet { synchronizeModelBusy() } }
    private(set) var dataPreparationReservationID = UUID()
    var cancelDataPreparation: (() -> Void)? { didSet { dataPreparationReservationID = UUID() } }
    @Published var previewCachePath: String?
    @Published var previewSignature: Data?
    @Published var previewSourceHash: String?
    @Published var previewWeightLabel = "Not evaluated"
    @Published var previewExecuted: [String] = []
    @Published var previewReused: [String] = []
    var previewArtifactHashes: [String: String] = [:]
    var previewCheckpoint: String?
    var previewWorker: WorkflowWorkerHandle?
    var previewLaunchTask: Task<Void, Never>?
    var previewWorkerLauncher: WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker
    var previewRequestToken: UUID?
    var previewModelRequest: ModelExecutionRequest?
    var previewCancelled = false
    @Published private(set) var isAssessingLearning = false
    @Published private(set) var lessonContext: LearningProjectContext?
    lazy var learningProjects = LearningProjectCoordinator(workspace: self)
    func updateLearningContext(_ value: LearningProjectContext?) { lessonContext = value }
    func setLearningAssessmentActive(_ value: Bool) { isAssessingLearning = value; synchronizeModelBusy() }
    var affineLessonProjectID: String? { lessonContext?.kind == .affine ? lessonContext?.projectID : nil }
    var isLearningLessonActive: Bool { lessonContext != nil && lessonContext?.projectID == activeProjectID }
    var isNormalizationLessonActive: Bool { isLearningLessonActive && lessonContext?.kind == .normalization }
    var isAffineLessonActive: Bool { isLearningLessonActive && lessonContext?.kind == .affine }
    @discardableResult func allowOrdinaryExecution() -> Bool {
        guard !isLearningLessonActive else { statusMessage = "Use the lesson assessment; ordinary execution and model export omit lesson inputs."; return false }
        return true
    }
    var isLocallyBusy: Bool { workflowSession?.modelLoader.blocksWorkspaceOperations == true || workflowSession?.isManagingTrainedArtifacts == true || workflowSession?.isStagingCheckpoint == true || workflowSession?.isStagingEvaluation == true || isAssessingLearning || isRunning || isPreviewing || isPreparingData || isInspectingData }
    @Published var metrics: [EpochMetric] = []
    var latestReceivedMetric: EpochMetric?
    @Published var previews: [String: [String: Any]] = [:]
    @Published var predictions: [[String: Any]] = []
    @Published var outputDirectory: URL?
    @Published var resultSignature: Data?
    @Published var projects: [ProjectRecord] = []
    @Published var activeProjectID: String? { didSet { if oldValue != activeProjectID { invalidateDraftCode(); cancelPreviewExecution() } } }
    @Published var availableDevices = ["cpu"]
    @Published var resolvedDevice = "Not run"
    @Published var runtimeContractVersion: String?
    @Published var runtimeExecutionContract: BuiltInGraphExecutionContract?
    @Published var embeddingRuntimeCapability: EmbeddingRuntimeCapability?
    @Published var embeddingBagRuntimeCapabilities = EmbeddingBagRuntimeCapabilities()
    @Published var rmsNormRuntimeCapability:RMSNormRuntimeCapability?
    @Published var layerNormRuntimeCapability: LayerNormRuntimeCapability?
    @Published var matrixMultiplyRuntimeCapability: MatrixMultiplyRuntimeCapability?
    @Published var mixtureExpertsRuntimeAvailable=false
    @Published var bilinearRuntimeCapability: BilinearRuntimeCapability?
    @Published var conv1dRuntimeCapability: Conv1dRuntimeCapability?
    @Published var conv2dRuntimeCapability: Conv2dRuntimeCapability?
    @Published var backendRuntimeReadinessErrors: [String: String] = [:]
    @Published var sklearnRuntimeCapability: SklearnRuntimeCapabilities?
    @Published var tensorflowRuntimeCapability: TensorFlowRuntimeCapabilities?
    @Published var configurableActivationRuntimeCapabilities = ConfigurableActivationRuntimeCapabilities()
    @Published var poolingRuntimeCapabilities = PoolingRuntimeCapabilities()
    @Published var tensorCompositionRuntimeCapabilities = TensorCompositionRuntimeCapabilities()
    @Published var remainingActivationRuntimeCapabilities = RemainingActivationRuntimeCapabilities()
    @Published var spatialPatchRuntimeCapabilities = SpatialPatchRuntimeCapabilities()
    @Published var statefulNormalizationRuntimeCapabilities = StatefulNormalizationRuntimeCapabilities()
    @Published var optimizerRuntimeCapabilities = OptimizerRuntimeCapabilities()
    @Published var schedulerRuntimeCapabilities = SchedulerRuntimeCapabilities()
    @Published var dataLoadingRuntimeCapabilities = DataLoadingRuntimeCapabilities()
    @Published var programExportRuntimeCapability = ProgramExportRuntimeCapability()
    @Published var lifecycleRuntimeCapability = LifecycleRuntimeCapability()
    @Published var moduleCompositionRuntimeCapability: ModuleCompositionRuntimeCapability? = nil
  @Published var moduleCompositionDiscoveryEvidence: ModuleCompositionDiscoveryEvidence? = nil
    @Published var initializationRuntimeCapabilities = InitializationRuntimeCapabilities()
    @Published var gradientControlRuntimeCapabilities = GradientControlRuntimeCapabilities()
    @Published var lossRuntimeCapabilities = LossRuntimeCapabilities()
    @Published var recurrentRuntimeCapabilities = RecurrentRuntimeCapabilities()
    @Published var attentionRuntimeCapabilities = AttentionRuntimeCapabilities()
    @Published var tensorRoutingRuntimeCapabilities = TensorRoutingRuntimeCapabilities()
    @Published var depthwiseConv2dRuntimeCapability: DepthwiseConv2dRuntimeCapability?
    @Published var convTranspose3dRuntimeCapability: ConvTranspose3dRuntimeCapability?
    @Published var convTranspose2dRuntimeCapability: ConvTranspose2dRuntimeCapability?
    @Published var convTranspose1dRuntimeCapability: ConvTranspose1dRuntimeCapability?
    @Published var conv3dRuntimeCapability: Conv3dRuntimeCapability?
    @Published var lazyLinearRuntimeCapability: LazyLinearRuntimeCapability?
    @Published var softmaxRuntimeCapability: SoftmaxRuntimeCapability?
    @Published var geluRuntimeCapability: GeluRuntimeCapability?
    @Published var sigmoidRuntimeCapability: SigmoidRuntimeCapability?
    @Published var tensorLayoutRuntimeCapability: TensorLayoutRuntimeCapability?
    var capabilityProbeToken = UUID()
    let capabilityDiscoveryScope = UUID().uuidString
    var capabilityOperation: CapabilityDiscoveryOperation?
    var pendingCapabilityOperation: CapabilityDiscoveryOperation?
    var capabilityLaunch: CapabilityDiscoveryLaunch?
    @Published var capabilityStatus = "Checking local runtime…"
    @Published var dataColumns: [String] = []
    @Published var dataRows: [[String]] = []
    @Published var dataRowCount = 0
    @Published var dataError: String?
    @Published var isInspectingData = false { didSet { synchronizeModelBusy() } }
    var dataInspectionHandle: WorkflowWorkerHandle?
    var dataInspectionTask: Task<Void, Never>?
    var dataInspectionValidationTask: Task<Void, Never>?
    // Unsaved inspections have an operation scope without creating a managed project.
    let transientDataInspectionProjectID = UUID().uuidString
    var dataInspectionGeneration = UUID()
    var activeDataInspectionToken: UUID?
    var dataInspectionTerminalToken: UUID?
    var dataPreviewSourcePath: String?
    var dataInspectionExecute: SourceInspectionExecutor?
    @Published var generatedCode = ""
    @Published var draftCode = ""
    @Published var draftCodeFiles: [String: String] = [:]
    @Published var codePackageMessage: String?
    @Published var workflowCodeScope: WorkflowCodePlan.Scope = .workflow
    @Published var draftCodeError: String?
    @Published var isGeneratingCode = false
    @Published var codeSignature: Data?
    @Published var codeContentIdentity: String?
    @Published var codeSourceContext: String?
    var currentCodeContentIdentity: String? { try? codeGenerationSnapshot().identity }
    var currentCodeGenerationKey: String { codeGenerationContextID + ":" + (currentCodeContentIdentity ?? "invalid") }
    var currentCodeReady: Bool {
        guard let current = try? codeGenerationSnapshot() else { return false }
        return codeSignature == current.signature && codeContentIdentity == current.identity && codeSourceContext == codeGenerationContextID
    }
    var codeGenerationToken = UUID()
    var codeGenerationWorker: WorkflowWorkerHandle?
    var codeGenerationTask: Task<Void, Never>?
    var codeRuntimeEvidence: WorkflowRuntimeEvidence?
    var codeWorkerLauncher: WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker
    @Published var experiments: [ExperimentRecord] = []
    @Published var comparisonA: String?
    @Published var comparisonB: String?
    var runStartedAt: Date?
    @Published var tutorialStep: Int?
    var execution: ExecutionSettings { modelAuthoring == nil ? (legacyGraph.execution ?? ExecutionSettings()) : (modelSourceExecution ?? ExecutionSettings()) }
    var epochCount: Int { get { execution.epochs } set { setExecution { $0.epochs = newValue } } }
    var projectLibrary: ProjectLibrary {
        let directory = workspaceDirectory.appendingPathComponent("projects")
        if let externalLocation {
            return ProjectLibrary(directory: directory, externalFlowRoot: externalLocation.rootURL, projectID: externalLocation.projectID)
        }
        return ProjectLibrary(directory: directory)
    }
    @Published var runError: String?
    @Published var eventLog: [String] = []
    @Published var projectURL: URL?
    @Published var lastRunRecipeVersion = 2
    @Published var lastCompleted = false
    @Published var activeTrainingRequest: TrainingRequestIdentity?
    @Published var verifiedCheckpointParent: CheckpointParent?
    @Published var checkpointPresentationError: String?

    var history = EditHistory<GraphEditorState>()
    var trainingWorkerLauncher: WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker
    private var trainingWorker: WorkflowWorkerHandle?
    private var trainingLaunchTask: Task<Void, Never>?
    private var pendingCompletion: [String: Any]?
    private var stdoutBuffer = ""
    private var stderrBuffer = ""
    private var cancelRequested = false
    private var activeSignature: Data?
    private var runToken = UUID()
    let projectRoot: URL
    let workspaceDirectory: URL
    var scheduledExecutionID:UUID?
    var isBusy:Bool {isLocallyBusy || ScheduledRunAdmission.isBlocked(self)}
    let executionAdmissionKey:String
    let applicationDirectory: URL
    let externalLocation: ExternalProjectLocation?
    var externalFolderLease: FlowFolderAccess.Lease?
    let runtime: RuntimeLocation?
    let runtimeError: String?

    init(rootOverride: URL? = nil, storageOverride: URL? = nil, resourcesOverride: URL? = nil, detectCapabilities: Bool = true, externalLocation: ExternalProjectLocation? = nil) {
        registry = try! ComponentRegistry.builtIn()
        legacyGraph = ExampleFlows.parallelClassifier
        let bundledRoot = Bundle.main.url(forResource: "ProjectRoot", withExtension: "txt").flatMap { try? String(contentsOf: $0, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) }
        projectRoot = rootOverride ?? URL(fileURLWithPath: ProcessInfo.processInfo.environment["KWIPML_ROOT"] ?? bundledRoot ?? FileManager.default.currentDirectoryPath)
        let resources = resourcesOverride ?? (rootOverride == nil ? Bundle.main.resourceURL : nil)
        // A malformed present bundle still uses customer storage and fails
        // closed below; it must never redirect a packaged app into .local.
        let hasBundledRuntime: Bool
        do { hasBundledRuntime = try RuntimeLocation.bundledRuntimeRoot(resources: resources) != nil }
        catch { hasBundledRuntime = true }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
        #if DEBUG
        let developmentStorage=MCPDevelopmentFixture.directory
        #else
        let developmentStorage:URL?=nil
        #endif
        let baseDirectory = storageOverride ?? developmentStorage ?? (hasBundledRuntime ? support.appendingPathComponent(DesktopDeployment.applicationSupportDirectoryName) : projectRoot.appendingPathComponent(".local"))
        applicationDirectory = baseDirectory
        executionAdmissionKey = baseDirectory.standardizedFileURL.resolvingSymlinksInPath().path
        self.externalLocation = externalLocation
        workspaceDirectory = externalLocation.map { baseDirectory.appendingPathComponent("external-projects").appendingPathComponent($0.id.uuidString) } ?? baseDirectory
        do {
            runtime = try RuntimeLocation.resolve(resources: resources, developmentRoot: projectRoot, allowDevelopment: rootOverride != nil || bundledRoot != nil || ProcessInfo.processInfo.environment["KWIPML_ROOT"] != nil)
            runtimeError = nil
        } catch { runtime = nil; runtimeError = error.localizedDescription }
        loadAppPreferences()
        refreshProjects()
        if let version = runtime?.packageVersions["torch"] {
            do { try installRuntimeContract(version: version, expectedVersion: version) }
            catch { capabilityStatus = error.localizedDescription }
        }
        if detectCapabilities { refreshCapabilities() }
        initializeWorkspaceArrangements()
        ScheduledRunAdmission.register(self)
    }
    var validation: ValidationResult {
        if let error = modelExecutionError { return ValidationResult(issues: [.init(id: "model-source", message: error)]) }
        return GraphValidator.validate(graph, registry: registry)
    }
    var selectedNode: GraphNode? { graph.nodes.first { $0.id == selectedNodeID } }
    var canUndo: Bool { (workflowSession?.owner.canUndo ?? modelAuthoring?.canUndo ?? history.canUndo) && !isBusy }
    var canRedo: Bool { (workflowSession?.owner.canRedo ?? modelAuthoring?.canRedo ?? history.canRedo) && !isBusy }
    var resultsStale: Bool {
        guard resultSignature != nil else { return false }
        if lastRunRecipeVersion < 2 { return true }
        if resultSignature != semanticSignature(graph) { return true }
        if let prepared = execution.preparedTraining { return sourceDigest(prepared.manifestPath, maximumBytes: 2_000_000) != prepared.manifestSHA256 }
        if let path = execution.datasetCSV { return lastRunSourceHash == nil || lastRunSourceHash != sourceDigest(path) }
        return false
    }
    private var sourceDigestCache: [String: (Date, Int, String)] = [:]
    func sourceDigest(_ path: String, maximumBytes: Int = 10_000_000) -> String? {
        do {
            return try sourceAccess.withAccess(to: path) { url -> String? in
                guard let attributes = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]), let size = attributes.fileSize, size <= maximumBytes, let modified = attributes.contentModificationDate else { return nil }
                if let cached = sourceDigestCache[path], cached.0 == modified, cached.1 == size { return cached.2 }
                guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
                defer { try? handle.close() }
                guard let data = try? handle.read(upToCount: maximumBytes + 1), data.count <= maximumBytes else { return nil }
                let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                sourceDigestCache[path] = (modified, size, digest)
                return digest
            }
        } catch { return nil }
    }
    func semanticSignature(_ document: GraphDocument) -> Data? {
        try? ModelExecutionIdentity.fullSignature(document, registry: registry)
    }
    func edit(_ action: (inout GraphDocument) -> Void) {
        guard modelAuthoring == nil, workflowSession == nil else { statusMessage = "Use the model canvas to edit its canonical source."; return }
        guard !isBusy else { statusMessage = "Stop the current run before changing the model."; return }
        let before = editorState
        action(&graph)
        history.record(before: before, after: editorState)
        if graph != before.document { hasUnsavedChanges = true; saveManagedProject() }
    }
    func addNode(type: String, x: Double, y: Double) {
        if type == ModelBlockPalette.type { _ = addModelBlock(x: x, y: y); return }
        guard !isBusy else { return }
        guard graph.nodes.count < 128 else { statusMessage = "This POC supports up to 128 blocks."; return }
        guard let definition = registry.component(type: type) else { return }
        let count = graph.nodes.filter { $0.type == type }.count + 1
        let node = GraphNode(id: UUID().uuidString, type: type, name: "\(definition.title) \(count)", x: max(0,x), y: max(0,y), config: Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) }))
        edit { $0.nodes.append(node) }; selectNode(node.id)
        statusMessage = "Added \(node.name). Configure it in the inspector."
    }
    func moveNode(id: String, x: Double, y: Double) {
        edit { document in guard let i = document.nodes.firstIndex(where: { $0.id == id }) else { return }; document.nodes[i].x = max(0,x); document.nodes[i].y = max(0,y) }
    }
    func renameNode(_ name: String) {
        guard let id = selectedNodeID else { return }
        edit { doc in if let i = doc.nodes.firstIndex(where: { $0.id == id }) { doc.nodes[i].name = String(name.prefix(80)) } }
    }
    func setConfig(_ key: String, value: Double) {
        guard let id = selectedNodeID else { return }
        _ = setConfigValue(key, value: .number(value), nodeID: id)
    }
    @discardableResult func setConfigValue(_ key: String, value: JSONValue, nodeID: String) -> Bool {
        guard !isBusy, selectedNodeID == nodeID, let node = selectedNode,
              let field = registry.component(type: node.type)?.fields(configuration:node.config).first(where: { $0.key == key }), let accepted = RegistryFieldCompatibility.normalized(value, for: field) else {
            statusMessage = "Value not applied. Select its block and use the field's declared type and bounds."; return false
        }
        edit { doc in if let i = doc.nodes.firstIndex(where: { $0.id == nodeID }) { doc.nodes[i].config[key] = accepted } }
        return graph.nodes.first(where: { $0.id == nodeID })?.config[key] == accepted
    }
    func connect(source: String, target: String, targetPort: String) {
        applyCommand(.connect(source: GraphPortAddress(source, "out"), target: GraphPortAddress(target, targetPort)))
    }
    func undo() {
        if let owner = workflowSession?.owner { reportWorkflowEdit(owner.undo(lease: owner.lease())); return }
        if let owner = modelAuthoring { reportModelEdit(owner.undo(lease: owner.lease())); return }
        guard !isBusy, let value = history.undo(current: editorState) else { return }
        installEditorState(value); hasUnsavedChanges = true; saveManagedProject()
    }
    func redo() {
        if let owner = workflowSession?.owner { reportWorkflowEdit(owner.redo(lease: owner.lease())); return }
        if let owner = modelAuthoring { reportModelEdit(owner.redo(lease: owner.lease())); return }
        guard !isBusy, let value = history.redo(current: editorState) else { return }
        installEditorState(value); hasUnsavedChanges = true; saveManagedProject()
    }
    @discardableResult func loadExample() -> Bool {
        guard !isBusy else { return false }
        guard beginProject(ExampleFlows.parallelClassifier) else { return false }
        statusMessage = "Parallel classifier loaded. Explore it or start a guided build."
        return true
    }
    func newProject() {
        if let newProjectInWindow { newProjectInWindow(); return }
        guard !isBusy else { return }
        if hasUnsavedChanges && !saveManagedProject() { return }
        guard flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition(), learningProjects.beforeProjectTransition() else { return }
        do {
            _ = try workflowDataRegistry()
            let document = WorkflowDocument(projectID: UUID().uuidString, name: "Untitled flow")
            _ = try projectLibrary.saveSource(.workflow(document))
            refreshProjects()
            _ = beginWorkflowProject(id: document.projectID)
        } catch { statusMessage = "New flow failed: \(error.localizedDescription)" }
    }
    @discardableResult func beginProject(_ document: GraphDocument, id: String? = nil) -> Bool {
        guard !isBusy else { return false }
        if let id {
            do {
                let url = try projectLibrary.documentURL(id: id)
                if FileManager.default.fileExists(atPath: url.path) {
                    let existing = try projectLibrary.loadSource(id: id)
                    guard case .legacyFlat = existing else {
                        statusMessage = "Open this project through its canonical source."
                        return false
                    }
                }
            } catch { statusMessage = "Open failed: \(error.localizedDescription)"; return false }
        }
        let current: GraphDocument
        do { current = try GraphMigration.toCurrent(document, registry: registry) }
        catch { statusMessage = "Open failed: \(error.localizedDescription)"; return false }
        if hasUnsavedChanges && !saveManagedProject() { return false }
        guard flushWorkflowCanvasBeforeTransition(), flushChildCanvasBeforeTransition(), learningProjects.beforeProjectTransition() else { return false }
        persistWorkspaceView()
        retireWorkflowSession()
        retireModelOwner()
        installProjectSession(current, id: id, owned: false)
        return true
    }
    func installProjectSession(_ current: GraphDocument, id: String?, owned: Bool) {
        isRestoringArrangement = true
        activeTrainingRequest = nil; verifiedCheckpointParent = nil; checkpointPresentationError = nil
        historySessionRevision = UUID()
        legacyGraph = current; activeProjectID = id; hasUnsavedChanges = !owned; history = EditHistory<GraphEditorState>()
        projectURL = nil; clearSelection(); pendingSourceID = nil
        previewCachePath = nil; previewSignature = nil; previewSourceHash = nil; previewExecuted = []; previewReused = []; previewWeightLabel = "Not evaluated"; previewCheckpoint = nil
        wasCancelled = false
        lastRunRecipeVersion = 2
        tutorialStep = nil; dataRowCount = 0; isInspectingData = false; eventLog = []; lastRunSourceHash = nil; latestReceivedMetric = nil; metrics = []; previews = [:]; predictions = []; outputDirectory = nil; resultSignature = nil; lastCompleted = false; generatedCode = ""; dataError = nil; dataColumns = []; dataRows = []; runError = nil; resolvedDevice = "Not run"
        activeSection = "Build"
        if owned { projectURL = workflowSession?.owner.flowRootURL ?? (try? projectLibrary.documentURL(id: id ?? "")); saveError = nil }
        else { _ = saveManagedProject() }
        if workflowSession == nil { restoreLastRun(); refreshExperiments() }
        invalidateDraftCode()
        restoreWorkspaceView()
        isRestoringArrangement = false
        persistWorkspaceView()
        if workflowSession == nil { learningProjects.restoreCurrentProject() }
        if workflowSession == nil && execution.datasetCSV != nil && !isLearningLessonActive { inspectData() }
    }
    func saveProject() {
        guard allowOrdinaryExecution() else { return }
        let panel = NSSavePanel(); panel.nameFieldStringValue = "\(modelSourceName).kwipml"; panel.title = "Save KwipML project"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try exportModelProject(to: url); statusMessage = "Saved \(url.lastPathComponent)" }
        catch { statusMessage = "Save failed: \(error.localizedDescription)" }
    }
    func openProject() {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = false; panel.canChooseDirectories = false; panel.title = "Open KwipML project"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if let openFileInWindow { openFileInWindow(url); return }
        do { let record = try projectLibrary.importSource(from: url); openManagedProject(record) }
        catch { statusMessage = "Open failed: \(error.localizedDescription)" }
    }
    func run(continuation: CheckpointContinuation? = nil) {
        if let host = workflowSession {
            guard continuation == nil else { statusMessage = "Checkpoint continuation must be configured on the workflow; this path is not yet available."; return }
            _ = host.run(); statusMessage = host.statusMessage; return
        }
        guard allowExecution() else {return}
        guard allowOrdinaryExecution() else { return }
        guard validation.isValid, !isBusy else { statusMessage = "Fix validation issues before running."; return }
        guard let runtime else { runError = runtimeError ?? "Runtime unavailable."; return }
        if let error = capabilityBoundaryError { runError = error; return }
        do {
            guard saveManagedProject(), let projectID = activeProjectID else { return }
            let capturedModelRequest = try modelAuthoring.map { try ModelExecutionRequest.capture(from: $0, registry: registry) }
            let capturedGraph = capturedModelRequest?.graph ?? graph
            let settings = capturedModelRequest?.settings ?? execution
            if let continuation, let message = checkpointCompatibility(continuation) { throw WorkflowArchiveError(message) }
            let runID = UUID().uuidString
            let directory = workspaceDirectory.resolvingSymlinksInPath().appendingPathComponent("workflow-operations/\(projectID)/\(runID)")
            let policy = try executionPolicy(for: projectID)
            let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let session = historySessionRevision
            let permissions = sourceAccess.directory, devices = gradientCapture.supportedDevices
            let capturedRegistry = registry
            let contextCurrent: () -> Bool = { [weak self] in
                guard let self, self.activeProjectID == projectID, self.historySessionRevision == session else { return false }
                if let capturedModelRequest { return capturedModelRequest.isCurrent(in: self.modelAuthoring) }
                return self.modelAuthoring == nil && self.graph == capturedGraph
            }
            previewCachePath = nil; previewSignature = nil; previewArtifactHashes = [:]; previewExecuted = []; previewReused = []; previewWeightLabel = "Training in progress"; wasCancelled = false; resolvedDevice = "Starting"
            latestReceivedMetric = nil; metrics = []; previews = [:]; predictions = []; eventLog = []; runError = nil; lastCompleted = false; pendingCompletion = nil; stdoutBuffer = ""; cancelRequested = false
            if execution.compositeSource != nil && !compositeSourceIsCurrent { eventLog.append(compositeSourceStatus) }
            gradientCapture.begin(config: execution.gradientCapture)
            runStartedAt = Date()
            activeTrainingRequest = TrainingRequestIdentity(runID: runID, continuation: continuation)
            outputDirectory = nil; resultSignature = nil; activeSignature = semanticSignature(capturedGraph); isRunning = true; activeSection = "Run"
            let token = UUID(); runToken = token
            trainingLaunchTask = Task { [self] in
                do {
                    let capture = Task.detached(priority: .userInitiated) {
                        try ModelTrainingLaunchPlan.capture(graph: capturedGraph, settings: settings,
                            modelRequest: capturedModelRequest, continuation: continuation,
                            sourcePermissions: permissions, supportedGradientDevices: devices, registry: capturedRegistry,
                            sampleCapture: policy.sampleCapture,
                            checkCancellation: {
                                try Task.checkCancellation()
                                _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                            })
                    }
                    let plan = try await withTaskCancellationHandler(operation: { try await capture.value }, onCancel: { capture.cancel() })
                    try Task.checkCancellation()
                    guard contextCurrent(), self.runToken == token else { throw CancellationError() }
                    if plan.inputs.contains(where: { input in
                        if case .bytes(let path, _) = input { return path == "input/model-registry.json" }
                        return false
                    }) {
                        guard let proof = self.moduleCompositionDiscoveryEvidence else {
                            throw GraphConfigurationError("Refresh the selected runtime to verify native module composition before running this model.")
                        }
                        _ = try proof.verifiedDescriptor(for: runtime, evidence: WorkflowRuntimeEvidence.capture(runtime))
                    }
                    let handle = try await WorkflowModelTrainingExecution.launch(workspace: self.workspaceDirectory, accountingRoot: self.applicationDirectory,
                        projectID: projectID, runtime: runtime, window: window, operationID: runID, storageLimitBytes: policy.storageLimitBytes,
                        inputs: { _ in plan.inputs }, arguments: { plan.arguments(in: $0) },
                        launcher: self.trainingWorkerLauncher, onLine: { [self] line in
                            guard self.runToken == token, contextCurrent() else { return }
                            self.consume(line + "\n")
                        }, onExit: { [self] code, stderr, verifiedOutput in
                            guard self.runToken == token else { return }
                            // Keep editing/project switching reserved until publication and history finish.
                            defer { self.isRunning = false; self.trainingWorker = nil; self.trainingLaunchTask = nil; self.activeTrainingRequest = nil }
                            guard contextCurrent() else { self.pendingCompletion = nil; self.lastCompleted = false; self.outputDirectory = nil; self.gradientCapture.interrupt(status: "cancelled"); return }
                            if self.cancelRequested {
                                self.pendingCompletion = nil; self.lastCompleted = false; self.outputDirectory = nil; self.wasCancelled = true; self.runError = nil
                                self.gradientCapture.interrupt(status: "cancelled")
                                self.recordExperiment(status: "cancelled", directory: directory.appendingPathComponent("artifacts"), error: nil)
                                self.statusMessage = "Run cancelled. No completed result published."; self.eventLog.append("Cancelled by user")
                            } else if code != 0 {
                                self.pendingCompletion = nil; self.runError = code == 125 && !stderr.isEmpty ? stderr : self.runError ?? (stderr.isEmpty ? "Worker failed with exit code \(code)." : stderr)
                                self.statusMessage = "Run failed. Inspect the error and retry."
                                self.gradientCapture.interrupt(status: "failed"); self.recordExperiment(status: "failed", directory: directory.appendingPathComponent("artifacts"), error: self.runError)
                            } else if let completed = self.pendingCompletion, let verifiedOutput {
                                let path = verifiedOutput.path
                                self.lastCompleted = true; self.resultSignature = self.activeSignature; self.outputDirectory = URL(fileURLWithPath: path)
                                self.previewCachePath = nil; self.previewSignature = nil; self.previewWeightLabel = "Training · final checkpoint"
                                self.previews = completed["previews"] as? [String: [String: Any]] ?? [:]
                                self.predictions = completed["predictions"] as? [[String: Any]] ?? []
                                self.statusMessage = "Run complete. Inspect real results or reveal the exported artifacts."
                                self.eventLog.append("Completed · checkpoint, model export and provenance saved")
                                self.pendingCompletion = nil
                                self.generatedCode = (try? String(contentsOf: URL(fileURLWithPath: path).appendingPathComponent("model.py"), encoding: .utf8)) ?? ""
                                let manifestData = (try? DiagnosticFileIO.read(root: URL(fileURLWithPath: path), parts: ["manifest.json"], limit: 20_000_000)) ?? Data()
                                if let manifest = (try? JSONSerialization.jsonObject(with: manifestData)) as? [String: Any] { self.lastRunRecipeVersion = manifest["recipeVersion"] as? Int ?? 1; self.lastRunSourceHash = (manifest["provenance"] as? [String: Any])?["sourceHash"] as? String }
                                if continuation != nil { self.restoreContinuationMetrics(from: URL(fileURLWithPath: path)) }
                                self.gradientCapture.complete(directory: URL(fileURLWithPath: path), manifestData: manifestData, completedMetadata: completed["gradientCapture"].flatMap { try? JSONSerialization.data(withJSONObject: $0) })
                                self.refreshVerifiedCheckpointParent()
                                self.rememberLastRun(path)
                                self.recordExperiment(status: "completed", directory: URL(fileURLWithPath: path), error: nil)
                            } else { self.runError = "Worker exited without a completed result and manifest."; self.gradientCapture.interrupt(status: "failed"); self.recordExperiment(status: "failed", directory: directory.appendingPathComponent("artifacts"), error: self.runError) }
                        })
                    if self.runToken == token, self.isRunning {
                        self.trainingWorker = handle; self.trainingLaunchTask = nil
                        if self.cancelRequested || Task.isCancelled { handle.cancel() }
                    }
                } catch {
                    guard self.runToken == token else { return }
                    defer { self.isRunning = false; self.trainingWorker = nil; self.trainingLaunchTask = nil; self.activeTrainingRequest = nil }
                    self.pendingCompletion = nil; self.lastCompleted = false; self.outputDirectory = nil
                    guard contextCurrent() else { self.gradientCapture.interrupt(status: "cancelled"); return }
                    let cancelled = self.cancelRequested || error is CancellationError
                    self.wasCancelled = cancelled
                    self.runError = cancelled ? nil : error.localizedDescription
                    self.gradientCapture.interrupt(status: cancelled ? "cancelled" : "failed")
                    self.recordExperiment(status: cancelled ? "cancelled" : "failed", directory: directory.appendingPathComponent("artifacts"), error: self.runError)
                    self.statusMessage = cancelled ? "Run cancelled. No completed result published." : "Run failed: \(error.localizedDescription)"
                }
            }
            statusMessage = "Capturing training inputs for \(settings.device.uppercased()) compute…"
        } catch { isRunning = false; trainingWorker = nil; trainingLaunchTask = nil; activeTrainingRequest = nil; runError = error.localizedDescription; gradientCapture.interrupt(status: "failed") }
    }

    func consume(_ text: String) {
        stdoutBuffer += text
        while let newline = stdoutBuffer.firstIndex(of: "\n") {
            let line = String(stdoutBuffer[..<newline]); stdoutBuffer.removeSubrange(...newline)
            guard let data = line.data(using: .utf8), let event = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any], let name = event["event"] as? String else { continue }
            switch name {
            case "started":
                gradientCapture.consumeStarted(data)
                resolvedDevice = event["device"] as? String ?? "cpu"
                let dataDescription = execution.preparedTraining == nil ? "split seed \(execution.splitSeed)" : "frozen prepared partitions"
                eventLog.append("Worker started · \(resolvedDevice.uppercased()) · model seed \(execution.seed) · \(dataDescription)")
            case "gradientSummary":
                gradientCapture.consumeSummary(data)
            case "warning":
                if !cancelRequested, let warning = PoolingExecutionWarning.decode(data, graph: graph) {
                    eventLog.append(warning.message)
                }
            case "epoch":
                if let epoch = event["epoch"] as? Int, let train = event["trainLoss"] as? Double, let val = event["validationLoss"] as? Double, let accuracy = event["accuracy"] as? Double {
                    let metric = EpochMetric(epoch: epoch, trainLoss: train, validationLoss: val, accuracy: accuracy)
                    latestReceivedMetric = metric
                    if MetricSampling.shouldPublish(epoch: epoch, totalEpochs: epochCount) { metrics.append(metric) }
                }
            case "completed":
                guard !cancelRequested else { continue }
                pendingCompletion = event
            case "error":
                guard !cancelRequested else { continue }
                runError = event["message"] as? String ?? "Unknown worker error"; eventLog.append("Worker reported an error")
            default: break
            }
        }
    }
    func cancelRun() { if workflowSession?.isManagingTrainedArtifacts == true || workflowSession?.isStagingCheckpoint == true || workflowSession?.isStagingEvaluation == true { workflowSession?.cancel(); return }; if isAssessingLearning { learningProjects.cancelAssessment(); return }; if isInspectingData { cancelDataInspection(); return }; if isPreparingData { cancelDataPreparation?(); return }; cancelPreviewExecution(); cancelRequested = true; trainingLaunchTask?.cancel(); trainingWorker?.cancel(); statusMessage = "Cancelling at the worker boundary…" }
    func revealArtifacts() { if let directory = outputDirectory { NSWorkspace.shared.open(directory) } }
}
