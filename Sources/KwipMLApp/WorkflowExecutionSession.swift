import Foundation
import Combine
import CryptoKit
import KwipMLCore

struct WorkflowCapturedSource: Equatable {
    let url: URL
    let sha256: String
    let byteCount: Int
    var collectionMembers:[FileInputCapturedMember] = []
    var isFileCollection:Bool = false
    var totalCapturedByteCount:Int {byteCount+collectionMembers.reduce(0){$0+$1.byteCount}}
    func stagedInputs() -> (binding:String,files:[WorkflowPreviewCapture.Input]) {
        if isFileCollection {
            let prefix="input/file-collections/"+sha256
            return (prefix+"/manifest.json",[.file(path:prefix+"/manifest.json",source:url,expectedSHA256:sha256)]+collectionMembers.map {.file(path:prefix+"/"+$0.capturePath,source:$0.url,expectedSHA256:$0.sha256)})
        }
        let path="input/sources/"+sha256+"/"+url.lastPathComponent
        return (path,[.file(path:path,source:url,expectedSHA256:sha256)])
    }
    static func capture(_ url: URL) throws -> Self {
        let bytes = try DiagnosticFileIO.read(root: url.deletingLastPathComponent(), parts: [url.lastPathComponent], limit: 10_000_000)
        return Self(url: url, sha256: WorkflowPhaseArtifacts.hash(bytes), byteCount: bytes.count)
    }
}
struct WorkflowWorkerHandle { let cancel: () -> Void }
typealias WorkflowWorkerLauncher = (URL, [String], URL, @escaping (String) -> Void, @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle
struct WorkflowExecutionReservation {
    let acquire: () -> UUID?
    let release: (UUID) -> Void
}
enum WorkflowExecutionPhase: String, Codable { case idle, preparing, training, completed, failed, cancelled }
struct WorkflowExecutionOutcome {
    let runID: UUID
    let request: WorkflowExecutionRequest
    let status: WorkflowExecutionPhase
    let failedPhase: WorkflowExecutionPhase?
    let sources: [WorkflowNodeAddress: WorkflowCapturedSource]
    let directory: URL
    let preparationManifestSHA256: String?
    let modelManifestSHA256: String?
    let artifactHashes: [String: String]
    let error: String?
    let recordError: String?
}

/// One captured operation owns its reservation across both processes. No editable
/// store is constructed from the compiled plan, and worker output never edits source.
@MainActor final class WorkflowExecutionSession: ObservableObject {
    private(set) var preparationMeasurements = PreparationPhaseMeasurements()
    let request: WorkflowExecutionRequest
    let console = WorkflowRunConsole()
    let runID = UUID()
    private let historyPolicy:RunHistoryPolicy
    @Published private(set) var canvasProgress:WorkflowCanvasProgress
    @Published private(set) var modelOperationProgress:WorkflowModelOperationProgress?
    @Published private(set) var phase: WorkflowExecutionPhase = .idle
    @Published private(set) var statusMessage = "Ready"
    @Published private(set) var predictionPreview: PredictionPreview?
    @Published private(set) var objectivePredictionPreview: ObjectivePredictionPreview?
    @Published private(set) var bagPredictionPreview: BagPredictionPreview?
    @Published private(set) var objectiveEvaluation: ObjectiveEvaluation?
    @Published private(set) var trainingControlReport:TrainingControlReport?
    @Published private(set) var performanceCapture: WorkflowPerformanceCapture?
    @Published private(set) var derivativeResults: DerivativeResults?
    // Exact validated ledger for completion checks and evidence. Only the live
    // presentation uses the shared ~200-per-run epoch schedule, at most five
    // times per second. Delayed delivery retains the eligible epoch snapshot.
    private(set) var metrics: [EpochMetric] = []
    @Published private(set) var liveMetrics: [EpochMetric] = []
    private var progressLedger=TrainingProgressLedger()
    @Published private(set) var liveTrainingProgress:[TrainingProgressPoint]=[]
    @Published private(set) var trainingProgressObserved=0
    private var progressWorker:String?
    private var progressInitialEpoch=0
    private var progressEpochBatches:Int?
    private let reportingPolicy=TrainingReportingPolicy.current
    private let progressChanges=PassthroughSubject<[TrainingProgressPoint],Never>()
    private lazy var progressPublication:AnyCancellable=progressChanges.throttle(for:.milliseconds(200),scheduler:RunLoop.main,latest:true).sink {[weak self] rows in
        guard let self,!self.finished else {return};self.liveTrainingProgress=rows;self.trainingProgressObserved=self.progressLedger.observed
    }

    private let metricChanges = PassthroughSubject<[EpochMetric], Never>()
    private lazy var metricPublication: AnyCancellable = metricChanges
        .throttle(for: .milliseconds(200), scheduler: RunLoop.main, latest: true)
        .sink { [weak self] snapshot in
            guard let self, !self.finished else { return }
            self.liveMetrics = snapshot
        }
    @Published private(set) var previews: [String: [String: Any]] = [:]
    @Published private(set) var modelCaptures: WorkflowVerifiedModelCaptures?
    private var typedPreviews: [String: [String: JSONValue]] = [:]
    var preparationOutputs:[String:[String:JSONValue]] { typedPreviews }
    private(set) var capturedTables:WorkflowCapturedTables?
    private(set) var routingTables:[String:WorkflowPredictionTableEvidence]=[:]
    func outputTableArtifact(_ port:String)->TableViewQuerySession.Artifact? {guard phase == .completed else{return nil};return port=="predictions" ? predictionTableArtifact:routingTables[port]?.artifact(directory:directory)}
    private(set) var predictionTable:WorkflowPredictionTableEvidence?
    var predictionTableArtifact:TableViewQuerySession.Artifact? {guard phase == .completed else {return nil};return predictionTable?.artifact(directory:directory)}
    private let groupOwner:WorkflowGroupOwner?
    private var publishedLaunch:WorkflowPublishedLaunch?
    private var operationOwner: WorkflowOperationOwner?
    private var operationRequest: WorkflowOperationRequest?
    private let preparationOperationID = UUID().uuidString
    private let trainingOperationID = UUID().uuidString
    private let storageLimitBytes: Int64?
    private let acceptedBudgetWindow: ExecutionBudgetWindow?
    private let accountingRoot: URL?
    private var storageAdmission: WorkflowExecutionStorageAdmission?
    private var budgetWindow: ExecutionBudgetWindow?
    private var failureCode: WorkflowFailureCode?
    private var launchReady=false
    private var launchProtocolFailed=false
    private let checkpointSnapshot: WorkflowCheckpointSnapshot?
    private var parentRecord: WorkflowLifecycleParentRecord?
    private var archiveCaptureTask:Task<Void,Never>?
    private var archiveCaptureID: UUID?
    private let owner: WorkflowAuthoringStore
    private let sources: [WorkflowNodeAddress: WorkflowCapturedSource]
    private let directory: URL
    private let reservation: WorkflowExecutionReservation
    private let launch: WorkflowWorkerLauncher
    private let onTerminal: (WorkflowExecutionOutcome) -> Void
    // Accepted work must outlive a replaced UI host until its terminal callback.
    private var retainedSession: WorkflowExecutionSession?
    private var reservationID: UUID?
    private var worker: WorkflowWorkerHandle?
    private var phaseToken: UUID?
    #if KWIPML_RECOVERY_UAT
    private var recoveryUATPreparationToken: UUID?
    #endif
    private var cancelled = false
    private var finished = false
    private var started = false
    private var completion: [String: Any]?
    private var workerError: String?
    @Published private(set) var preparationFailure:DataFlowFailureDetail?
    private var stagedPreparationFailure:DataFlowFailureDetail?
    /// The owning host supplies binding/source/destination and replacement freshness.
    /// Standalone/retired sessions cannot claim current canvas attribution.
    var preparationFailureContextIsCurrent:(()->Bool)?
    var preparationFailureIsCurrent:Bool {
        phase == .failed && preparationFailure != nil
            && owner.isCurrent(owner.lease()) && owner.sessionID == request.lease.sessionID
            && owner.document.matchesCapturedConstruction(request.authoringDocument)
            && preparationFailureContextIsCurrent?() == true
    }
    private var preparation: WorkflowPreparedSelection?
    private var artifactHashes: [String: String] = [:]
    private var failedGradientCollector: WorkflowFailedGradientCollector?
    private var failedGradientDescriptor: WorkflowFailedGradientDescriptor?
    private var failedGradientNotice: String?
    private var preparedHash: String?
    private var trainedHash: String?
    private var ownsDirectory = false
    private var startedAt: Date?
    private var captureReference: WorkflowArchiveFile?
    private var artifactFiles: [WorkflowArchiveFile] = []
    private var workerRunID: String?
    private var sourceInputs: [String: String] = [:]
    private var objectiveLiveContext:WorkflowObjectiveArtifactContext?
    private var bagLiveContext:WorkflowBagArtifactContext?
    private var objectiveLiveEvaluationContext:WorkflowObjectiveEvaluationContext?
    private var schedulerWorkerRunID:String?
    private var schedulerInitialTick=0,schedulerRecordCount=0,schedulerEventBytes=0,schedulerBatchesPerEpoch=0
    @Published private(set) var savedOutputs:[[String:JSONValue]]=[]
    private var outputDestinationAccess:DataOutputDestinationAccess?
    var databaseWrite:((DataFlowNode,TableViewQuerySession.Artifact,String,ExecutionBudgetWindow)async throws->[String:JSONValue])?
    private var previewBytes = 0

    init(request: WorkflowExecutionRequest, owner: WorkflowAuthoringStore,
         sources: [WorkflowNodeAddress: WorkflowCapturedSource], directory: URL,
         reservation: WorkflowExecutionReservation,
         groupOwner:WorkflowGroupOwner? = nil, outputDestinationAccess:DataOutputDestinationAccess? = nil,
         checkpointSnapshot: WorkflowCheckpointSnapshot? = nil,
         storageLimitBytes: Int64? = nil, accountingRoot: URL? = nil,
         acceptedBudgetWindow: ExecutionBudgetWindow? = nil,
         launch: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
         onTerminal: @escaping (WorkflowExecutionOutcome) -> Void) {
        self.outputDestinationAccess=outputDestinationAccess
        self.groupOwner=groupOwner
        self.storageLimitBytes = storageLimitBytes
        self.accountingRoot = accountingRoot
        self.acceptedBudgetWindow = acceptedBudgetWindow
        self.checkpointSnapshot = checkpointSnapshot
        canvasProgress = WorkflowCanvasProgress(context:UUID(),allowedNodes:Set(request.plan.dataflow.nodes.map {WorkflowNodeAddress(domain:.data,localID:$0.id)}).union([request.train]),edges:request.document.edges)
        self.historyPolicy=request.sampleCapture.map {.init(metricsAndDiagnostics:$0.savesDiagnosticDetails,componentOutputs:$0.savesComponentOutputs)} ?? owner.runHistoryPolicy
        self.request = request; self.owner = owner; self.sources = sources
        self.directory = directory; self.reservation = reservation; self.launch = launch; self.onTerminal = onTerminal
    }
    nonisolated static func launchWorker(_ executable: URL, _ arguments: [String], _ directory: URL,
        _ line: @escaping (String) -> Void, _ exit: @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle {
        if arguments.contains("--kwip-supervise") || arguments.contains("--kwip-evaluate-supervise") || arguments.contains("--kwip-export-supervise") || arguments.contains("--kwip-operation-supervise") {
            let handle=try ProcessPump.launchWithLifetime(executable:executable,arguments:arguments,directory:directory,onLine:line,onExit:exit)
            return WorkflowWorkerHandle(cancel:{handle.cancel()})
        }
        let process = try ProcessPump.launch(executable: executable, arguments: arguments, directory: directory, onLine: line, onExit: exit)
        return WorkflowWorkerHandle(cancel: { if process.isRunning { process.terminate() } })
    }

    @discardableResult func start() -> Bool {
        guard !started, !finished else { return false }
        do {
            try owner.captureHistoryDecision(id:runID.uuidString,policy:historyPolicy)
            try checkGroupAuthority()
            if let acceptedBudgetWindow {
                guard acceptedBudgetWindow.policy == request.executionBudget else {
                    throw WorkflowPhaseError("The accepted execution window differs from the captured request policy.")
                }
                _ = try acceptedBudgetWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
            }
            if storageLimitBytes != nil {
                guard request.executionBudget != nil, request.experiment == nil, groupOwner == nil else {
                    throw WorkflowPhaseError("Storage admission requires an owned ordinary workflow execution.")
                }
            }
            if request.plan.trainingMode == .fresh {
                guard checkpointSnapshot == nil,request.checkpointParent == nil else {throw WorkflowPhaseError("Fresh training cannot use a checkpoint parent.")}
            } else {
                guard let snapshot=checkpointSnapshot,let selected=request.checkpointParent,
                      snapshot.reference.projectID == selected.projectID,snapshot.reference.directoryID.uuidString == selected.directoryID,
                      snapshot.runID == selected.runID,snapshot.requestIdentity == selected.requestIdentity,
                      snapshot.manifestSHA256 == selected.modelManifestSHA256 else {
                    throw WorkflowPhaseError("Checkpoint continuation requires the exact verified parent snapshot. No worker was started.")
                }
            }
            guard request.isCurrent(in: owner), !owner.isBusy, !owner.hasPendingDataEdits else { throw WorkflowPhaseError("Workflow changed or is busy; capture a new request.") }
            guard try request.runtimeEvidence.matchesCurrent(request.runtime) else { throw WorkflowPhaseError("Worker source changed after capture.") }
            if let binding=request.experiment {
                guard let inputs=request.experimentInputs,inputs.sources == sources,
                      inputs.manifestSHA256 == binding.inputsManifestSHA256,
                      inputs.directory.lastPathComponent == binding.groupID else {
                    throw WorkflowPhaseError("Experiment execution requires its exact captured group inputs.")
                }
                try inputs.verify()
            }
            guard (outputDestinationAccess?.paths ?? [:]) == request.outputDestinationPaths else {throw WorkflowPhaseError("Captured output destinations require their retained permission access.")}
            guard Set(sources.keys) == Set(request.plan.sourceAddresses) else { throw WorkflowPhaseError("Provide exactly the captured workflow's source bindings.") }
            guard let token = reservation.acquire() else { outputDestinationAccess?.close();outputDestinationAccess=nil;statusMessage = "Wait for the current workspace operation."; return false }
            reservationID = token; retainedSession = self
            owner.setExecutionBusy(true); started = true
            startedAt = Date()
            if let budget = request.executionBudget {
                budgetWindow = try acceptedBudgetWindow ?? ExecutionBudgetWindow(policy: budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            }
            if let storageLimitBytes {
                phase = .preparing; statusMessage = "Checking execution storage…"
                archiveCaptureTask = Task {
                    do {
                        let directory = self.directory, projectID = request.lease.projectID, accountingRoot = self.accountingRoot
                        let preparationID = preparationOperationID, trainingID = trainingOperationID
                        let hasFileCollections=sources.values.contains {$0.isFileCollection}
                        let sourceCounts = try sources.values.map{try WorkflowCapturedSourceArchive.declaredByteCount($0)}, checkpointFiles = checkpointSnapshot?.files ?? []
                        let staging = Task.detached(priority: .userInitiated) {
                            try Task.checkCancellation()
                            let workspace = directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                            guard directory.deletingLastPathComponent().lastPathComponent == projectID,
                                  directory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-runs" else {
                                throw WorkflowStorageAdmission.Failure.invalidScope
                            }
                            return try WorkflowExecutionStorageAdmission(workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
                                directoryID: directory.lastPathComponent, preparationOperationID: preparationID,
                                trainingOperationID: trainingID, sourceByteCounts: sourceCounts, checkpointFiles: checkpointFiles,hasFileCollections:hasFileCollections,
                                limitBytes: storageLimitBytes)
                        }
                        storageAdmission = try await withTaskCancellationHandler(operation: { try await staging.value }, onCancel: { staging.cancel() })
                        try Task.checkCancellation(); try checkPhase()
                        try createArchiveAndCapture()
                    } catch {
                        archiveCaptureTask = nil
                        finish(cancelled ? .cancelled : .failed, error: cancelled ? nil : error.localizedDescription)
                    }
                }
            } else { try createArchiveAndCapture() }
            return true
        } catch {
            if started { finish(.failed, error: error.localizedDescription) }
            else { outputDestinationAccess?.close();outputDestinationAccess=nil;statusMessage = error.localizedDescription }
            return false
        }
    }
    private func createArchiveAndCapture() throws {
        guard UUID(uuidString: directory.lastPathComponent) != nil else { throw WorkflowPhaseError("Execution archive directory requires a UUID identity.") }
        try FileManager.default.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        try DiagnosticFileIO.createExclusiveDirectory(root: directory.deletingLastPathComponent(), name: directory.lastPathComponent)
        ownsDirectory = true
        if let snapshot=checkpointSnapshot {
            phase = .preparing
            statusMessage="Capturing verified parent checkpoint…"
            console.append(phase:phase,message:statusMessage)
            let captureID = UUID(); archiveCaptureID = captureID
            archiveCaptureTask=Task { [weak self] in
                guard let self else {return}
                defer {if self.archiveCaptureID == captureID {self.archiveCaptureTask=nil;self.archiveCaptureID=nil}}
                do {
                    try Task.checkCancellation()
                    let directory=self.directory,mode=self.request.plan.trainingMode,settings=self.request.plan.settings
                    let work=Task.detached(priority:.userInitiated) {
                        let record=try snapshot.parentRecord()
                        if mode == .exactResume {
                            guard record.device == "cpu",settings.device == "cpu",settings.epochs>record.evidence.completedEpoch else {
                                throw WorkflowPhaseError("Exact resume requires a CPU parent, explicit CPU compute and a target epoch above the completed parent epoch.")
                            }
                        }
                        return (record,try snapshot.archive(to:directory))
                    }
                    let (record,files)=try await withTaskCancellationHandler(operation:{try await work.value},onCancel:{work.cancel()})
                    try Task.checkCancellation()
                    try self.checkPhase()
                    self.parentRecord=record
                    try await self.commitCapture(checkpointFiles:files)
                    try self.launchPreparation()
                } catch {
                    self.finish(self.cancelled ? .cancelled:.failed,error:self.cancelled ? nil:error.localizedDescription)
                }
            }
        } else {
            phase = .preparing
            statusMessage="Capturing workflow inputs…"
            console.append(phase:phase,message:statusMessage)
            let captureID = UUID(); archiveCaptureID = captureID
            archiveCaptureTask=Task { [weak self] in
                guard let self else {return}
                defer {if self.archiveCaptureID == captureID {self.archiveCaptureTask=nil;self.archiveCaptureID=nil}}
                do {
                    try Task.checkCancellation()
                    try await self.commitCapture()
                    try Task.checkCancellation()
                    try self.launchPreparation()
                } catch {
                    self.finish(self.cancelled ? .cancelled:.failed,error:self.cancelled ? nil:error.localizedDescription)
                }
            }
        }
    }
    private func commitCapture(checkpointFiles:[WorkflowArchiveFile]?=nil) async throws {
        let result=try await WorkflowArchiveCaptureWriter.writeInBackground(request:request,runID:runID,sources:sources,
            directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,checkpointFiles:checkpointFiles)
        captureReference=result.reference;sourceInputs=result.sourceInputs
    }

    func cancel() {
        guard started, !finished, !cancelled else { return }
        canvasProgress.clear();modelOperationProgress?.clear()
        cancelled = true; statusMessage = "Cancelling workflow execution…"
        console.append(phase:phase,level:.warning,message:statusMessage)
        archiveCaptureTask?.cancel()
        worker?.cancel()
    }
    private func checkGroupAuthority()throws {
        if let binding=request.experiment {
            if request.runtime.kind == .development {
                let evidence=request.runtimeEvidence.workerFiles
                guard evidence[".local/native-watchdog/lifetime_watchdog.dylib"] != nil,
                      evidence[".local/native-watchdog/lifetime_watchdog.json"] != nil else {
                    throw WorkflowPhaseError("Experiment execution requires captured development lifetime guard evidence. Rebuild the development app and capture a new run.")
                }
            }
            guard let groupOwner,let inputs=request.experimentInputs,groupOwner.metadata.groupID == binding.groupID,
                  binding.ordinal >= 1,binding.ordinal <= 32,inputs.directory.lastPathComponent == binding.groupID,
                  inputs.directory.deletingLastPathComponent().lastPathComponent == request.lease.projectID else {
                throw WorkflowPhaseError("Experiment execution requires its matching live group owner.")
            }
            try groupOwner.validateDirectory(inputs.directory)
        } else if groupOwner != nil {throw WorkflowPhaseError("Ordinary execution cannot borrow experiment group ownership.")}
    }
    private func remainingBudget() throws -> Double? {
        guard let budgetWindow else { return nil }
        do { return try budgetWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime) }
        catch ExecutionBudgetError.wallTimeLimitExceeded(let seconds) {
            failureCode = .wallTimeLimitExceeded
            throw ExecutionBudgetError.wallTimeLimitExceeded(seconds: seconds)
        }
    }
    private func checkPhase() throws {
        try checkGroupAuthority()
        _ = try remainingBudget()
        guard !cancelled, request.isCurrent(in: owner) else { throw WorkflowPhaseError("Workflow execution context is no longer current.") }
        guard try request.runtimeEvidence.matchesCurrent(request.runtime) else { throw WorkflowPhaseError("Worker source changed between workflow phases.") }
    }
    @Published private(set) var preparationObservations:PreparationObservationLedger?
    private func launchPreparation() throws {
        try checkPhase()
        let folder = directory.appendingPathComponent("preparation")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let graphBytes = try DataFlowCodec.encodeGraph(request.plan.dataflow)
        let bindingBytes = try WorkflowArchiveCodec.encode(sourceInputs)
        guard try WorkflowArchiveCodec.registryHash(owner.dataRegistry) == request.dataRegistrySHA256 else {throw WorkflowPhaseError("Captured data registry changed before preparation.")}
        let digests=Dictionary(uniqueKeysWithValues:sources.map {($0.key.localID,$0.value.sha256)})
        let identity=PreparationObservationIdentity(requestID:runID.uuidString,graphSHA256:WorkflowArchiveCodec.hash(graphBytes),sourceBindingsSHA256:WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(digests)),revision:String(request.document.revision))
        var kinds:[String:[String:String]]=[:]
        for node in request.plan.dataflow.nodes {
            guard let definition=owner.dataRegistry.operatorDefinition(id:node.operatorID) else {throw WorkflowPhaseError("Captured data operator is unavailable.")}
            kinds[node.id]=Dictionary(uniqueKeysWithValues:definition.executionOutputs(node.config,connectedInputs:Set(request.plan.dataflow.edges.filter{$0.to.node==node.id}.map{$0.to.port})).map {($0.id,$0.kind)})
        }
        preparationObservations=try PreparationObservationLedger(identity:identity,graph:request.plan.dataflow,outputKinds:kinds,expectedSourceDigests:digests)
        guard graphBytes.count <= 2_000_000, bindingBytes.count <= 1_000_000 else {
            throw WorkflowPhaseError("Preparation inputs exceed their captured control-file allowance.")
        }
        try DiagnosticFileIO.writeImmutable(graphBytes, root: folder, parts: ["request.json"], limit: 2_000_000)
        try DiagnosticFileIO.writeImmutable(bindingBytes, root: folder, parts: ["bindings.json"], limit: 1_000_000)
        if !request.outputDestinationPaths.isEmpty {
            let destinationBytes=try WorkflowArchiveCodec.encode(request.outputDestinationPaths)
            try DiagnosticFileIO.writeImmutable(destinationBytes,root:folder,parts:["destinations.json"],limit:1_000_000)
        }
        phase = .preparing
        try request.priorCache?.revalidate()
        var arguments = ["--execute-dataflow", folder.appendingPathComponent("request.json").path, "--retain-partial-dataflow",
            "--source-bindings", folder.appendingPathComponent("bindings.json").path,
            "--dataflow-live-previews","true","--dataflow-measurements",historyPolicy.metricsAndDiagnostics ? "true":"false","--dataflow-capture-outputs",historyPolicy.componentOutputs ? "true":"false","--dataflow-cache-policy", request.plan.preparationCachePolicy.rawValue, "--output", folder.appendingPathComponent("artifacts").path]
        arguments += ["--dataflow-required-nodes",String(decoding:try WorkflowArchiveCodec.encode(request.plan.requiredPreparationNodes),as:UTF8.self)]
        arguments += ["--dataflow-model-input-nodes",String(decoding:try WorkflowArchiveCodec.encode(request.plan.requiredPreparationNodes),as:UTF8.self)]
        if request.plan.dataflow.nodes.contains(where:{$0.operatorID == "kwip.data.outputDelimited" && APITableSettings.isExternal($0.config)}) {arguments += ["--defer-database-writes","true"]}
        if !request.outputDestinationPaths.isEmpty {arguments += ["--destination-bindings",folder.appendingPathComponent("destinations.json").path]}
        if let limit = request.preparationCacheLimitBytes {
            arguments += ["--dataflow-cache-limit-bytes", String(limit)]
        }
        if let prior = request.priorCache {
            arguments += ["--dataflow-reuse-manifest", prior.manifestURL.path,
                "--dataflow-reuse-manifest-sha256", prior.reference.preparationManifestSHA256]
        }
        try launchPhase(arguments)
    }
    private func launchTraining() throws {
        try checkPhase()
        guard let preparation else { throw WorkflowPhaseError("Preparation has no verified completed output.") }
        let folder = directory.appendingPathComponent("training")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let graphBytes = try WorkflowPhaseArtifacts.encode(request.plan.model.graph)
        guard graphBytes.count <= 2_000_000 else { throw WorkflowPhaseError("Training graph exceeds its captured control-file allowance.") }
        try DiagnosticFileIO.writeImmutable(graphBytes, root: folder, parts: ["request.json"], limit: 2_000_000)
        let settings = request.plan.settings
        if settings.gradientCapture != nil && request.sampleCapture?.savesDiagnosticDetails != false {
            do {
                let rows=try WorkflowFailedGradientEvidence.trainingRows(root:directory,selection:preparation)
                failedGradientCollector=WorkflowFailedGradientCollector(plan:request.plan,trainingRows:rows,initialEpoch:request.plan.trainingMode == .exactResume ? (parentRecord?.evidence.completedEpoch ?? 0):0)
            } catch {failedGradientNotice="Failed-gradient evidence unavailable: "+String(error.localizedDescription.prefix(1500))}
        }
        var arguments = ["--graph", folder.appendingPathComponent("request.json").path,
            "--output", folder.appendingPathComponent("artifacts").path, "--registry", request.runtime.registryURL.path,
            "--epochs", String(settings.epochs), "--seed", String(settings.seed), "--device", settings.device,
            "--batch-size", String(settings.batchSize), "--learning-rate", String(settings.learningRate)]
        if let policy=settings.trainingControl {arguments += ["--training-control-json",String(decoding:try JSONEncoder().encode(policy),as:UTF8.self)]}
        if let recipe=request.plan.optimizerRecipe {
            let bytes=try WorkflowPhaseArtifacts.encode(recipe.configuration.recipeValue)
            guard bytes.count<=8192 else {throw WorkflowPhaseError("Optimizer recipe exceeds its launch limit.")}
            arguments += ["--optimizer-recipe-json",String(decoding:bytes,as:UTF8.self)]
        }
        if let recipe=request.plan.schedulerRecipe {
            let bytes=try WorkflowPhaseArtifacts.encode(recipe.configuration.recipeValue)
            guard bytes.count<=8192 else {throw WorkflowPhaseError("Scheduler recipe exceeds its launch limit.")}
            arguments += ["--scheduler-recipe-json",String(decoding:bytes,as:UTF8.self)]
        }
        if let recipe = request.plan.initializationRecipe {
            let bytes = try WorkflowPhaseArtifacts.encode(recipe.configuration.recipeValue)
            guard bytes.count <= InitializationConfiguration.maximumRecipeBytes else { throw WorkflowPhaseError("Starting settings exceed their launch limit.") }
            arguments += ["--initialization-recipe-json", String(decoding: bytes, as: UTF8.self)]
        }
        arguments += try WorkflowGradientRunArguments.arguments(for: request)
        if let mapping = request.plan.bagInput {
            guard let registry = request.capturedModelRegistry else { throw WorkflowPhaseError("Bag training requires its captured model registry.") }
            arguments += try WorkflowPhaseArtifacts.preparedArguments(preparation, bagInput: mapping, model: request.plan.model.graph, registry: registry)
        } else { arguments += try WorkflowPhaseArtifacts.preparedArguments(preparation) }
        arguments += ["--training-progress-json",String(decoding:try JSONEncoder().encode(reportingPolicy),as:UTF8.self)]
        arguments += request.gradientArguments
        if let sampleCapture = request.sampleCapture {
            arguments += ["--sample-capture-json", String(decoding: try WorkflowPhaseArtifacts.encode(sampleCapture), as: UTF8.self)]
        }
        if let parentRecord {
            arguments += [request.plan.trainingMode == .exactResume ? "--resume-checkpoint":"--finetune-checkpoint",
                          directory.appendingPathComponent("continuation-parent/checkpoint.pt").path]
            if request.plan.trainingMode == .exactResume {metrics=parentRecord.metrics;liveMetrics=metrics}
        }
        if let provenance = request.plan.model.provenance {
            let bytes = try WorkflowPhaseArtifacts.encode(provenance)
            guard bytes.count <= 2_000_000 else { throw WorkflowPhaseError("Block provenance exceeds its size limit.") }
            let path = folder.appendingPathComponent("composite-source.json")
            try bytes.write(to: path, options: .atomic)
            arguments += ["--composite-source", path.path, "--composite-source-sha256", WorkflowPhaseArtifacts.hash(bytes)]
        }
        phase = .training
        try launchPhase(arguments)
    }
    private func launchPhase(_ arguments: [String]) throws {
        completion = nil; workerError = nil;launchReady=false;launchProtocolFailed=false;publishedLaunch=nil
        let token = UUID(); phaseToken = token
        statusMessage = phase == .preparing ? "Preparing captured sources…" : "Training on verified prepared partitions…"
        console.append(phase:phase,message:statusMessage)
        var invocation=arguments
        if let groupOwner,let binding=request.experiment {
            let published=try groupOwner.publishLaunch(projectID:request.lease.projectID,ordinal:binding.ordinal,runID:runID.uuidString,directoryID:directory.lastPathComponent,requestIdentity:request.requestIdentity,phase:phase == .preparing ? .preparation:.training,arguments:arguments,executionBudget:request.executionBudget,remainingSeconds:try remainingBudget())
            publishedLaunch=published
            invocation=["--kwip-supervise",published.groupDirectory.path,published.nonce,published.intentSHA256,published.ownershipSHA256]
        } else if let budgetWindow {
            let workspace = directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let envelope = WorkflowOperationRequest(operationID: phase == .preparing ? preparationOperationID : trainingOperationID,
                projectID: request.lease.projectID, runID: runID.uuidString,
                directoryID: directory.lastPathComponent, requestIdentity: request.requestIdentity,
                phase: phase == .preparing ? .preparation : .training, arguments: arguments,
                executionBudget: budgetWindow.policy,
                remainingSeconds: try remainingBudget())
            let authority = try WorkflowOperationOwnership.create(workspace: workspace, request: envelope)
            operationOwner = authority; operationRequest = envelope
            try storageAdmission?.bind(to: authority, phase: envelope.phase)
            try authority.publishRequest(envelope)
            let pin = try authority.prepareLaunch()
            invocation = ["--kwip-operation-supervise", authority.directory.path, try envelope.requestSHA256(), pin]
        }
        #if KWIPML_RECOVERY_UAT
        if let publishedLaunch, let binding = request.experiment {
            if phase == .preparing, RecoveryUATGate.shared.armed == .duringPreparation {
                try DiagnosticFileIO.writeImmutable(Data(publishedLaunch.nonce.utf8), root: directory,
                    parts: ["preparation", "recovery-uat-arm"], limit: 256)
            }
            let deferredInvocation = invocation
            if let pause = RecoveryUATGate.shared.hold(.beforeWorkerSpawn,
                identity: .init(groupID: binding.groupID, runID: runID.uuidString, ordinal: binding.ordinal, nonce: publishedLaunch.nonce),
                resume: { [weak self] in
                    guard let self, self.phaseToken == token, !self.finished else { return }
                    do {
                        try self.checkPhase()
                        let handle = try self.launch(self.request.runtime.executableURL, self.request.runtime.argumentPrefix + deferredInvocation,
                            self.request.runtime.workingDirectoryURL,
                            { [weak self] line in self?.consume(line, token: token) },
                            { [weak self] code, stderr in self?.exited(code, stderr: stderr, token: token) })
                        if self.phaseToken == token, !self.finished {
                            self.worker = handle
                            if self.cancelled || self.launchProtocolFailed { handle.cancel() }
                        }
                    } catch { self.finish(self.cancelled ? .cancelled : .failed, error: error.localizedDescription) }
                }, cancel: { [weak self] in
                    guard let self, self.phaseToken == token, !self.finished else { return }
                    self.cancel()
                    // This boundary precedes every process launch; there is no child exit to await.
                    self.exited(143, stderr: "Diagnostic pre-spawn pause cancelled", token: token)
                }) {
                worker = WorkflowWorkerHandle(cancel: { RecoveryUATGate.shared.cancel(pause) })
                return
            }
        }
        #endif
        let handle = try launch(request.runtime.executableURL, request.runtime.argumentPrefix + invocation,
            request.runtime.workingDirectoryURL, { [weak self] line in self?.consume(line, token: token) },
            { [weak self] code, stderr in self?.exited(code, stderr: stderr, token: token) })
        // A test launcher may synchronously exit; never overwrite the next phase's handle.
        if phaseToken == token, !finished { worker = handle; if cancelled || launchProtocolFailed { handle.cancel() } }
    }
    /// Qualified projection is based on the captured plan, never the current canvas.
    func preparationOutput(at address: WorkflowNodeAddress) -> [String: Any]? {
        guard address.domain == .data, request.plan.dataflow.nodes.contains(where: { $0.id == address.localID }) else { return nil }
        return previews[address.localID]
    }

    func typedPreparationOutput(at address: WorkflowNodeAddress) -> [String: JSONValue]? {
        guard address.domain == .data, request.plan.dataflow.nodes.contains(where: { $0.id == address.localID }) else { return nil }
        return typedPreviews[address.localID]
    }
    private struct TypedPreviewEvent: Decodable { let outputs: [String: JSONValue] }

    private func rejectLaunchProtocol(_ message:String) {
        guard !launchProtocolFailed else {return}
        launchProtocolFailed=true
        if workerError == nil {workerError=message}
        worker?.cancel()
    }
    private func consume(_ line: String, token: UUID) {
        let profile = InteractionProfiler.shared.begin(.workerEvent)
        defer { InteractionProfiler.shared.end(profile) }
        guard phaseToken == token, !finished, !cancelled else { return }
        guard !launchProtocolFailed else {return}
        guard line.utf8.count <= 1_000_000, let bytes = line.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              let name = event["event"] as? String else {
            if publishedLaunch != nil || operationRequest != nil {rejectLaunchProtocol("Worker emitted a malformed event before or after its ready acknowledgement.")}
            else {workerError="Worker emitted a malformed event."}
            return
        }
        if let operationRequest, let operationOwner {
            if name == "operationReady" {
                guard !launchReady, event["operationID"] as? String == operationRequest.operationID,
                      event["phase"] as? String == operationRequest.phase.rawValue,
                      event["requestSHA256"] as? String == operationOwner.metadata.requestSHA256,
                      event["ownershipSHA256"] as? String == operationOwner.ownershipSHA256 else {
                    rejectLaunchProtocol("Worker acknowledgement differs from its captured operation."); return
                }
                launchReady = true; return
            }
            if !launchReady && name != "error" {
                rejectLaunchProtocol("Worker emitted progress before its operation acknowledgement."); return
            }
        }
        if let publishedLaunch {
            if name == "launchReady" {
                guard !launchReady,workerError == nil,Set(event.keys) == ["event","nonce","phase"],event["nonce"] as? String == publishedLaunch.nonce,event["phase"] as? String == publishedLaunch.intent.phase.rawValue else {rejectLaunchProtocol("Worker launchReady acknowledgement is duplicated or does not match the ready token and phase.");return}
                launchReady=true;return
            }
            if !launchReady && name != "error" {rejectLaunchProtocol("Worker emitted progress before its matching ready acknowledgement.");return}
        }
        #if KWIPML_RECOVERY_UAT
        if name == "recoveryUATPaused" {
            guard phase == .preparing, let publishedLaunch, let binding = request.experiment,
                  launchReady, recoveryUATPreparationToken == nil,
                  event["stage"] as? String == "duringPreparation",
                  event["nonce"] as? String == publishedLaunch.nonce,
                  let nodeID = event["nodeID"] as? String,
                  request.plan.dataflow.nodes.contains(where: { $0.id == nodeID }) else {
                rejectLaunchProtocol("Diagnostic preparation pause did not match its owned launch."); return
            }
            recoveryUATPreparationToken = RecoveryUATGate.shared.hold(.duringPreparation,
                identity: .init(groupID: binding.groupID, runID: runID.uuidString, ordinal: binding.ordinal, nonce: publishedLaunch.nonce),
                resume: { [weak self] in
                    guard let self, self.phaseToken == token, !self.finished else { return }
                    self.recoveryUATPreparationToken = nil
                    do {
                        try self.checkPhase()
                        try DiagnosticFileIO.writeImmutable(Data(publishedLaunch.nonce.utf8), root: self.directory,
                            parts: ["preparation", "recovery-uat-resume"], limit: 256)
                    } catch { self.rejectLaunchProtocol("Diagnostic preparation resume failed: " + error.localizedDescription) }
                }, cancel: { [weak self] in
                    self?.recoveryUATPreparationToken = nil
                    self?.cancel() // Actual guarded worker is alive; normal exit acknowledgement remains mandatory.
                })
            if recoveryUATPreparationToken == nil { rejectLaunchProtocol("Diagnostic preparation pause was not armed.") }
            return
        }
        #endif
        if name == "dataflowNodeTiming" {
            do {
                guard phase == .preparing else { throw WorkflowArchiveError("Preparation timing arrived outside preparation.") }
                try preparationMeasurements.consume(bytes, allowedNodes: Set(request.plan.dataflow.nodes.map(\.id)), enabled: historyPolicy.metricsAndDiagnostics)
            } catch { rejectLaunchProtocol(error.localizedDescription) }
            return
        }
        if name == "dataflowNodeStarted",phase == .preparing,let node=event["nodeID"] as? String {
            guard let identity=preparationObservations?.identity,preparationObservations?.nodeStarted(node,identity:identity) == true else {rejectLaunchProtocol("Preparation started an unknown or duplicate stage.");return}
            if canvasProgress.start(.init(domain:.data,localID:node),stage:.preparing,context:canvasProgress.context,sequence:canvasProgress.sequence+1) {
                console.append(phase:phase,message:"Preparation started",component:.init(domain:.data,localID:node))
            }
        }
        if name == "started", phase == .training, let mapping=request.plan.bagInput {
            do {
                guard bagLiveContext == nil, let preparation,
                      let registry=request.capturedModelRegistry,
                      try WorkflowArchiveCodec.registryHash(registry) == request.modelRegistrySHA256,
                      let workerID=event["runID"] as? String, UUID(uuidString:workerID) != nil,
                      let device=event["device"] as? String, ["cpu","mps"].contains(device),
                      request.plan.settings.device == "auto" || request.plan.settings.device == device,
                      event["totalEpochs"] as? Int == request.plan.settings.epochs else {
                    throw WorkflowPhaseError("Bag start differs from the captured request or is duplicated.")
                }
                let validationContext:GraphValidationContext=request.plan.lossRecipe == nil ? .preparedWorkflow:.explicitObjective
                let requirement=try WorkflowMappingRequirements.bagInputRequirement(request.plan.model.graph,registry:registry,context:validationContext)
                let evidence=try WorkflowBagPreparedEvidence.preparedEvidence(root:directory,selection:preparation,mapping:mapping,requirement:requirement)
                let initial=request.plan.trainingMode == .exactResume ? (parentRecord?.evidence.completedEpoch ?? 0):0
                guard event["initialEpoch"] as? Int == initial,
                      event["trainingSamples"] as? Int == evidence.training.count,
                      event["validationSamples"] as? Int == evidence.validation.count else {
                    throw WorkflowPhaseError("Bag start differs from its verified whole prepared observations.")
                }
                let target:BagArtifactTarget
                if let recipe=request.plan.lossRecipe {
                    guard case .objective(let selected)=preparation,
                          selected.objectiveRecipe == recipe.configuration.recipeValue,
                          selected.targetMapping == recipe.targetMapping,
                          selected.scalarization == request.plan.objectiveScalarization else {
                        throw WorkflowPhaseError("Bag objective start differs from the captured target or scalarization.")
                    }
                    target = .objective(selected,recipe)
                } else {
                    guard case .legacy(let selected)=preparation,let column=request.plan.targetColumn,selected.targetColumn == column else {
                        throw WorkflowPhaseError("Bag classifier start differs from its captured target.")
                    }
                    target = .binary(column)
                }
                let context=WorkflowBagArtifactContext(requirement:requirement,mapping:mapping,preparation:evidence,
                    runID:workerID,device:device == "cpu" ? .cpu:.mps,
                    outputContract:try WorkflowMappingRequirements.outputContract(request.plan.model.graph,registry:registry),
                    target:target,policy:request.sampleCapture ?? .factory,explicitSampleCapture:request.sampleCapture != nil)
                if request.plan.lossRecipe != nil {
                    objectiveLiveEvaluationContext=try WorkflowBagArtifacts.objectiveEvaluationContext(context:context,expectedEpochs:request.plan.settings.epochs,trainingCount:WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:evidence.training.count,datasetRecipe:request.plan.datasetRecipe,loaderRecipe:request.plan.loaderRecipe,capturedBatchSize:request.plan.settings.batchSize,trainOrigin:request.plan.train,inlineRoles:request.plan.inlineDataLoadingRoles))
                }
                bagLiveContext=context
            } catch {rejectLaunchProtocol(error.localizedDescription);return}
        }
        if name == "started",phase == .training,request.plan.bagInput == nil,let recipe=request.plan.lossRecipe {
            do {
                guard objectiveLiveContext == nil,let preparation,case .objective(let selected)=preparation,
                      let registry=request.capturedModelRegistry,try WorkflowArchiveCodec.registryHash(registry) == request.modelRegistrySHA256,
                      selected.objectiveRecipe == recipe.configuration.recipeValue,selected.targetMapping == recipe.targetMapping,
                      selected.scalarization == request.plan.objectiveScalarization,
                      let workerID=event["runID"] as? String,UUID(uuidString:workerID) != nil,
                      let device=event["device"] as? String,["cpu","mps"].contains(device),request.plan.settings.device == "auto" || request.plan.settings.device == device,
                      event["totalEpochs"] as? Int == request.plan.settings.epochs else {throw WorkflowPhaseError("Objective start differs from the captured request or is duplicated.")}
                let counts=try WorkflowFailedGradientEvidence.preparedRowCounts(root:directory,selection:preparation)
                let initial=request.plan.trainingMode == .exactResume ? (parentRecord?.evidence.completedEpoch ?? 0):0
                guard event["initialEpoch"] as? Int == initial,event["trainingSamples"] as? Int == counts.trainingRows,event["validationSamples"] as? Int == counts.validationRows else {throw WorkflowPhaseError("Objective start differs from its verified prepared partitions.")}
                let input=try WorkflowMappingRequirements.inputContract(request.plan.model.graph,registry:registry,context:.explicitObjective)
                let output=try WorkflowMappingRequirements.outputContract(request.plan.model.graph,registry:registry)
                objectiveLiveContext=try WorkflowObjectiveArtifactContext(recipe:recipe,runID:workerID,inputContract:input,outputSampleShape:output.layout.sample.dimensions,trainingCount:WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:counts.trainingRows,datasetRecipe:request.plan.datasetRecipe,loaderRecipe:request.plan.loaderRecipe,capturedBatchSize:request.plan.settings.batchSize,trainOrigin:request.plan.train,inlineRoles:request.plan.inlineDataLoadingRoles),evaluationCount:counts.validationRows,expectedEpochs:request.plan.settings.epochs,scalarization:selected.scalarization,device:device == "cpu" ? .cpu:.mps,policy:request.sampleCapture ?? .factory,explicitSampleCapture:request.sampleCapture != nil)
                objectiveLiveEvaluationContext=try objectiveLiveContext?.evaluationContext()
            } catch {rejectLaunchProtocol(error.localizedDescription);return}
        }
        if name == "started",phase == .training,
           let progressRunID=event["runID"] as? String,UUID(uuidString:progressRunID) != nil,event["totalEpochs"] as? Int == request.plan.settings.epochs,
           let device=event["device"] as? String,["cpu","mps"].contains(device),
           request.plan.settings.device == "auto" || request.plan.settings.device == device {
            guard progressWorker==nil,let initial=event["initialEpoch"] as? Int,initial>=0,initial<request.plan.settings.epochs else {rejectLaunchProtocol("Duplicated or invalid training start.");return}
            do {
                guard let preparation else {throw WorkflowPhaseError("Training progress started without prepared data.")}
                let rows=try WorkflowFailedGradientEvidence.trainingRows(root:directory,selection:preparation)
                let retained:Int
                if request.plan.datasetRecipe != nil || request.plan.loaderRecipe != nil {
                    retained=try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:rows,datasetRecipe:request.plan.datasetRecipe,loaderRecipe:request.plan.loaderRecipe,capturedBatchSize:request.plan.settings.batchSize,trainOrigin:request.plan.train,inlineRoles:request.plan.inlineDataLoadingRoles)
                } else {retained=rows}
                guard retained>0 else {throw WorkflowPhaseError("Training progress has no prepared rows.")}
                progressEpochBatches=(retained-1)/request.plan.settings.batchSize+1
            } catch {rejectLaunchProtocol(error.localizedDescription);return}
            progressWorker=progressRunID;progressInitialEpoch=initial
            _=canvasProgress.start(request.train,stage:.training,context:canvasProgress.context,sequence:canvasProgress.sequence+1)
            modelOperationProgress = try? WorkflowModelOperationProgress(graph:request.plan.model.graph,origins:request.plan.model.origins,workerRunID:progressRunID)
            if let recipe=request.plan.schedulerRecipe {
                do {
                    guard schedulerWorkerRunID == nil,let preparation else {throw WorkflowPhaseError("Schedule started without prepared data or started twice.")}
                    let rows=try WorkflowFailedGradientEvidence.trainingRows(root:directory,selection:preparation)
                    let frame=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
                    let initial=request.plan.trainingMode == .exactResume ? (parentRecord?.evidence.completedEpoch ?? 0):0
                    guard frame["initialEpoch"] == .integer(Int64(initial)),frame["trainingSamples"] == .integer(Int64(rows)),rows>0 else {throw WorkflowPhaseError("Schedule start differs from captured training rows or continuation.")}
                    let retained=try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:rows,datasetRecipe:request.plan.datasetRecipe,loaderRecipe:request.plan.loaderRecipe,capturedBatchSize:request.plan.settings.batchSize,trainOrigin:request.plan.train,inlineRoles:request.plan.inlineDataLoadingRoles)
                    schedulerBatchesPerEpoch=(retained+request.plan.settings.batchSize-1)/request.plan.settings.batchSize
                    schedulerInitialTick=initial*(recipe.configuration.componentID == "scheduler_one_cycle_lr" ? schedulerBatchesPerEpoch:1)
                    schedulerWorkerRunID=progressRunID
                } catch {rejectLaunchProtocol(error.localizedDescription);return}
            }
        }
        if phase == .training {failedGradientCollector?.consume(bytes,event:event,name:name)}
        if name == "warning", phase == .training,
           let warning = PoolingExecutionWarning.decode(bytes, graph: request.plan.model.graph) {
            let origin = request.plan.model.origins[warning.nodeID]
            let component = origin.map { WorkflowNodeAddress(domain: .model, localID: $0.instancePath.first ?? $0.localNodeID) }
            console.append(phase: phase, level: .warning, message: warning.message, component: component)
        }
        if name == "error",workerError == nil {
            workerError = String((event["message"] as? String ?? "Worker failed").prefix(4000))
            if phase == .preparing,let frame=try? JSONDecoder().decode([String:JSONValue].self,from:bytes) {
                // Training preparation currently sends no node-label sidecar.
                stagedPreparationFailure=try? WorkflowPreparationFailure.detail(event:frame,graph:request.plan.dataflow,labels:[:],registry:owner.dataRegistry)
            }
        }
        if (phase == .preparing && name == "dataflowCompleted") || (phase == .training && name == "completed") {
            if completion != nil { workerError = "Worker emitted duplicate completion." }
            completion = event
        }
        if name.hasPrefix("model"),phase == .training,var feed=modelOperationProgress,feed.consume(event) {
            modelOperationProgress=feed
            let origin=(event["nodeID"] as? String).flatMap {request.plan.model.origins[$0]}
            let component=origin.map {WorkflowNodeAddress(domain:.model,localID:$0.instancePath.first ?? $0.localNodeID)}
            let label=origin.map {($0.instancePath+[$0.localNodeID]).joined(separator:" / ")} ?? "Model"
            console.append(phase:phase,message:"Validation capture · \(name) · \(label)",component:component)
        }
        if name == "dataflowPreview", phase == .preparing, let node = event["nodeID"] as? String,
           request.plan.dataflow.nodes.contains(where: { $0.id == node }), let outputs = event["outputs"] as? [String: Any] {
            previewBytes += bytes.count
            guard previewBytes <= 32_000_000 else { workerError = "Workflow previews exceed the retention limit."; worker?.cancel(); return }
            guard let typed = try? JSONDecoder().decode(TypedPreviewEvent.self, from: bytes) else {
                workerError = "Worker emitted invalid typed preview values."; return
            }
            if let execution=event["execution"] as? [String:Any],execution["nodeID"] as? String == node,
               let status=execution["status"] as? String,["executed","reused"].contains(status) {
                _=canvasProgress.complete(.init(domain:.data,localID:node),disposition:status == "reused" ? .reused:.computed,context:canvasProgress.context,sequence:canvasProgress.sequence+1)
            }

            guard let identity=preparationObservations?.identity,let frame=try? JSONDecoder().decode([String:JSONValue].self,from:bytes),preparationObservations?.consume(frame,identity:identity) == true else {rejectLaunchProtocol("Preparation output differs from its captured stage, source or ports.");return}
            typedPreviews[node] = typed.outputs
            previews[node] = outputs
            console.append(phase:phase,message:"Captured preparation output",component:.init(domain:.data,localID:node))
        }
        if name == "schedulerTick",phase == .training {
            do {
                guard let recipe=request.plan.schedulerRecipe,let runID=schedulerWorkerRunID,
                      schedulerRecordCount<SchedulerConfiguration.maximumTicks,bytes.count<=SchedulerConfiguration.maximumTraceBytes-schedulerEventBytes else {throw WorkflowPhaseError("Unexpected or oversized live schedule evidence.")}
                let frame=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
                guard Set(frame.keys)==["event","schemaVersion","runID","scheduler","record"],frame["schemaVersion"] == 1,frame["runID"] == .string(runID),let selected=frame["scheduler"],try SchedulerConfiguration.decodeRecipe(selected)==recipe.configuration,let record=frame["record"] else {throw WorkflowPhaseError("Live schedule identity differs from its captured recipe.")}
                let tick=schedulerInitialTick+schedulerRecordCount+1
                try WorkflowLifecycleVerifier.validateSchedulerRecord(record,configuration:recipe.configuration,expectedTick:tick)
                guard case .object(let values)=record else {throw WorkflowPhaseError("Live schedule record is malformed.")}
                let isBatch=recipe.configuration.componentID == "scheduler_one_cycle_lr"
                let epoch=isBatch ? (tick-1)/schedulerBatchesPerEpoch+1:tick
                guard epoch<=request.plan.settings.epochs,values["epoch"] == .integer(Int64(epoch)),!isBatch || values["batch"] == .integer(Int64((tick-1)%schedulerBatchesPerEpoch+1)) else {throw WorkflowPhaseError("Live schedule timing differs from its training groups.")}
                schedulerRecordCount+=1;schedulerEventBytes+=bytes.count
                let used=values["usedRate"]?.doubleValue ?? 0,next=values["nextRate"]?.doubleValue ?? 0
                let position=isBatch ? "Epoch \(epoch) · group \((tick-1)%schedulerBatchesPerEpoch+1)":"Epoch \(epoch)"
                let terminal=values["nextRateUnused"] == true ? " · next rate unused (cycle complete)":""
                console.append(phase:phase,message:position+" · rate used \(used) · next \(next)"+terminal,component:recipe.source)
            } catch {rejectLaunchProtocol(error.localizedDescription);return}
        }
        if name == "trainingProgress",phase == .training {
            do {
                guard bytes.count<=2048,let worker=progressWorker else {throw WorkflowPhaseError("Progress arrived without its training start.")}
                let row=try TrainingProgressPoint.decode(bytes)
                guard row.epochBatches==progressEpochBatches,row.batches==(row.epoch-progressInitialEpoch-1)*row.epochBatches+row.epochBatch else {throw WorkflowPhaseError("Progress counters differ from captured training batches.")}

                try progressLedger.append(row,worker:worker,maximumEpoch:request.plan.settings.epochs,initialEpoch:progressInitialEpoch)
                if row.kind=="validation" {
                    guard let metric=metrics.last,metric.epoch==row.epoch,metric.trainLoss==row.trainingLoss,metric.validationLoss==row.validationLoss,metric.accuracy==row.accuracy else {throw WorkflowPhaseError("Live progress differs from canonical epoch measurement.")}
                }
                _=progressPublication;progressChanges.send(progressLedger.points)
            } catch {rejectLaunchProtocol(error.localizedDescription);return}
        }
        if name == "epoch", phase == .training {
            let epoch:Int
            if request.plan.lossRecipe != nil {
                do {
                    guard let context=objectiveLiveEvaluationContext,metrics.count<10_000 else {throw WorkflowPhaseError("Objective metric arrived without its captured start or exceeds its bound.")}
                    let value=try JSONDecoder().decode(JSONValue.self,from:bytes)
                    let row=try WorkflowObjectiveArtifacts.epoch(value,expectedEpoch:(metrics.last?.epoch ?? 0)+1,context:context)
                    epoch=row.epoch
                    metrics.append(.init(epoch:row.epoch,trainLoss:row.trainLoss,validationLoss:row.validationLoss,accuracy:row.accuracy))
                } catch {rejectLaunchProtocol(error.localizedDescription);return}
            } else {
                guard let observed=try? TrainingEpochObservation.decode(bytes), observed.epoch <= request.plan.settings.epochs,
                      metrics.count < 10_000, observed.epoch == (metrics.last?.epoch ?? 0) + 1 else {
                    workerError = "Worker metric sequence is invalid."; return
                }
                epoch=observed.epoch
                metrics.append(.init(epoch:epoch,trainLoss:observed.trainLoss,validationLoss:observed.validationLoss,accuracy:observed.accuracy))
            }
            if MetricSampling.shouldPublish(epoch: epoch, totalEpochs: request.plan.settings.epochs) {
                _ = metricPublication
                metricChanges.send(metrics)
            }
        }
    }
    private func exited(_ code: Int32, stderr: String, token: UUID) {
        guard phaseToken == token, !finished else { return }
        #if KWIPML_RECOVERY_UAT
        if let pause = recoveryUATPreparationToken {
            RecoveryUATGate.shared.dismiss(pause)
            recoveryUATPreparationToken = nil
        }
        #endif
        phaseToken = nil; worker = nil
        operationOwner?.close(); operationOwner = nil
        if cancelled { finish(.cancelled, error: nil); return }
        do {
            if code == 124, let budget = request.executionBudget {
                failureCode = .wallTimeLimitExceeded
                throw ExecutionBudgetError.wallTimeLimitExceeded(seconds: budget.maximumRunSeconds)
            }
            if code == 125 {
                failureCode = .resourceBudgetFailed
                throw WorkflowPhaseError(stderr.isEmpty ? "The worker could not maintain its resource budget. Inspect the run log before retrying." : String(stderr.prefix(4000)))
            }
            if (publishedLaunch != nil || operationRequest != nil) && !launchReady && workerError == nil {workerError="Worker exited without its matching ready acknowledgement."}
            if code == 1, phase == .training, launchReady, !launchProtocolFailed, workerError != nil, completion == nil {
                publishFailedGradientEvidence()
            }
            guard code == 0, workerError == nil, let completion else {
                throw WorkflowPhaseError(workerError ?? (stderr.isEmpty ? "Worker exited without a completed result." : String(stderr.prefix(4000))))
            }
            try checkPhase()
            if phase == .preparing {
                let verified = try WorkflowPhaseArtifacts.preparation(request: request, root: directory, completion: completion,modelInputNodes:request.plan.requiredPreparationNodes)
                preparation = verified.selection; preparedHash = verified.selection.manifestSHA256
                artifactHashes.merge(verified.hashes) { _, new in new }
                artifactFiles = verified.artifactFiles
                statusMessage = "Verifying saved tables and preparing training…"
                let captureID = UUID(); archiveCaptureID = captureID
                archiveCaptureTask = Task {
                    defer {if archiveCaptureID == captureID {archiveCaptureTask=nil;archiveCaptureID=nil}}
                    do {
                        let raw=try DiagnosticFileIO.read(root:directory,parts:["preparation","artifacts","manifest.json"],limit:2_000_000)
                        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:raw)
                        capturedTables=try WorkflowCapturedTables.admit(bytes:raw,expectedSHA256:verified.selection.manifestSHA256,
                            directory:directory.appendingPathComponent("preparation/artifacts"),graph:request.plan.dataflow,
                            runtime:request.runtime,storageLimitBytes:storageLimitBytes)
                        var receipts:[[String:JSONValue]]
                        if case .array(let values)=manifest["outputs"] {
                            receipts=try values.map {guard case .object(let receipt)=$0 else {throw WorkflowPhaseError("Malformed saved table receipt.")};return receipt}
                        } else {receipts=[]}
                        try await WorkflowDataPreparationSession.verifySavedOutputs(receipts,graph:request.plan.dataflow,destinations:request.outputDestinationPaths)
                        try Task.checkCancellation();try checkPhase()

                        guard let identity=preparationObservations?.identity,preparationObservations?.completeVerified(identity:identity,allOutputsVerified:true) == true else {throw WorkflowPhaseError("Verified preparation omitted declared live observations.")}
                        for node in request.plan.dataflow.nodes where node.operatorID == "kwip.data.outputDelimited" && APITableSettings.isExternal(node.config) {
                            guard let databaseWrite,let artifact=capturedTables?.artifacts[.init(address:.init(domain:.data,localID:node.id),port:"table")],let window=budgetWindow else {throw WorkflowPhaseError("Database output requires its native broker and complete captured table.")}
                            let receipt=try await databaseWrite(node,artifact,runID.uuidString,window)
                            receipts.append(receipt.merging(["nodeID":.string(node.id)]){_,v in v})
                            savedOutputs=receipts
                        }
                        savedOutputs=receipts
                        if let storageAdmission {
                            let admission=Task.detached(priority:.userInitiated) {try Task.checkCancellation();try storageAdmission.transitionToTraining()}
                            try await withTaskCancellationHandler(operation:{try await admission.value},onCancel:{admission.cancel()})
                        }
                        try Task.checkCancellation();try checkPhase();try launchTraining()
                    } catch {finish(cancelled ? .cancelled:.failed,error:cancelled ? nil:error.localizedDescription)}
                }
            } else {
                _=canvasProgress.start(request.train,stage:.verifying,context:canvasProgress.context,sequence:canvasProgress.sequence+1)
                guard let preparation else { throw WorkflowPhaseError("Prepared partition identity is missing.") }
                let controlValue=completion["trainingControl"].map {try? JSONDecoder().decode(JSONValue.self,from:JSONSerialization.data(withJSONObject:$0))} ?? nil
                let controlReport=try TrainingControlReport.verify(controlValue,policy:request.plan.settings.trainingControl,epochLimit:request.plan.settings.epochs,losses:metrics.map(\.validationLoss))
                guard metrics.count == (controlReport?.completedEpoch ?? request.plan.settings.epochs) else { throw WorkflowPhaseError("Worker did not deliver every completed epoch observation.") }
                let verified = try WorkflowPhaseArtifacts.training(request: request, root: directory, preparation: preparation, completion: completion, lifecycleParent: parentRecord?.evidence)
                if let context=verified.objectiveContext {
                    guard let live=objectiveLiveContext,live.runID == context.runID,live.recipe == context.recipe,live.inputContract == context.inputContract,live.outputSampleShape == context.outputSampleShape,live.trainingCount == context.trainingCount,live.evaluationCount == context.evaluationCount,live.device == context.device,live.scalarization == context.scalarization,
                          metrics.count == verified.metrics.count,zip(metrics,verified.metrics).allSatisfy({pair in pair.0.epoch == pair.1.epoch && pair.0.trainLoss == pair.1.trainLoss && pair.0.validationLoss == pair.1.validationLoss && pair.0.accuracy == pair.1.accuracy}) else {throw WorkflowPhaseError("Live objective history differs from the verified completed artifacts.")}
                }
                if let context=verified.bagContext {
                    guard let live=bagLiveContext,live.requirement == context.requirement,live.mapping == context.mapping,
                          live.preparation == context.preparation,live.runID == context.runID,live.device == context.device,
                          live.outputContract == context.outputContract,live.target == context.target,
                          live.policy == context.policy,live.explicitSampleCapture == context.explicitSampleCapture,
                          metrics.count == verified.metrics.count,
                          zip(metrics,verified.metrics).allSatisfy({pair in pair.0.epoch == pair.1.epoch && pair.0.trainLoss == pair.1.trainLoss && pair.0.validationLoss == pair.1.validationLoss && pair.0.accuracy == pair.1.accuracy}) else {
                        throw WorkflowPhaseError("Live bag identity or epoch history differs from the verified completed artifacts.")
                    }
                }
                if let context=verified.objectiveEvaluationContext {
                    guard let live=objectiveLiveEvaluationContext,live.runID == context.runID,live.recipe == context.recipe,
                          live.outputSampleShape == context.outputSampleShape,live.trainingCount == context.trainingCount,
                          live.evaluationCount == context.evaluationCount,live.expectedEpochs == context.expectedEpochs,
                          live.device == context.device,live.scalarization == context.scalarization,
                          live.policy == context.policy,live.explicitSampleCapture == context.explicitSampleCapture,
                          metrics.count == verified.metrics.count,
                          zip(metrics,verified.metrics).allSatisfy({pair in pair.0.epoch == pair.1.epoch && pair.0.trainLoss == pair.1.trainLoss && pair.0.validationLoss == pair.1.validationLoss && pair.0.accuracy == pair.1.accuracy}) else {
                        throw WorkflowPhaseError("Live objective context or history differs from the verified completed artifacts.")
                    }
                }
                // Legacy receipts may omit samples entirely. Never publish unverified ones.
                let capturedModel: WorkflowVerifiedModelCaptures?
                if completion["previews"] != nil {
                    let event = try JSONDecoder().decode([String:JSONValue].self, from: JSONSerialization.data(withJSONObject:completion))
                    capturedModel = try WorkflowVerifiedModelCaptures(event:event,graph:request.plan.model.graph,
                        origins:request.plan.model.origins,sampleCapture:request.sampleCapture,
                        project:request.plan.model.provenance?.sourceCompositeDocument,objectiveContext:verified.objectiveContext,capturedRegistry:request.capturedModelRegistry,bagContext:verified.bagContext,objectiveEvaluationContext:verified.bagContext == nil ? nil:verified.objectiveEvaluationContext)
                } else { capturedModel = nil }
                if verified.bagContext == nil && verified.objectiveContext == nil && (completion["predictionInput"] != nil || completion["predictionRows"] != nil) {
                    let preview:PredictionPreview
                    if request.plan.model.graph.namedInputs != nil {
                        let status=request.sampleCapture?.predictionRows == false ? "omittedByPolicy":"captured"
                        let envelope:[String:Any]=["schemaVersion":4,"captureStatus":status,"evaluationCount":completion["evaluationCount"]!,"predictionInput":completion["predictionInput"]!,"predictionRows":completion["predictionRows"]!]
                        preview=try PredictionPreview.decodeNamed(JSONSerialization.data(withJSONObject:envelope),graph:request.plan.model.graph)
                    } else {
                        preview=try PredictionPreview.decode(JSONSerialization.data(withJSONObject: completion))
                        let contract = try WorkflowMappingRequirements.inputContract(request.plan.model.graph, registry: owner.modelRegistry)
                        guard preview.dtype == contract.dtype.rawValue, preview.sampleShape == contract.layout.sample.dimensions else {throw WorkflowPhaseError("Prediction preview differs from the captured input contract.")}
                    }
                    guard verified.predictionPreview == nil || verified.predictionPreview == preview else {
                        throw WorkflowPhaseError("Live predictions differ from the saved artifact.")
                    }
                    predictionPreview = verified.predictionPreview ?? preview
                }
                bagPredictionPreview=verified.bagPredictionPreview
                objectivePredictionPreview=verified.objectivePredictionPreview
                objectiveEvaluation=verified.objectiveEvaluation
                routingTables=verified.routingTables
                predictionTable=verified.predictionTable
                trainingControlReport=verified.trainingControl
                performanceCapture=verified.performanceCapture
                derivativeResults=verified.derivativeResults
                modelCaptures = capturedModel
                trainedHash = verified.manifestSHA256; artifactHashes.merge(verified.hashes) { _, new in new }
                artifactFiles += verified.artifactFiles; workerRunID = verified.workerRunID
                finish(.completed, error: nil)
            }
        } catch { finish(.failed, error: error.localizedDescription) }
    }
    private func publishFailedGradientEvidence() {
        guard let collector=failedGradientCollector else {return}
        if let notice=collector.notice {failedGradientNotice=notice;return}
        do {
            guard let pin=preparedHash,let value=try collector.result(workflowRunID:runID.uuidString,requestSHA256:request.requestIdentity,preparationSHA256:pin) else {return}
            let bytes=try WorkflowFailedGradientEvidence.wire(value)
            try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:["diagnostics","failed-gradients.json"],limit:FailedGradientCaptureCodec.maximumArtifactBytes)
            let file=WorkflowArchiveFile(path:WorkflowFailedGradientEvidence.path,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
            failedGradientDescriptor = .init(file:file,workerRunID:value.context.workerRunID)
            artifactFiles.append(file);artifactHashes[file.path]=file.sha256
        } catch {failedGradientNotice="Failed-gradient evidence unavailable: "+String(error.localizedDescription.prefix(1500))}
    }
    private func finish(_ status: WorkflowExecutionPhase, error: String?) {
        if status == .failed {preparationObservations?.finish(.failed)}
        else if status == .cancelled {preparationObservations?.finish(.cancelled)}
        guard !finished else { return }
        defer { retainedSession = nil }
        operationOwner?.close(); operationOwner = nil
        metricPublication.cancel();progressPublication.cancel()
        liveTrainingProgress=progressLedger.points;trainingProgressObserved=progressLedger.observed
        liveMetrics = metrics // Final observations are visible before terminal publication.
        canvasProgress.clear();modelOperationProgress?.clear()
        let previous = phase
        preparationFailure = status == .failed && previous == .preparing && !launchProtocolFailed ? stagedPreparationFailure:nil
        finished = true; phaseToken = nil; worker = nil; phase = status
        statusMessage = error ?? (status == .completed ? "Workflow completed." : "Workflow cancelled; no completed model result published.")
        if status != .completed,!request.outputDestinationPaths.isEmpty {
            console.append(phase:status,level:.warning,message:"A completed preparation phase may already have saved table files. Cancelling or failing training does not undo those files; inspect its saved-table receipts before retrying.")
        }
        console.append(phase:status,level:status == .failed ? .error : status == .cancelled ? .warning : .info,message:statusMessage)
        var recordError: String?
        if ownsDirectory {
            do {
                let start = startedAt ?? Date()
                let finish = max(Date(), start)
                let suffix = "… [truncated to archive error limit]"
                let recordedError: String? = error.map { value in
                    guard value.utf8.count > 16_000 else { return value }
                    var prefix = String(decoding: value.utf8.prefix(16_000 - suffix.utf8.count - 3), as: UTF8.self)
                    while prefix.utf8.count + suffix.utf8.count > 16_000 { prefix.removeLast() }
                    return prefix + suffix
                }
                if status == .completed,historyPolicy.metricsAndDiagnostics,!progressLedger.points.isEmpty,let worker=progressWorker {
                    let archive=TrainingProgressArchive(schemaVersion:1,requestIdentity:request.requestIdentity,workerRunID:worker,policy:reportingPolicy,observed:progressLedger.observed,points:progressLedger.points)
                    try archive.validate(request:request.requestIdentity,worker:worker,metrics:metrics)
                    let data=try JSONEncoder().encode(archive)
                    try DiagnosticFileIO.writeImmutable(data,root:directory,parts:["diagnostics","training-progress.json"],limit:TrainingProgressArchive.maximumBytes)
                    let file=WorkflowArchiveFile(path:TrainingProgressArchive.path,bytes:data.count,sha256:WorkflowArchiveCodec.hash(data))
                    artifactFiles.append(file);artifactHashes[file.path]=file.sha256
                }
                let published = status == .completed ? artifactFiles : artifactFiles.filter { $0.path.hasPrefix("preparation/artifacts/") || (status == .failed && $0.path == failedGradientDescriptor?.file.path) }
                let diagnostic = status == .failed && previous == .training ? failedGradientDescriptor:nil
                let diagnosticNotice = status == .failed && previous == .training ? failedGradientNotice:nil
                let receipt = WorkflowArchiveOutcome(schemaVersion:diagnostic != nil || diagnosticNotice != nil ? 3:2,projectID: request.lease.projectID,
                    directoryID: directory.lastPathComponent, runID: runID.uuidString, requestIdentity: request.requestIdentity,
                    status: status, terminalPhase: previous, startedAt: WorkflowArchiveCodec.timestamp(start), finishedAt: WorkflowArchiveCodec.timestamp(finish),
                    artifacts: published.sorted { $0.path < $1.path }, capture: captureReference,
                    preparationManifestSHA256: preparedHash, modelManifestSHA256: status == .completed ? trainedHash : nil,
                    workerRunID: status == .completed ? workerRunID : nil,
                    error: status == .failed ? (recordedError?.isEmpty == false ? recordedError : "Workflow execution failed.") : recordedError,
                    failureCode: status == .failed ? failureCode : nil,failedGradientEvidence:diagnostic,failedGradientNotice:diagnosticNotice)
                let bytes = try WorkflowArchiveCodec.encodeOutcome(receipt)
                try DiagnosticFileIO.writeImmutable(bytes, root: directory, parts: ["outcome.json"], limit: 256_000)
            } catch { recordError = error.localizedDescription }
        }
        outputDestinationAccess?.close(); outputDestinationAccess=nil
        storageAdmission?.close(); storageAdmission = nil
        if let reservationID {
            self.reservationID = nil
            // Only this owner is released; the reservation adapter must compare its token.
            owner.setExecutionBusy(false); reservation.release(reservationID)
        }
        onTerminal(.init(runID: runID, request: request, status: status,
            failedPhase: status == .failed ? previous : nil, sources: sources, directory: directory,
            preparationManifestSHA256: preparedHash, modelManifestSHA256: status == .completed ? trainedHash : nil,
            artifactHashes: artifactHashes, error: error, recordError: recordError))
    }
}
