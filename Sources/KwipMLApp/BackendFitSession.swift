import Foundation
import Combine
import KwipMLCore

/// One retained preparation→backend operation. Uses existing capture, deadline,
/// process ownership and result verification rather than direct Process launches.
@MainActor final class BackendFitSession: ObservableObject {
    @Published private(set) var status = "Ready"
    @Published private(set) var isRunning = false
    @Published private(set) var output: URL?
    @Published private(set) var manifest: [String: JSONValue]?
    @Published private(set) var messages: [String] = []
    private(set) var capturedTables:WorkflowCapturedTables?
    private(set) var preparationMeasurements = PreparationPhaseMeasurements()
    let runID = UUID()
    private(set) var historyOperationID:String?
    let bindingRevision: UInt64
    @Published private(set) var phase: WorkflowExecutionPhase = .idle
    @Published private(set) var requestIdentity = ""
    @Published private(set) var preparationOutputs: [String: [String: JSONValue]] = [:]
    @Published private(set) var preparationFailure:DataFlowFailureDetail?
    private var stagedPreparationFailure:DataFlowFailureDetail?
    private var workerFailureMessage:String?
    var preparationFailureIsCurrent:Bool {
        guard phase == .failed,!isRunning,preparationFailure != nil,
              owner.document.matchesCapturedConstruction(document),bindings.revision == bindingRevision,
              workspace.activeProjectID == owner.projectID,workspace.workflowSession?.owner === owner,
              workspace.workflowSession?.currentBackendFit === self,
              capturedModelBindingRevision == nil || workspace.workflowSession?.modelLoader.bindings.revision == capturedModelBindingRevision,
              workspace.workflowSession?.sourceProvider.observedSourcesMatch(preparationOutputs) == true else {return false}
        return true
    }
    private var measurementCaptureEnabled = false
    private var previewBytes = 0
    let plan: WorkflowEstimatorPlan
    let document: WorkflowDocument
    private let workspace: WorkspaceStore
    private let owner: WorkflowAuthoringStore
    private let bindings: WorkflowPrivateSourceBindings
    private let history: BackendFitHistoryStore
    var onCompleted:(()->Void)?
    private let documentModel:ModelAssetReference?
    private var capturedModelBindingRevision:UInt64?
    private let runtime: RuntimeLocation
    private let acceptedBudgetWindow:ExecutionBudgetWindow?
    private let policy: ResolvedExecutionPreferences
    private let launcher: WorkflowWorkerLauncher
    private var task: Task<Void, Never>?
    private var worker: WorkflowWorkerHandle?
    private var cancelled = false
    private var reservationID: UUID?
    private var retained: BackendFitSession?
    init(workspace: WorkspaceStore, owner: WorkflowAuthoringStore, bindings: WorkflowPrivateSourceBindings, history: BackendFitHistoryStore,
         plan: WorkflowEstimatorPlan, runtime: RuntimeLocation, policy: ResolvedExecutionPreferences,
         launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,acceptedBudgetWindow:ExecutionBudgetWindow?=nil,documentModel:ModelAssetReference?=nil) {
        self.documentModel=documentModel
        self.acceptedBudgetWindow=acceptedBudgetWindow
        self.workspace = workspace; self.owner = owner; self.bindings = bindings; self.history = history; self.plan = plan
        self.runtime = runtime; self.policy = policy; self.launcher = launcher; self.document = owner.document; self.bindingRevision = bindings.revision
    }
    func start() throws {
        guard !isRunning, !workspace.isBusy, !owner.isBusy, !owner.hasPendingInspectorEdits, owner.document == document else { throw WorkflowArchiveError("Use the idle current estimator before running.") }
        let historyPolicy=owner.runHistoryPolicy
        measurementCaptureEnabled = historyPolicy.metricsAndDiagnostics
        let lease = owner.lease(), revision = bindings.revision
        var paths: [WorkflowNodeAddress: String] = [:]
        for address in plan.sourceAddresses {
            guard let path = bindings.path(for: address) else { throw WorkflowArchiveError("Choose a CSV for \(address.localID) before running.") }
            paths[address] = path
        }
        let window = try acceptedBudgetWindow ?? ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        workspace.isPreparingData = true
        workspace.cancelDataPreparation = { [weak self] in self?.cancel() }
        reservationID = workspace.dataPreparationReservationID
        requestIdentity = try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(document))
        owner.setExecutionBusy(true); phase = .preparing; isRunning = true; retained = self; status = "Capturing sources…"
        RunFeedbackController.shared.begin(runID)
        task = Task { [self] in
            var sources: WorkflowRunSourceCapture?
            defer {
                if let feedbackState = RunFeedbackController.State(rawValue: phase.rawValue) { RunFeedbackController.shared.transition(runID, state: feedbackState) }
                sources?.remove(); worker = nil; task = nil; isRunning = false
                if owner.isCurrent(lease) { owner.setExecutionBusy(false) }
                if workspace.dataPreparationReservationID == reservationID { workspace.isPreparingData = false; workspace.cancelDataPreparation = nil }
                retained = nil
                if phase == .completed {onCompleted?()}
            }
            do {
                sources = try await WorkflowRunSourceCapture.capture(paths: paths, workspace: workspace.workspaceDirectory, accountingRoot: workspace.applicationDirectory,
                    sourcePermissions: workspace.sourceAccess.directory, storageLimitBytes: policy.storageLimitBytes,fileConfigurations:DataFlowSourceOperators.fileConfigurations(in:document).filter {paths[$0.key] != nil})
                try current(lease, revision: revision)
                guard let sources else { throw WorkflowFileLeaseError.unsafePath }
                status = plan.configuration.isPrediction ? "Preparing data for the frozen model…" : "Preparing training and validation data…"
                let graph = try DataFlowCodec.encodeGraph(plan.dataflow)
                let labels = try WorkflowArchiveCodec.encode(Dictionary(uniqueKeysWithValues:plan.dataflow.nodes.map {node -> (String,String) in
                    let name=document.nodes.first {$0.address == .init(domain:.data,localID:node.id)}?.name.trimmingCharacters(in:.whitespacesAndNewlines) ?? ""
                    return (node.id,name.isEmpty ? (owner.dataRegistry.operatorDefinition(id:node.operatorID)?.title ?? node.id):String(name.prefix(200)))
                }))
                let captureRoot=workspace.workspaceDirectory
                let ocrInputs=plan.dataflow.nodes.contains(where:{$0.operatorID==OCRConfiguration.operatorID}) ? try await Task.detached(priority:.utility) {try OCRPackStore.inputs(workspace:captureRoot)}.value : []
                guard let modelStore=workspace.workflowSession?.modelLoader else {throw WorkflowArchiveError("The model workspace is unavailable.")}
                capturedModelBindingRevision=modelStore.bindings.revision
                let preparationModelInputs=try await PrebuiltPreparationModelCapture.inputs(document:document,store:modelStore,nodeIDs:Set(plan.dataflow.nodes.map(\.id)))
                try current(lease,revision:revision)
                let makeInputs: (URL) throws -> [WorkflowPreviewCapture.Input] = { directory in
                    var inputs: [String: WorkflowPreviewCapture.Input] = [:], mapped: [String: String] = [:]
                    for (address, source) in sources.sources {
                        if source.isFileCollection {let staged=source.stagedInputs();for input in staged.files {inputs[input.path]=input};mapped[address.localID]=directory.appendingPathComponent(staged.binding).path}
                        else {
                            let name = "input/sources/\(source.sha256)/source.\(WorkflowSourceFilename.suffix(source.url))"
                            inputs[name] = .file(path: name, source: source.url, expectedSHA256: source.sha256)
                            mapped[address.localID] = directory.appendingPathComponent(name).path
                        }
                    }
                    return [.bytes(path: "input/graph.json", data: graph), .bytes(path: "input/node-labels.json", data: labels),
                            .bytes(path: "input/bindings.json", data: try WorkflowArchiveCodec.encode(mapped))] + inputs.keys.sorted().compactMap { inputs[$0] } + ocrInputs + preparationModelInputs
                }
                let operation = UUID().uuidString
                let directory = workspace.workspaceDirectory.appendingPathComponent("workflow-operations/\(plan.projectID)/\(operation)")
                let inputBound = try WorkflowPreviewCapture.byteCountBound(inputs: makeInputs(directory), purpose: .preparation, workload: policy.budget.resources?.workloadPolicy)
                let admission = try policy.storageLimitBytes.map { try WorkflowCaptureStorageReservation(workspace: workspace.workspaceDirectory, accountingRoot: workspace.applicationDirectory,
                    projectID: plan.projectID, operationID: operation, purpose: .preparation, maximumCaptureBytes: inputBound, limitBytes: $0, workload: policy.budget.resources?.workloadPolicy) }
                defer { admission?.close() }
                let requiredNodes=String(decoding:try WorkflowArchiveCodec.encode(Array(Set([plan.training.node,plan.validation.node])).sorted()),as:UTF8.self)
                let modelInputOptions = plan.configuration.isPrebuilt || plan.configuration.isDocumentClassification ? [] : ["--dataflow-model-input-nodes",requiredNodes]
                let prepared = try await executePhase(.preparation, window: window, operationID: operation, storageAdmission: admission, inputs: makeInputs, arguments: { directory in
                    modelInputOptions + ["--dataflow-required-nodes",requiredNodes,"--dataflow-live-previews","true","--dataflow-measurements",historyPolicy.metricsAndDiagnostics ? "true":"false","--dataflow-capture-outputs",historyPolicy.componentOutputs ? "true":"false","--execute-dataflow", directory.appendingPathComponent("input/graph.json").path,
                     "--source-bindings", directory.appendingPathComponent("input/bindings.json").path,
                     "--dataflow-node-labels", directory.appendingPathComponent("input/node-labels.json").path,
                     "--dataflow-cache-policy", "off", "--output", directory.appendingPathComponent("artifacts").path]
                })
                try current(lease, revision: revision)
                phase = .training
                switch plan.configuration.operation {
                case "predict": status = "Predicting with frozen \(plan.configuration.backendID) estimator…"
                case "evaluate": status = "Evaluating frozen \(plan.configuration.backendID) estimator…"
                default: status = "Fitting \(plan.configuration.backendID) estimator…"
                }
                let fitID = UUID().uuidString
                historyOperationID=fitID
                try owner.captureHistoryDecision(id:fitID,policy:historyPolicy)
                let fitDirectory = workspace.workspaceDirectory.appendingPathComponent("workflow-operations/\(plan.projectID)/\(fitID)")
                let input = try BackendFitCapturedInput.make(plan: plan, prepared: prepared, directory: fitDirectory, capturePredictions: policy.sampleCapture.predictionRows && historyPolicy.componentOutputs,documentModel:documentModel.flatMap{model in model.files.first(where:{$0.path=="state.json"}).map{(model.directory(root:workspace.workspaceDirectory),$0)}},modelPreparation:documentModel?.files.first(where:{$0.path=="preparation.json"}))
                requestIdentity = WorkflowArchiveCodec.hash(input.requestData)
                let result = try await executePhase(.backendFit, window: window, operationID: fitID, inputs: { _ in input.files }, arguments: { _ in input.arguments })
                try current(lease, revision: revision)
                let bytes = try DiagnosticFileIO.read(root: result, parts: ["manifest.json"], limit: 2_000_000)
                let parsed = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
                // The immutable receipt is the acceptance boundary. A later Cancel
                // must not relabel a durable, accepted model as rejected.
                let capturedDirectory=prepared
                let capturedBytes=try DiagnosticFileIO.read(root:capturedDirectory,parts:["manifest.json"],limit:2_000_000)
                capturedTables=try WorkflowCapturedTables.admit(bytes:capturedBytes,expectedSHA256:input.preparationManifestSHA256,
                    directory:capturedDirectory,graph:plan.dataflow,runtime:runtime,storageLimitBytes:policy.storageLimitBytes)
                try await history.record(operationID: fitID, workerRequest: input.requestData, document: document)
                output = result; manifest = parsed; phase = .completed
                switch plan.configuration.operation {
                case "predict": status = "Completed · verified frozen-model table predictions."
                case "evaluate": status = "Completed · verified frozen-model evaluation."
                default: status = "Completed · verified \(plan.configuration.backendID) model and portable inference package."
                }
            } catch {
                let failedPreparation = phase == .preparing
                phase = cancelled || Task.isCancelled ? .cancelled : .failed
                preparationFailure = phase == .failed && failedPreparation ? stagedPreparationFailure:nil
                status = cancelled || Task.isCancelled ? "Cancelled · no model result accepted." : error.localizedDescription
                messages.append(status)
            }
        }
    }
    private func current(_ lease: WorkflowEditorLease, revision: UInt64) throws {
        try Task.checkCancellation()
        guard !cancelled, capturedModelBindingRevision == nil || workspace.workflowSession?.modelLoader.bindings.revision == capturedModelBindingRevision, owner.isCurrent(lease), owner.document == document, bindings.revision == revision,
              workspace.activeProjectID == plan.projectID, workspace.dataPreparationReservationID == reservationID else { throw WorkflowArchiveError("Estimator or source authority changed while fitting.") }
    }
    private func executePhase(_ mode: WorkflowCapturedToolExecution.Mode, window: ExecutionBudgetWindow, operationID: String,
                       storageAdmission: WorkflowCaptureStorageReservation? = nil,
                       inputs: @escaping (URL) throws -> [WorkflowPreviewCapture.Input], arguments: @escaping (URL) -> [String]) async throws -> URL {
        try Task.checkCancellation()
        workerFailureMessage=nil
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let handle = try await WorkflowCapturedToolExecution.launch(mode: mode, workspace: workspace.workspaceDirectory, accountingRoot: workspace.applicationDirectory,
                        projectID: plan.projectID, runtime: runtime, window: window, operationID: operationID,
                        storageLimitBytes: mode == .backendFit ? policy.storageLimitBytes : nil,
                        storageAdmission: storageAdmission, inputs: inputs, arguments: arguments, launcher: launcher,
                        onLine: { [weak self] line in
                            self?.consume(line)
                        }, onExit: { code, message, result in
                            if code == 0, let result { continuation.resume(returning: result) }
                            else { continuation.resume(throwing: WorkflowArchiveError(message.isEmpty ? (self.workerFailureMessage ?? "Backend phase exited with code \(code).") : message)) }
                        })
                    worker = handle
                    if cancelled { handle.cancel() }
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    private func consume(_ line: String) {
        guard let bytes = line.data(using: .utf8), bytes.count <= 1_000_000,
              let event = try? JSONDecoder().decode([String: JSONValue].self, from: bytes), let name = event["event"]?.stringValue else { return }
        if name == "dataflowNodeTiming" {
            do { try preparationMeasurements.consume(bytes, allowedNodes: Set(plan.dataflow.nodes.map(\.id)), enabled: measurementCaptureEnabled) }
            catch { cancel(); status = error.localizedDescription }
            return
        }
        if name == "dataflowPreview", let node = event["nodeID"]?.stringValue,
           plan.dataflow.nodes.contains(where: { $0.id == node }), case .object(let values) = event["outputs"] {
            previewBytes += bytes.count
            if previewBytes > 32_000_000 { cancel(); status = "Preparation previews exceed the retained display limit."; return }
            preparationOutputs[node] = values; messages.append("Captured preparation output · " + node)
        } else if name == "error" {
            let message=String((event["message"]?.stringValue ?? "Worker reported an error.").prefix(4000))
            messages.append(message)
            if workerFailureMessage == nil {
                workerFailureMessage=message
                if phase == .preparing {
                    stagedPreparationFailure=try? WorkflowPreparationFailure.detail(event:event,graph:plan.dataflow,
                        labels:WorkflowPreparationFailure.labels(document:document,registry:owner.dataRegistry),registry:owner.dataRegistry)
                }
            }
        }
        else if name == "backendStarted" { messages.append(plan.configuration.isPrediction ? "Applying the frozen model to captured rows without refitting.":"Fitting the captured training partition; validation remains held out.") }
        else if name == "backendCompleted" { messages.append("Worker completed; verifying its artifacts and successful exit.") }
        if messages.count > 200 { messages.removeFirst(messages.count - 200) }
    }
    func predictionMetadata(for endpoint:WorkflowEndpoint)->[String:JSONValue]? {
        if ["profiles","confusion"].contains(endpoint.port),predictionArtifact(for:endpoint) != nil,case .object(let metadata)=manifest?[endpoint.port+"Metadata"] {return metadata}
        guard predictionArtifact(for:endpoint) != nil,case .array(let columns)=manifest?["predictionColumns"],case .object(let metrics)=manifest?["evaluation"] else {return nil}
        return ["columns":.array(columns),"rowCount":metrics["count"] ?? 0,"partition":manifest?["predictionPartition"] ?? .string("unassigned")]
    }
    func predictionArtifact(for endpoint:WorkflowEndpoint)->TableViewQuerySession.Artifact? {
        guard phase == .completed,endpoint.address==plan.estimator,["predictions","profiles","confusion"].contains(endpoint.port),let output,case .array(let files)=manifest?["artifacts"],let entry=files.first(where:{if case .object(let e)=$0 {return e["path"] == .string(endpoint.port+".json")};return false}),case .object(let file)=entry,let pin=file["sha256"]?.stringValue else {return nil}
        return .init(indexURL:output.appendingPathComponent(endpoint.port+".json"),indexSHA256:pin,sourceSHA256:pin,inputKind:"table")
    }
    func verifiedOutput() async throws -> URL {
        guard phase == .completed, let output, let entry = history.entries.first(where: { $0.artifactDirectory == output }) else { throw WorkflowArchiveError("No accepted backend output is available.") }
        return try await history.verify(entry).artifactDirectory
    }
    func cancel() { guard isRunning, phase != .completed else { return }; RunFeedbackController.shared.transition(runID, state: .cancelling); cancelled = true; task?.cancel(); worker?.cancel(); status = "Cancelling estimator operation…" }
}

struct BackendFitCapturedInput {
    let files: [WorkflowPreviewCapture.Input]
    let arguments: [String]
    let requestData: Data
    let preparationManifestSHA256:String
    static func make(plan: WorkflowEstimatorPlan, prepared: URL, directory: URL, capturePredictions: Bool = false,documentModel:(URL,WorkflowArchiveFile)?=nil,modelPreparation:WorkflowArchiveFile?=nil) throws -> Self {
        let raw = try DiagnosticFileIO.read(root: prepared, parts: ["manifest.json"], limit: 2_000_000)
        struct Manifest: Decodable { let graphDigest: String; let artifacts: [WorkflowArchiveFile] }
        let manifest = try JSONDecoder().decode(Manifest.self, from: raw)
        let pin = WorkflowArchiveCodec.hash(raw)
        let manifestFile = WorkflowArchiveFile(path: "manifest.json", bytes: raw.count, sha256: pin)
        try WorkflowCompletedPreparationInventory.verify(data: raw, files: manifest.artifacts + [manifestFile], prefix: "")
        let selection = PreparedTrainingSelection(manifestPath: directory.appendingPathComponent("input/preparation/manifest.json").path,
            manifestSHA256: pin, graphDigest: manifest.graphDigest, trainingNode: plan.training.node, trainingPort: plan.training.port,
            validationNode: plan.validation.node, validationPort: plan.validation.port,
            featureColumns: plan.configuration.featureColumns, targetColumn: plan.configuration.targetColumn)
        let request: Data
        if plan.configuration.backendID == "sklearn" {
            var documentConfiguration:[String:JSONValue]=["task":.string(DocumentClassificationConfiguration.task),"settings":.object(plan.configuration.settings),"operation":.string(plan.configuration.operation)]
            if let threshold=plan.configuration.reviewThreshold {documentConfiguration["reviewThreshold"] = .number(threshold)}
            let config=plan.configuration.isPrebuilt ? ["task":JSONValue.string("prebuilt"),"algorithm":.string(plan.configuration.algorithm ?? ""),"operation":.string(plan.configuration.operation),"settings":.object(plan.configuration.settings)]:plan.configuration.isDocumentClassification ? documentConfiguration:plan.configuration.settings
            var value = try SklearnPreparedWorkerRequest(configuration: config, selection: selection,model:documentModel.map{.init(path:directory.appendingPathComponent("input/model-state.json").path,sha256:$0.1.sha256,preparationPath:modelPreparation.map{_ in directory.appendingPathComponent("input/model-preparation.json").path},preparationSHA256:modelPreparation?.sha256)})
            value.capturePredictions = capturePredictions
            request = try value.encoded()
        } else if plan.configuration.backendID == "tensorflow", let model = plan.model {
            var value = try TensorFlowPreparedWorkerRequest(graph: model.graph, settings: plan.configuration.settings, selection: selection)
            value.capturePredictions = capturePredictions
            request = try WorkflowArchiveCodec.encode(value)
        } else { throw WorkflowArchiveError("Unsupported estimator backend or missing model.") }
        var files: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/backend-request.json", data: request)] + (manifest.artifacts + [manifestFile]).map {
            .file(path: "input/preparation/\($0.path)", source: prepared.appendingPathComponent($0.path), expectedSHA256: $0.sha256)
        }
        if let documentModel {files.append(.file(path:"input/model-state.json",source:documentModel.0.appendingPathComponent("state.json"),expectedSHA256:documentModel.1.sha256))}
        if let modelPreparation {
            guard let documentModel,modelPreparation.path=="preparation.json" else {throw WorkflowFileLeaseError.unsafePath}
            files.append(.file(path:"input/model-preparation.json",source:documentModel.0.appendingPathComponent(modelPreparation.path),expectedSHA256:modelPreparation.sha256))
        }
        return .init(files: files, arguments: ["--fit-backend", directory.appendingPathComponent("input/backend-request.json").path,
            "--backend-request-sha256", WorkflowArchiveCodec.hash(request), "--output", directory.appendingPathComponent("artifacts").path], requestData: request,preparationManifestSHA256:pin)
    }
}
