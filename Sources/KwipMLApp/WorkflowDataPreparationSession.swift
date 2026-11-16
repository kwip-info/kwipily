import Foundation
import Combine
import KwipMLCore

/// Data-only execution on the same authoring owner, reservation, isolated worker,
/// Results provider and persistent History used by the universal canvas.
@MainActor final class WorkflowDataPreparationSession:ObservableObject {
    var onTerminal:((Bool)->Void)?
    private(set) var preparationMeasurements = PreparationPhaseMeasurements()
    let runID=UUID()
    private(set) var historyOperationID:String?
    let document:WorkflowDocument
    let graph:DataFlowDocument
    let bindingRevision:UInt64
    let destinationRevision:UInt64
    @Published private(set) var isRunning=false
    @Published private(set) var phase:WorkflowExecutionPhase = .idle
    @Published private(set) var statusMessage="Ready to prepare the connected table."
    @Published private(set) var requestIdentity=""
    @Published private(set) var output:URL?
    @Published private(set) var messages:[String]=[]
    @Published private(set) var preparationOutputs:[String:[String:JSONValue]]=[:]
    @Published private(set) var executionCoverage: WorkflowPreparationCoverage.Receipt?
    @Published private(set) var savedOutputs:[[String:JSONValue]]=[]
    @Published private(set) var canvasProgress:WorkflowCanvasProgress?
    @Published private(set) var observations:PreparationObservationLedger?
    @Published private(set) var preparationFailure:DataFlowFailureDetail?
    private var stagedPreparationFailure:DataFlowFailureDetail?
    private var workerFailureMessage:String?
    var preparationFailureIsCurrent:Bool {
        guard phase == .failed,!isRunning,preparationFailure != nil,
              owner.document.matchesCapturedConstruction(document),bindings.revision == bindingRevision,speechModelBindingsMatch,
              destinations.revision == destinationRevision,workspace.activeProjectID == owner.projectID,
              workspace.workflowSession?.owner === owner,workspace.workflowSession?.ownsPreparation(self) == true,
              workspace.workflowSession?.sourceProvider.observedSourcesMatch(preparationOutputs) == true,
              graph.nodes.allSatisfy({populationMatches($0.id)}) else {return false}
        return true
    }
    private let workspace:WorkspaceStore
    private let owner:WorkflowAuthoringStore
    private let bindings:WorkflowPrivateSourceBindings
    private let destinations:DataOutputDestinationProvider
    private let history:WorkflowDataPreparationHistory
    private let runtime:RuntimeLocation
    private let sampling: PreparationSampling
    private let samplingJSON: String
    private let completeConsumerAncestors: Set<String>
    private let acceptedBudgetWindow:ExecutionBudgetWindow?
    private let policy:ResolvedExecutionPreferences
    private let launcher:WorkflowWorkerLauncher
    private var task:Task<Void,Never>?
    private var launchTask:Task<Void,Never>?
    private var worker:WorkflowWorkerHandle?
    private var reservationID:UUID?
    private var cancelled=false
    private var protocolFailure:String?
    private var measurementCaptureEnabled = false
    private var previewBytes=0
    // Complete View indexes are admitted only after the durable History receipt.
    private var capturedAt: Date?
    private var capturedImageResolver: FileImageArtifactResolver?
    private var tableViewArtifacts: [String: TableViewQuerySession.Artifact] = [:]
    private var tableViewSessions: [WorkflowEndpoint: TableViewQuerySession] = [:]
    private var capturedTableArtifacts: [WorkflowEndpoint: TableViewQuerySession.Artifact] = [:]
    private var speechBindingRevision:UInt64?
    private var speechModelBindingsMatch:Bool {speechBindingRevision == nil || workspace.workflowSession?.modelLoader.bindings.revision == speechBindingRevision}
    private var retained:WorkflowDataPreparationSession?
    private let executionDocument:WorkflowDocument
    private let requiredOutputNodes:Set<String>
    private let predictionInputs:[WorkflowNodeAddress:TableViewQuerySession.Artifact]
    init(workspace:WorkspaceStore,owner:WorkflowAuthoringStore,bindings:WorkflowPrivateSourceBindings,destinations:DataOutputDestinationProvider,history:WorkflowDataPreparationHistory,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,launcher:@escaping WorkflowWorkerLauncher=WorkflowExecutionSession.launchWorker,acceptedBudgetWindow:ExecutionBudgetWindow?=nil,predictionPlan:WorkflowPredictionPreparation?=nil,predictionInputs:[WorkflowNodeAddress:TableViewQuerySession.Artifact]=[:],allowTrainingTerminals:Bool=false,requiredOutputNodes:Set<String>=[],preparationDocument:WorkflowDocument?=nil)throws {
        self.acceptedBudgetWindow=acceptedBudgetWindow
        self.sampling = try workspace.preparationSampling.validated()
        self.samplingJSON = try workspace.preparationSampling.transportJSON()
        self.workspace=workspace;self.owner=owner;self.bindings=bindings;self.destinations=destinations;self.history=history;self.runtime=runtime;self.policy=policy;self.launcher=launcher
        self.requiredOutputNodes=requiredOutputNodes;self.executionDocument=preparationDocument ?? predictionPlan?.document ?? owner.document;self.predictionInputs=predictionInputs
        if executionDocument.nodes.contains(where:{if case .dataOperator(let id,_)=($0.payload) {return [SpeechTranscriptionConfiguration.operatorID,AudioClassificationConfiguration.operatorID,TypedDecisionConfiguration.operatorID,"kwip.data.preprocess"].contains(id)};return false}) {speechBindingRevision=workspace.workflowSession?.modelLoader.bindings.revision}
        document=owner.document;graph=try WorkflowDataPreparationPlan.compile(executionDocument,registry:owner.dataRegistry,allowTrainingTerminals:allowTrainingTerminals);bindingRevision=bindings.revision;destinationRevision=destinations.revision
        var complete=Set(graph.nodes.filter { ["kwip.data.view","kwip.data.outputDelimited"].contains($0.operatorID) }.map(\.id))
        var pending=Array(complete)
        while let node=pending.popLast() {
            for edge in graph.edges where edge.to.node == node && complete.insert(edge.from.node).inserted {pending.append(edge.from.node)}
        }
        completeConsumerAncestors=complete
    }
    func start()throws {
        guard !isRunning,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,owner.document==document,workspace.activeProjectID==owner.projectID,workspace.workflowSession?.owner === owner else {throw DataFlowContractError("Use the idle current canvas and apply or discard pending edits before running data preparation.")}
        let historyPolicy=owner.runHistoryPolicy
        measurementCaptureEnabled = historyPolicy.metricsAndDiagnostics
        let lease=owner.lease(),window=try acceptedBudgetWindow ?? ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
        var paths:[WorkflowNodeAddress:String]=[:]
        for node in graph.nodes where DataFlowSourceOperators.contains(node.operatorID) {
            let address=WorkflowNodeAddress(domain:.data,localID:node.id)
            if predictionInputs[address] != nil {continue}
            guard let path=bindings.path(for:address) else {throw DataFlowContractError("Choose a source file for \(node.id) before preparing data.")};paths[address]=path
        }
        let databaseNodes=graph.nodes.filter {$0.operatorID=="kwip.data.outputDelimited" && APITableSettings.isExternal($0.config)}
        let databaseStore=databaseNodes.isEmpty ? nil:destinations.databaseOutputs
        var databaseAuthority:[String:DatabaseOutputStore.Authority]=[:]
        for node in databaseNodes {databaseAuthority[node.id]=try databaseStore!.captureAuthority(node)}
        let sinks=Set(graph.nodes.filter {$0.operatorID=="kwip.data.outputDelimited" && DatabaseOutputSettings.requiresLocalFolder($0.config)}.map {WorkflowNodeAddress(domain:.data,localID:$0.id)})
        let access=try destinations.acquire(sinks)
        requestIdentity=try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(document))
        workspace.isPreparingData=true;workspace.cancelDataPreparation={ [self] in cancel() };reservationID=workspace.dataPreparationReservationID
        owner.setExecutionBusy(true);phase = .preparing;isRunning=true;retained=self
        canvasProgress = .init(context:runID,allowedNodes:Set(graph.nodes.map {.init(domain:.data,localID:$0.id)}),edges:document.edges)
        RunFeedbackController.shared.begin(runID);statusMessage="Capturing data sources…"
        task=Task { [self] in
            var captured:WorkflowRunSourceCapture?
            defer {
                captured?.remove();access.close();worker=nil;launchTask=nil;task=nil;isRunning=false;canvasProgress?.clear()
                if owner.isCurrent(lease) {owner.setExecutionBusy(false)}
                if workspace.dataPreparationReservationID==reservationID {workspace.isPreparingData=false;workspace.cancelDataPreparation=nil}
                if let feedback=RunFeedbackController.State(rawValue:phase.rawValue) {RunFeedbackController.shared.transition(runID,state:feedback)}
                retained=nil;let completion=onTerminal;onTerminal=nil;completion?(phase == .completed)
            }
            do {
                captured=try await WorkflowRunSourceCapture.capture(paths:paths,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,sourcePermissions:workspace.sourceAccess.directory,storageLimitBytes:policy.storageLimitBytes,fileConfigurations:DataFlowSourceOperators.fileConfigurations(in:document).filter {paths[$0.key] != nil})
                try current(lease);guard let captured else {throw WorkflowFileLeaseError.unsafePath}
                let bytes=try DataFlowCodec.encodeGraph(graph)
                let labels=try WorkflowArchiveCodec.encode(Dictionary(uniqueKeysWithValues:executionDocument.nodes.filter {$0.address.domain == .data}.map {node -> (String,String) in
                    let operatorID:String;if case .dataOperator(let id,_)=node.payload {operatorID=id}else {operatorID=""}
                    return (node.address.localID,node.name.isEmpty ? (owner.dataRegistry.operatorDefinition(id:operatorID)?.title ?? node.address.localID):String(node.name.prefix(200)))
                }))
                let outputKinds=Dictionary(uniqueKeysWithValues:graph.nodes.map {node in (node.id,Dictionary(uniqueKeysWithValues:(owner.dataRegistry.operatorDefinition(id:node.operatorID)?.executionOutputs(node.config,connectedInputs:Set(graph.edges.filter{$0.to.node==node.id}.map{$0.to.port})) ?? []).map {($0.id,$0.kind)}))})
                let identity=PreparationObservationIdentity(requestID:runID.uuidString,graphSHA256:WorkflowArchiveCodec.hash(bytes),sourceBindingsSHA256:try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(Dictionary(uniqueKeysWithValues:captured.sources.map {($0.key.localID,$0.value.sha256)}+predictionInputs.map {($0.key.localID,$0.value.sourceSHA256)}))),revision:String(document.revision))
                observations=try PreparationObservationLedger(identity:identity,graph:graph,outputKinds:outputKinds,expectedSourceDigests:Dictionary(uniqueKeysWithValues:captured.sources.map {($0.key.localID,$0.value.sha256)}+predictionInputs.map {($0.key.localID,$0.value.sourceSHA256)}))
                let fileIDs=Set(graph.nodes.filter {$0.operatorID=="kwip.data.outputDelimited" && !APITableSettings.isExternal($0.config)}.map(\.id))
                guard let modelStore=self.workspace.workflowSession?.modelLoader else {throw WorkflowArchiveError("The model workspace is unavailable.")}
                let preparationModelInputs=try await PrebuiltPreparationModelCapture.inputs(document:executionDocument,store:modelStore)
                let audioInputs=try await AudioClassificationModelCapture.inputs(document:executionDocument,store:modelStore)
                let decisionInputs=try await DecisionModelCapture.inputs(document:executionDocument,store:modelStore)
                try current(lease)
                let speechInputs=try await SpeechModelCapture.inputs(document:executionDocument,store:modelStore)
                try current(lease)
                let ocrInputs = graph.nodes.contains(where:{$0.operatorID=="kwip.data.extractText"}) ? try await Task.detached(priority:.utility) {try OCRPackStore.inputs(workspace:self.workspace.workspaceDirectory)}.value : []
                let sources=captured.sources,destinationPaths=access.paths.filter {fileIDs.contains($0.key)},predictions=predictionInputs
                let inputs:(URL)throws->[WorkflowPreviewCapture.Input] = {directory in
                    var files:[String:WorkflowPreviewCapture.Input]=[:],mapped:[String:String]=[:]
                    for (address,source) in sources {
                        let staged=source.stagedInputs();for input in staged.files {files[input.path]=input};mapped[address.localID]=directory.appendingPathComponent(staged.binding).path
                    }
                    for (address,artifact) in predictions {
                        let path="input/captured-tables/"+artifact.sourceSHA256+".json"
                        files[path] = .file(path:path,source:artifact.indexURL,expectedSHA256:artifact.sourceSHA256)
                        mapped[address.localID]=directory.appendingPathComponent(path).path
                    }
                    var controls:[WorkflowPreviewCapture.Input]=[.bytes(path:"input/graph.json",data:bytes),.bytes(path:"input/node-labels.json",data:labels),.bytes(path:"input/bindings.json",data:try WorkflowArchiveCodec.encode(mapped))]
                    if !destinationPaths.isEmpty {controls.append(.bytes(path:"input/destinations.json",data:try WorkflowArchiveCodec.encode(destinationPaths)))}
                    return controls+files.keys.sorted().compactMap {files[$0]}+ocrInputs+speechInputs+audioInputs+decisionInputs+preparationModelInputs
                }
                let operation=UUID().uuidString,directory=workspace.workspaceDirectory.appendingPathComponent("workflow-operations/\(owner.projectID)/\(operation)")
                historyOperationID=operation
                try owner.captureHistoryDecision(id:operation,policy:historyPolicy)
                let bound=try WorkflowPreviewCapture.byteCountBound(inputs:inputs(directory),purpose:.preparation,workload:policy.budget.resources?.workloadPolicy)
                let admission=try policy.storageLimitBytes.map {try WorkflowCaptureStorageReservation(workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,projectID:owner.projectID,operationID:operation,purpose:.preparation,maximumCaptureBytes:bound,limitBytes:$0,workload:policy.budget.resources?.workloadPolicy)}
                defer {admission?.close()}
                statusMessage="Preparing connected data components…"
                let result=try await execute(window:window,operation:operation,admission:admission,inputs:inputs,destinations:destinationPaths,historyPolicy:historyPolicy)
                try current(lease)
                if let protocolFailure {throw WorkflowArchiveError(protocolFailure)}
                guard Set(preparationOutputs.keys)==Set(graph.nodes.map(\.id)),var ledger=observations,ledger.completeVerified(identity:identity,allOutputsVerified:true) else {throw WorkflowArchiveError("Preparation did not verify every expected output.")}
                let manifestBytes=try DiagnosticFileIO.read(root:result,parts:["manifest.json"],limit:2_000_000)
                let manifest=try JSONDecoder().decode([String:JSONValue].self,from:manifestBytes)
                var receipts:[[String:JSONValue]]=[]
                if case .array(let raw)=manifest["outputs"] {for item in raw {guard case .object(let receipt)=item else {throw WorkflowArchiveError("Saved-output receipt is malformed.")};receipts.append(receipt)}}
                try await Self.verifySavedOutputs(receipts,graph:graph,destinations:destinationPaths)
                try current(lease)
                let admittedViews = try TableViewArtifactReference.parse(manifest: manifest, directory: result,
                    expectedNodes: Set(graph.nodes.filter { $0.operatorID == "kwip.data.view" }.map(\.id)))
                let entry=try await history.record(operationID:operation,document:executionDocument)
                guard entry.manifestSHA256 == WorkflowArchiveCodec.hash(manifestBytes) else { throw WorkflowArchiveError("Preparation manifest changed before View admission.") }
                capturedImageResolver = try FileImageArtifactResolver(directory: entry.artifactDirectory, verifiedArtifacts: entry.artifacts)
                capturedTableArtifacts = try TableViewArtifactReference.capturedTables(manifest:manifest,directory:entry.artifactDirectory,expectedNodes:Set(graph.nodes.map(\.id)))
                tableViewArtifacts = admittedViews
                executionCoverage = entry.coverage
                capturedAt = entry.completedAt
                for node in databaseNodes {
                    try current(lease)
                    guard let store=databaseStore,let authority=databaseAuthority[node.id],let artifact=capturedTableArtifacts[.init(address:.init(domain:.data,localID:node.id),port:"table")] else {throw WorkflowArchiveError("External output requires its complete verified prepared table.")}
                    statusMessage="Sending table output · \(node.id)"
                    do {let receipt=try await store.write(node:node,artifact:artifact,preparationID:operation,window:window,authority:authority);receipts.append(receipt.merging(["nodeID":.string(node.id)]){_,new in new})}
                    catch {
                        let labels=WorkflowPreparationFailure.labels(document:document,registry:owner.dataRegistry)
                        stagedPreparationFailure=try? DataFlowFailureDetail.decode(WorkflowArchiveCodec.encode(["schemaVersion":JSONValue.integer(1),"code":.string("database_output_failed"),"nodeID":.string(node.id),"nodeLabel":labels[node.id].map(JSONValue.string) ?? .null,"operatorID":.string(node.operatorID),"componentTitle":.string("Table Output"),"inputRowCount":.null,"cause":.string(String(error.localizedDescription.prefix(500))),"suggestion":.string("Inspect this component’s delivery/write receipt. Unknown or partial delivery must be checked and acknowledged before another write.")]),graph:graph,labels:labels,titles:[node.operatorID:"Table Output"])
                        savedOutputs=receipts;throw error
                    }
                }
                // Durable receipt acceptance wins over a later cancellation.
                observations=ledger;requestIdentity=entry.requestIdentity;savedOutputs=receipts;output=result;phase = .completed;statusMessage = entry.coverage?.isSampled == true ? "Completed · sampled working data and complete branches saved in History." : "Completed · verified complete data outputs saved in History."
            } catch {
                phase=cancelled || Task.isCancelled ? .cancelled:.failed
                observations?.finish(phase == .cancelled ? .cancelled:.failed)
                preparationFailure = phase == .failed && protocolFailure == nil ? stagedPreparationFailure:nil
                statusMessage=phase == .cancelled ? "Cancelled · reached-stage observations retained as partial; no completed result accepted.":(protocolFailure ?? workerFailureMessage ?? error.localizedDescription)
                if graph.nodes.contains(where:{$0.operatorID == "kwip.data.outputDelimited"}) {statusMessage += " External writes already committed may remain. Inspect file, database or API receipts before retrying; an unknown external outcome requires acknowledgment."}
                append(statusMessage)
            }
        }
    }
    private func current(_ lease:WorkflowEditorLease)throws {
        try Task.checkCancellation()
        guard !cancelled,owner.isCurrent(lease),owner.document==document,bindings.revision==bindingRevision,speechModelBindingsMatch,destinations.revision==destinationRevision,workspace.workflowSession?.owner === owner,workspace.activeProjectID==owner.projectID,workspace.dataPreparationReservationID==reservationID else {throw WorkflowArchiveError("Canvas, source or destination authority changed during preparation.")}
    }
    private func execute(window:ExecutionBudgetWindow,operation:String,admission:WorkflowCaptureStorageReservation?,inputs:@escaping (URL)throws->[WorkflowPreviewCapture.Input],destinations:[String:String],historyPolicy:RunHistoryPolicy)async throws->URL {
        return try await withCheckedThrowingContinuation {continuation in
            launchTask=Task { [self] in
                do {
                    let handle=try await WorkflowCapturedToolExecution.launch(mode:.preparation,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,projectID:owner.projectID,runtime:runtime,window:window,operationID:operation,storageAdmission:admission,inputs:inputs,arguments:{directory in
                        var args=["--dataflow-live-previews","true","--dataflow-measurements",historyPolicy.metricsAndDiagnostics ? "true":"false","--dataflow-capture-outputs",historyPolicy.componentOutputs ? "true":"false","--dataflow-sampling",samplingJSON,"--execute-dataflow",directory.appendingPathComponent("input/graph.json").path,"--source-bindings",directory.appendingPathComponent("input/bindings.json").path,"--dataflow-node-labels",directory.appendingPathComponent("input/node-labels.json").path,"--dataflow-cache-policy","off","--output",directory.appendingPathComponent("artifacts").path]
                        if !destinations.isEmpty {args += ["--destination-bindings",directory.appendingPathComponent("input/destinations.json").path]}
                        if graph.nodes.contains(where:{$0.operatorID=="kwip.data.outputDelimited" && APITableSettings.isExternal($0.config)}) {args += ["--defer-database-writes","true"]};return args
                    },retainInputsOnSuccess:true,launcher:launcher,onLine:{ [weak self] in self?.consume($0) },onExit:{code,message,result in
                        if code==0,self.workerFailureMessage == nil,let result {continuation.resume(returning:result)}else {continuation.resume(throwing:WorkflowArchiveError(message.isEmpty ? "Preparation worker exited with code \(code).":message))}
                    })
                    worker=handle;if cancelled || Task.isCancelled {handle.cancel()}
                } catch {continuation.resume(throwing:error)}
            }
        }
    }
    private func consume(_ line:String) {
        guard !cancelled,protocolFailure==nil,let bytes=line.data(using:.utf8),bytes.count<=1_000_000,let event=try? JSONDecoder().decode([String:JSONValue].self,from:bytes),let name=event["event"]?.stringValue else {return}
        do {
            if name == "dataflowNodeTiming" {
                try preparationMeasurements.consume(bytes, allowedNodes: Set(graph.nodes.map(\.id)), enabled: measurementCaptureEnabled)
            } else if name=="dataflowNodeStarted",let node=event["nodeID"]?.stringValue,let identity=observations?.identity {
                guard observations?.nodeStarted(node,identity:identity)==true else {throw WorkflowArchiveError("Preparation start does not match the captured graph.")}
                _ = canvasProgress?.start(.init(domain:.data,localID:node),stage:.preparing,context:runID,sequence:(canvasProgress?.sequence ?? 0)+1)
            } else if name=="dataflowPreview",let node=event["nodeID"]?.stringValue,case .object(let ports)=event["outputs"],let definition=graph.nodes.first(where:{$0.id==node}),let declared=owner.dataRegistry.operatorDefinition(id:definition.operatorID)?.configuredForExecution(definition.config,connectedInputs:Set(graph.edges.filter{$0.to.node==node}.map{$0.to.port})),let identity=observations?.identity {
                previewBytes += bytes.count
                guard previewBytes<=32_000_000,Set(ports.keys)==Set(declared.outputs.map(\.id)),preparationOutputs[node]==nil else {throw WorkflowArchiveError("Preparation output count, identity or display budget is invalid.")}
                for port in declared.outputs {guard let value=ports[port.id] else {throw WorkflowFileLeaseError.unsafePath};try WorkflowPreparationPreviewValidation.validate(value,kind:port.kind,operatorID:definition.operatorID)}
                guard observations?.consume(event,identity:identity)==true else {throw WorkflowArchiveError("Preparation schema observation failed provenance checks.")}
                preparationOutputs[node]=ports
                let disposition:WorkflowCanvasProgress.Disposition
                if case .object(let execution)=event["execution"],execution["status"] == .string("reused") {disposition = .reused}else {disposition = .computed}
                _ = canvasProgress?.complete(.init(domain:.data,localID:node),disposition:disposition,context:runID,sequence:(canvasProgress?.sequence ?? 0)+1)
                append("Captured table outputs · "+node)
            } else if name=="error" {
                let message=String((event["message"]?.stringValue ?? "Worker reported an error.").prefix(4000))
                append(message)
                if workerFailureMessage == nil {
                    workerFailureMessage=message
                    stagedPreparationFailure=try? WorkflowPreparationFailure.detail(event:event,graph:graph,labels:WorkflowPreparationFailure.labels(document:document,registry:owner.dataRegistry),registry:owner.dataRegistry)
                }
            }
            else if name=="dataflowOutputSaved" {append("Saved table · "+(event["nodeID"]?.stringValue ?? "output"))}
        } catch {protocolFailure=error.localizedDescription;worker?.cancel();statusMessage=error.localizedDescription}
    }
    private func append(_ text:String) {messages.append(text);if messages.count>200 {messages.removeFirst(messages.count-200)}}
    func snapshot(for address:WorkflowNodeAddress)->WorkflowCapturedDataOutput? {
        snapshots(for:[address])[address]
    }
    func snapshots(for addresses:Set<WorkflowNodeAddress>)->[WorkflowNodeAddress:WorkflowCapturedDataOutput] {
        guard !addresses.isEmpty,owner.isCurrent(owner.lease()),workspace.workflowSession?.owner === owner else {return [:]}
        let constructionMatches=owner.document.matchesCapturedConstruction(document)
        let bindingsMatch=speechModelBindingsMatch && bindings.revision==bindingRevision && destinations.revision==destinationRevision
        return addresses.reduce(into:[:]) { result,address in
            guard address.domain == .data,let ports=preparationOutputs[address.localID] else {return}
            result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:runID,requestIdentity:requestIdentity,capturedRevision:document.revision,bindingRevision:bindingRevision,phase:phase,matchesCurrentConstruction:constructionMatches && populationMatches(address.localID),matchesCurrentBindings:bindingsMatch,ports:ports,dataCompleteness:executionCoverage.map { $0.isComplete(address.localID) ? "Complete data" : "Sampled working data · first records" },capturedAt:capturedAt)
        }
    }
    private func populationMatches(_ nodeID: String) -> Bool {
        if completeConsumerAncestors.contains(nodeID) { return true }
        func effective(_ policy: PreparationSampling) -> PreparationSampling {
            policy.mode == .percentage && policy.percentage == 100 ? .all : policy
        }
        return effective(sampling) == effective(workspace.preparationSampling)
    }
    var currentObservationIdentity:PreparationObservationIdentity? {
        guard let identity=observations?.identity else {return nil}
        if owner.document.matchesCapturedConstruction(document),bindings.revision==bindingRevision,speechModelBindingsMatch,destinations.revision==destinationRevision,graph.nodes.allSatisfy({populationMatches($0.id)}) {return identity}
        return .init(requestID:identity.requestID,graphSHA256:identity.graphSHA256,sourceBindingsSHA256:identity.sourceBindingsSHA256,revision:identity.revision+"-stale")
    }
    func imageArtifactResolver(for address: WorkflowNodeAddress) -> FileImageArtifactResolver? {
        guard phase == .completed, snapshot(for: address) != nil else { return nil }
        return capturedImageResolver
    }
    func generationPromptArtifact(for endpoint:WorkflowEndpoint)->TableViewQuerySession.Artifact? {
        guard phase == .completed,!isRunning,bindings.revision==bindingRevision,speechModelBindingsMatch,populationMatches(endpoint.address.localID),WorkflowOutputConstruction.matches(owner.document,captured:document,provider:endpoint.address) else {return nil}
        return capturedTableArtifacts[endpoint]
    }
    func fullTableViewSession(for address: WorkflowNodeAddress) -> TableViewQuerySession? {
        capturedTableViewSession(for:.init(address:address,port:"table"))
    }
    func capturedTableViewSession(for endpoint:WorkflowEndpoint,formulaConsumer:WorkflowNodeAddress?=nil,selectionRequired:Bool=true) -> TableViewQuerySession? {
        let address=endpoint.address
        guard viewContextIsCurrent(endpoint,formulaConsumer:formulaConsumer,selectionRequired:selectionRequired),let artifact = (endpoint.port == "table" ? tableViewArtifacts[address.localID] : nil) ?? capturedTableArtifacts[endpoint] else { return nil }
        if selectionRequired,formulaConsumer == nil,let existing = tableViewSessions[endpoint] { return existing }
        let cancellation=FormulaSampleCancellationTarget()
        let reservation = WorkflowExecutionReservation(acquire: { [weak self] in
            guard let self, self.viewContextIsCurrent(endpoint,formulaConsumer:formulaConsumer,selectionRequired:selectionRequired), !self.workspace.isBusy, !self.owner.isBusy else { return nil }
            self.workspace.cancelDataPreparation = { [weak self,cancellation] in
                if formulaConsumer != nil || !selectionRequired {cancellation.session?.cancel()} else {self?.tableViewSessions[endpoint]?.cancel()}
            }
            let token = self.workspace.dataPreparationReservationID
            self.workspace.isPreparingData = true
            self.owner.setExecutionBusy(true)
            return token
        }, release: { [weak self] token in
            guard let self, self.workspace.dataPreparationReservationID == token else { return }
            self.workspace.isPreparingData = false; self.workspace.cancelDataPreparation = nil
            self.owner.setExecutionBusy(false)
        })
        let session = TableViewQuerySession(artifact: artifact, workspace: workspace.workspaceDirectory,
            accountingRoot: workspace.applicationDirectory, projectID: owner.projectID, runtime: runtime,
            storageLimitBytes: policy.storageLimitBytes, reservation: reservation, imageResolver: capturedImageResolver,
            accepts: { [weak self] in self?.viewContextIsCurrent(endpoint,formulaConsumer:formulaConsumer,selectionRequired:selectionRequired) == true })
        cancellation.session=session
        if selectionRequired,formulaConsumer == nil {tableViewSessions[endpoint] = session}
        return session
    }
    private func viewContextIsCurrent(_ endpoint: WorkflowEndpoint,formulaConsumer:WorkflowNodeAddress?=nil,selectionRequired:Bool=true) -> Bool {
        let address=endpoint.address
        return phase == .completed && !isRunning && address.domain == .data
            && owner.document.matchesCapturedConstruction(document) && speechModelBindingsMatch && bindings.revision == bindingRevision
            && destinations.revision == destinationRevision
            && workspace.activeProjectID == owner.projectID && workspace.workflowSession?.owner === owner
            && workspace.workflowSession?.ownsPreparation(self) == true
            && populationMatches(address.localID)
            && workspace.workflowSession?.sourceProvider.observedSourcesMatch(preparationOutputs) == true
            && (!selectionRequired || ownsViewSelection(endpoint,formulaConsumer:formulaConsumer))
    }
    private func ownsViewSelection(_ endpoint: WorkflowEndpoint,formulaConsumer:WorkflowNodeAddress?=nil) -> Bool {
        let address=endpoint.address
        if let consumer=formulaConsumer {
            return FormulaSampleScope.permits(consumer:consumer,input:endpoint,document:owner.document,selection:owner.selection)
        }
        if owner.selection.nodeAddresses == [address] && owner.selection.edgeIDs.isEmpty { return true }
        guard owner.selection.nodeAddresses.isEmpty, owner.selection.edgeIDs.count == 1,
              let id = owner.selection.edgeIDs.first,
              let edge = owner.document.edges.first(where: { $0.id == id }) else { return false }
        return edge.from == endpoint && [.table,.dataset].contains(edge.kind)
    }
    func cancel() {guard isRunning,phase != .completed else {return};cancelled=true;task?.cancel();launchTask?.cancel();worker?.cancel();RunFeedbackController.shared.transition(runID,state:.cancelling);statusMessage="Cancelling preparation; waiting for worker cleanup…"}
    nonisolated static func verifySavedOutputs(_ receipts:[[String:JSONValue]],graph:DataFlowDocument,destinations:[String:String])async throws {
        let task=Task.detached(priority:.userInitiated) {
            let sinks=graph.nodes.filter {$0.operatorID=="kwip.data.outputDelimited" && !APITableSettings.isExternal($0.config)}
            guard receipts.count==sinks.count,Set(receipts.compactMap {$0["nodeID"]?.stringValue})==Set(sinks.map(\.id)) else {throw WorkflowArchiveError("Saved output receipts differ from connected output components.")}
            for node in sinks {
                try Task.checkCancellation()
                guard let receipt=receipts.first(where:{$0["nodeID"] == .string(node.id)}),let folder=destinations[node.id],case .string(let filename)=node.config["filename"],receipt["path"] == .string(URL(fileURLWithPath:folder).appendingPathComponent(filename).path),receipt["status"] == .string("saved"),receipt["completeSelectedTable"] == .bool(true),case .integer(let count)=receipt["rowCount"],count>=0,case .integer(let length)=receipt["bytes"],(0...50_000_000).contains(length),case .string(let hash)=receipt["sha256"],WorkflowArchiveCodec.isHash(hash) else {throw WorkflowArchiveError("Output receipt differs from its granted destination or complete-table evidence.")}
                let raw=try DiagnosticFileIO.read(root:URL(fileURLWithPath:folder),parts:[filename],limit:50_000_000)
                guard raw.count==length,WorkflowArchiveCodec.hash(raw)==hash else {throw WorkflowArchiveError("Saved table bytes differ from the worker’s output receipt.")}
            }
        }
        try await withTaskCancellationHandler(operation:{try await task.value},onCancel:{task.cancel()})
    }
}

private extension Dictionary where Key==WorkflowNodeAddress,Value==String {
    var mapKeysToLocalIDs:[String:String] {Dictionary<String,String>(uniqueKeysWithValues:map {($0.key.localID,$0.value)})}
}

@MainActor private final class FormulaSampleCancellationTarget {weak var session:TableViewQuerySession?}
