import Foundation
import AppKit
import Combine
import KwipMLCore

/// Native host for one canonical workflow owner. Legacy model/preparation stores
/// are never installed as execution shadows. Completed outcomes retain their own
/// request identity; persistent History reopening is a separate integration.
@MainActor final class WorkflowWorkspaceSession: ObservableObject, WorkflowDataOutputs {
    private weak var workspace: WorkspaceStore?
    var settingsWorkspace:WorkspaceStore? {workspace}
    var configurationAppearance:AppAppearanceState {workspace?.appearanceState ?? AppAppearanceState()}
    private let historyHostID:UUID
    /// Identifies one externally initiated operation, including its automatic
    /// consumers. Automation cancellation must never borrow a later UI run.
    private(set) var operationIdentity=UUID()
    var hasActiveOperation:Bool {
        // Inspection owns the handoff after table preparation releases its worker.
        if fineTuningInspectionAddress != nil {return true}
        if campaignContinuationTask != nil || predictionPreparationScheduled {return true}
        if selfPlay?.isBusy == true || selfPlayValidation?.isBusy == true {return true}
        if gameEpisodes?.isBusy == true || gameValidation?.isBusy == true {return true}
        if apiRunTask != nil || modelSave?.isBusy == true || modelLoader.isBusy || runStagingID != nil || queuedRequestID != nil || generation?.isBusy == true || isManagingTrainedArtifacts || isStagingEvaluation || modelFlowID != nil{return true}
        if scopedPreparation?.isRunning == true || currentDataPreparation?.isRunning == true || predictionPreparationIsCurrent && predictionPreparation?.isRunning == true{return true}
        if let phase=currentExecution?.phase,[.preparing,.training].contains(phase){return true}
        if let phase=currentBackendFit?.phase,[.preparing,.training].contains(phase){return true}
        if let phase=evaluation?.phase,![.idle,.completed,.failed,.cancelled].contains(phase){return true}
        return false
    }
    var displayedStatusMessage:String {statusMessage}
    let owner: WorkflowAuthoringStore
    let executionQueue: WorkflowExecutionQueue
    @Published private(set) var queuedRequestID: UUID?
    let canvas: WorkflowCanvasAdapter
    let bindings: WorkflowPrivateSourceBindings
    let checkpointBindings: WorkflowPrivateCheckpointBindings
    let cacheBindings: WorkflowPrivateCacheBindings
    let modelDestinations:WorkflowPrivateModelDestinations
    @Published private(set) var modelSave:WorkflowSaveModelSession?
    @Published private(set) var externalModelSaveReceipts:[WorkflowNodeAddress:URL]=[:]
    @Published private(set) var modelSaveReceipts:[WorkflowSaveModelReceipt]=[]
    @Published private(set) var modelSaveReceiptsByAddress:[WorkflowNodeAddress:WorkflowSaveModelReceipt]=[:]
    @Published private(set) var isManagingTrainedArtifacts=false
    var selfPlayWorkspaceRoot:URL? {workspace?.workspaceDirectory}
    @Published private(set) var selfPlay:SelfPlaySession?
    @Published private(set) var selfPlayValidation:SelfPlaySession?
    @Published private(set) var selfPlayBindings:[WorkflowNodeAddress:String]=[:]
    @Published private(set) var selfPlayRecords:[SelfPlayRecord]=[]
    private var selfPlayObservation:AnyCancellable?,selfPlayValidationObservation:AnyCancellable?
    private var campaignContinuationTask:Task<Void,Never>?
    private var selfPlayQueries:[WorkflowEndpoint:TableViewQuerySession]=[:]
    @Published private(set) var gameRecords:[GameEpisodeRecord]=[]
    @Published private(set) var gameEpisodes:GameEpisodeSession?
    @Published private(set) var gameValidation:GameEpisodeSession?
    private var gameObservation:AnyCancellable?
    private var gameValidationObservation:AnyCancellable?
    private var gameQueries:[WorkflowEndpoint:TableViewQuerySession]=[:]
    @Published private(set) var generation:WorkflowGenerationSession?
    @Published private(set) var generationIsLatest=false
    @Published private(set) var generationRecords:[WorkflowGenerationRecord]=[]
    private var generationRefreshTask:Task<Void,Never>?
    private var generationObservation:AnyCancellable?
    private var generationQuery:TableViewQuerySession?
    private var generationMetadataCache:[String:WorkflowGenerationModelMetadata]=[:]
    private var modelArtifactTask:Task<Void,Never>?
    private var modelArtifactID:UUID?
    private var pendingModelConsumers:[WorkflowNodeAddress]=[]
    private var modelFlowID:UUID?
    private var modelSaveObservations:[AnyCancellable]=[]
    let modelLoader:ModelLoaderStore
    let frozenBindings:WorkflowPrivateFrozenModelBindings
    let backendHistory: BackendFitHistoryStore
    let evaluationHistory:WorkflowEvaluationHistoryStore
    @Published private(set) var evaluation:WorkflowEvaluationSession?
    @Published private(set) var evaluationOutcomes:[WorkflowEvaluationSessionResult]=[]
    @Published private(set) var isStagingEvaluation=false
    var evaluationLaunch:WorkflowWorkerLauncher=WorkflowExecutionSession.launchWorker
    var evaluationStage:WorkflowEvaluationRequest.Stage? = nil
    var selectFrozenReference:(WorkflowHistoryReference,URL) async throws -> WorkflowFrozenModelReference = {reference,root in
        try await WorkflowHistoryVerificationWork.run {try WorkflowFrozenModelSnapshot.select(reference,root:root)}
    }
    private var evaluationStagingTask:Task<Void,Never>?
    private var frozenSelectionTask:Task<WorkflowFrozenModelReference,Error>?
    private var evaluationStagingID:UUID?
    private var evaluationObservations:[AnyCancellable]=[]
    let outputDestinations:DataOutputDestinationProvider
    let dataPreparationHistory:WorkflowDataPreparationHistory
    @Published private(set) var dataPreparation:WorkflowDataPreparationSession?
    private var dataPreparationObservations:[AnyCancellable]=[]
    private var latestRunID:UUID? {didSet {generationIsLatest=false}}
    private var capturedQueryRunID:UUID?
    private var capturedQueries:[WorkflowEndpoint:TableViewQuerySession]=[:]
    /// One current-run authority for every Results tab. Retained prior sessions
    /// remain historical objects, never fallback data for a newer run kind.
    var currentResult: WorkflowCurrentResult? {
        guard evaluationHostIsCurrent,let latestRunID else {return nil}
        if let dataPreparation,dataPreparation.runID == latestRunID,
           dataPreparation.document.projectID == owner.projectID {return .preparation(dataPreparation)}
        if let execution,execution.runID == latestRunID,
           execution.request.lease.projectID == owner.projectID,
           execution.request.lease.sessionID == owner.sessionID {return .training(execution)}
        if let backendFit,backendFit.runID == latestRunID,
           backendFit.document.projectID == owner.projectID {return .backend(backendFit)}
        return nil
    }
    var currentExecution: WorkflowExecutionSession? {
        guard case .training(let value) = currentResult else {return nil};return value
    }
    var currentBackendFit: BackendFitSession? {
        guard case .backend(let value) = currentResult else {return nil};return value
    }
    var currentDataPreparation:WorkflowDataPreparationSession? {
        guard case .preparation(let value) = currentResult else {return nil};return value
    }
    let sourceProvider: WorkflowSourceProvider
    @Published private(set) var sourceCaptureProgress:WorkflowCanvasProgress?
    private var sourceCaptureIdentity:(staging:UUID,revision:UInt64,binding:UInt64)?
    /// Never expose a capture from a replaced owner, changed binding or cancelled task.
    var currentSourceCaptureProgress:WorkflowCanvasProgress? {
        guard let workspace,!isRetired,workspace.workflowSession === self,
              workspace.activeProjectID == owner.projectID,owner.isCurrent(owner.lease()),
              let identity=sourceCaptureIdentity,runStagingID == identity.staging,
              runStagingTask?.isCancelled == false,owner.document.revision == identity.revision,
              bindings.revision == identity.binding,let progress=sourceCaptureProgress,
              progress.context == identity.staging else {return nil}
        return progress
    }
    @Published private(set) var isStagingCheckpoint = false
    var captureRunSources: @MainActor ([WorkflowNodeAddress: String], URL, URL, Int64?) async throws -> WorkflowRunSourceCapture = { paths, root, permissions, limit in
        try await WorkflowRunSourceCapture.capture(paths: paths, workspace: root, sourcePermissions: permissions, storageLimitBytes: limit)
    }
    var captureCheckpoint: (WorkflowCheckpointReference, URL, Int64?) async throws -> WorkflowCheckpointSnapshot = { reference, root, limit in
        try await WorkflowHistoryVerificationWork.run {try WorkflowHistoryReader.stageCheckpoint(reference, root: root, storageLimitBytes: limit)}
    }
    private var runStagingTask: Task<Void, Never>?
    private var runStagingID: UUID?
    @Published private(set) var isStagingExperiment = false
    private var experimentStagingTask:Task<Void,Never>?
    var captureExperimentInputs:([WorkflowNodeAddress:String],URL) async throws -> WorkflowExperimentInputs
    @Published private(set) var experimentRunner:WorkflowExperimentRunner?
    private var experimentObservations:[AnyCancellable]=[]
    private var completedDocumentFits:[WorkflowNodeAddress:BackendFitSession]=[:]
    @Published private(set) var backendFit: BackendFitSession?
    private var backendObservation: AnyCancellable?
    private var backendStatusObservation: AnyCancellable?
    @Published private(set) var execution: WorkflowExecutionSession?
    @Published private(set) var outcomes: [WorkflowExecutionOutcome] = []
    @Published private(set) var statusMessage = "Select a training or estimator component to execute this workflow."
    @Published private(set) var isRetired = false
    var executionLaunch: WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker
    private var observations: [AnyCancellable] = []
    private var executionObservation: AnyCancellable?
    private var canvasProgressObservation:AnyCancellable?
    private var executionStatusObservation:AnyCancellable?
    private var executionBindingRevision: UInt64?
    private var retainedHost: WorkflowWorkspaceSession?
    private var retainedWorkspace: WorkspaceStore?

    var performanceCaptureEnabled:Bool {owner.runHistoryPolicy.metricsAndDiagnostics && ((try? workspace?.executionPolicy(for:owner.projectID).sampleCapture.performanceMeasurements) ?? false)}
    /// Convenience action across two independent preference stores. Each group
    /// is atomic; report a partial result instead of claiming a two-store commit.
    @discardableResult func setPerformanceCaptureEnabled(_ enabled:Bool)->Bool {
        guard let workspace,!isRetired else {return false}
        var sampleSaved=false
        do {
            let before=try workspace.projectSettingsSnapshot(projectID:owner.projectID)
            let capture=try workspace.previewProjectSettings(.trainingCapture(before.effective.sampleCapture.changing(performanceMeasurements:enabled)),baseline:before)
            let needsHistory=enabled && !before.history.metricsAndDiagnostics
            if needsHistory {
                // Validate both requested groups before making the first write.
                _ = try workspace.previewProjectSettings(.historyCapture(.init(metricsAndDiagnostics:true,componentOutputs:before.history.componentOutputs)),baseline:before)
            }
            _ = try workspace.applyProjectSettings(capture);sampleSaved=true
            if needsHistory {
                let latest=try workspace.projectSettingsSnapshot(projectID:owner.projectID)
                _ = try workspace.applyProjectSettings(workspace.previewProjectSettings(.historyCapture(.init(metricsAndDiagnostics:true,componentOutputs:latest.history.componentOutputs)),baseline:latest))
            }
            objectWillChange.send();return true
        } catch {
            if sampleSaved {
                let state:String
                if let saved=try? workspace.projectSettingsSnapshot(projectID:owner.projectID) {
                    state=saved.history.metricsAndDiagnostics && saved.effective.sampleCapture.performanceMeasurements ? "on":"off"
                } else {state="unavailable"}
                statusMessage="Performance capture choice saved, but History diagnostics were not saved. Effective capture is \(state). \(error.localizedDescription)"
            } else {statusMessage=error.localizedDescription}
            objectWillChange.send();return false
        }
    }
    @Published private(set) var predictionPreparation:WorkflowDataPreparationSession?
    private var predictionPreparationObservations:[AnyCancellable]=[]
    private var predictionPreparationPins:[WorkflowEndpoint:String]=[:]
    private var predictionPreparationScheduled=false
    private var predictionQueries:[WorkflowEndpoint:(String,TableViewQuerySession)]=[:]
    func predictionTableViewSession(for endpoint:WorkflowEndpoint,formulaConsumer:WorkflowNodeAddress?=nil,selectionRequired:Bool=true)->TableViewQuerySession? {
        if selfPlayOutputIsCurrent(endpoint) {return selfPlayTableViewSession(for:endpoint)}
        if gameOutputIsCurrent(endpoint) {return gameTableViewSession(endpoint)}
        if endpoint.port=="table" {return generationTableViewSession(for:endpoint,usesSharedSession:selectionRequired)}
        guard let workspace,["predictions","profiles","confusion","routing","routingSample"].contains(endpoint.port),predictionOutputIsCurrent(endpoint) else {return nil}
        let artifact:TableViewQuerySession.Artifact
        let runtime:RuntimeLocation
        if let value=backendFit?.predictionArtifact(for:endpoint),let location=workspace.runtime {artifact=value;runtime=location}
        else if let value=modelLoader.predictionArtifact(endpoint.address),let location=workspace.runtime {artifact=value;runtime=location}
        else if let execution=currentExecution,execution.request.plan.train==endpoint.address,let value=execution.outputTableArtifact(endpoint.port) {artifact=value;runtime=execution.request.runtime}
        else if let evaluation,evaluation.request.plan.evaluate==endpoint.address,let value=evaluation.predictionTableArtifact {artifact=value;runtime=evaluation.request.runtime}
        else {return nil}
        if selectionRequired,formulaConsumer==nil,let (pin,query)=predictionQueries[endpoint],pin==artifact.sourceSHA256 {return query}
        if selectionRequired,formulaConsumer==nil {predictionQueries[endpoint]?.1.invalidate()}
        let pin=artifact.sourceSHA256
        let accepts={ [weak self] in
            guard let self,self.predictionOutputIsCurrent(endpoint),(!selectionRequired || self.predictionSelectionPermits(endpoint,formulaConsumer:formulaConsumer)) else {return false}
            if self.backendFit?.predictionArtifact(for:endpoint)?.sourceSHA256==pin {return true}
            if self.modelLoader.predictionArtifact(endpoint.address)?.sourceSHA256==pin {return true}
            if let execution=self.currentExecution,execution.request.plan.train==endpoint.address {return execution.outputTableArtifact(endpoint.port)?.sourceSHA256==pin}
            return self.evaluation?.predictionTableArtifact?.sourceSHA256==pin
        }
        final class CancellationTarget {weak var query:TableViewQuerySession?}
        let target=CancellationTarget()
        let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
            guard let self,let workspace,accepts(),!workspace.isBusy,!self.owner.isBusy else {return nil}
            workspace.cancelDataPreparation={target.query?.cancel()};let token=workspace.dataPreparationReservationID
            workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return token
        },release:{[weak self,weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID==token else {return}
            workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
        })
        let query=TableViewQuerySession(artifact:artifact,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,
            projectID:owner.projectID,runtime:runtime,storageLimitBytes:(try? workspace.executionPolicy(for:owner.projectID).storageLimitBytes) ?? nil,
            reservation:reservation,accepts:accepts)
        target.query=query;if selectionRequired,formulaConsumer==nil {predictionQueries[endpoint]=(pin,query)};return query
    }
    private func predictionSelectionPermits(_ endpoint:WorkflowEndpoint,formulaConsumer:WorkflowNodeAddress?)->Bool {
        if let consumer=formulaConsumer {return FormulaSampleScope.permits(consumer:consumer,input:endpoint,document:owner.document,selection:owner.selection)}
        if owner.selection.nodeAddresses == [endpoint.address],owner.selection.edgeIDs.isEmpty {return true}
        if canvas.inspectedSourceEndpoints.contains(endpoint) {return true}
        if owner.selection.nodeAddresses.isEmpty,owner.selection.edgeIDs.count==1,let id=owner.selection.edgeIDs.first {
            return owner.document.edges.contains {$0.id==id && $0.from==endpoint}
        }
        return false
    }
    private func predictionEvidence(for endpoint:WorkflowEndpoint)->WorkflowPredictionTableEvidence? {
        if let execution=currentExecution,execution.request.plan.train==endpoint.address {return endpoint.port=="predictions" ? execution.predictionTable:execution.routingTables[endpoint.port]}
        if let evaluation,evaluation.request.plan.evaluate==endpoint.address {return evaluation.verifiedResult?.predictionTable}
        return nil
    }
    func predictionOutputIsCurrent(_ endpoint:WorkflowEndpoint)->Bool {
        guard evaluationHostIsCurrent else {return false}
        if selfPlayOutputIsCurrent(endpoint) {return true}
        if gameOutputIsCurrent(endpoint) {return true}
        if endpoint.port=="table" {return generationOutputIsCurrent(endpoint)}
        if let fit=backendFit,fit.predictionArtifact(for:endpoint) != nil {return fit.phase == .completed && WorkflowOutputConstruction.matches(owner.document,captured:fit.document,provider:endpoint.address) && bindings.revision==fit.bindingRevision}
        if endpoint.port=="predictions",modelLoader.predictionArtifact(endpoint.address) != nil {return true}
        if let execution=currentExecution,execution.request.plan.train==endpoint.address {
            if ["routing","routingSample"].contains(endpoint.port),execution.outputTableArtifact(endpoint.port)==nil{return false}
            return execution.phase == .completed && outcomes.contains(where:{$0.runID==execution.runID && $0.status == .completed && $0.recordError == nil}) && WorkflowOutputConstruction.matches(owner.document,captured:execution.request.authoringDocument,provider:endpoint.address) && bindings.revision==executionBindingRevision && currentSourceEvidenceMatches(execution:execution)
        }
        if let evaluation,evaluation.request.plan.evaluate==endpoint.address {
            return evaluation.phase == .completed && evaluation.verifiedResult != nil && evaluation.predictionTableArtifact != nil
                && WorkflowOutputConstruction.matches(owner.document,captured:evaluation.request.document,provider:endpoint.address)
                && frozenBindings.revision==evaluation.request.bindingRevision && frozenBindings.reference(for:evaluation.request.plan.frozenModel)==evaluation.request.reference
        }
        return false
    }
    init(workspace: WorkspaceStore, owner: WorkflowAuthoringStore) throws {
        guard owner.isCurrent(owner.lease()) else {
            throw DataFlowContractError("Use an active workflow owner when creating its workspace host.")
        }
        self.workspace = workspace; self.owner = owner
        if owner.document.nodes.contains(where:{$0.address.domain == .data}),
           !owner.document.nodes.contains(where:{node in
               if case .train=node.payload {return true}
               if case .estimator=node.payload {return true}
               return false
           }) {
            statusMessage="Configure the connected data components, then choose Run Flow to prepare data."
        }
        executionQueue = .shared(directory: workspace.applicationDirectory.appendingPathComponent("workflow-queue"))
        owner.useAuthoringRetentionPreferences(at: workspace.appPreferenceStore.fileURL)
        do {try owner.configureCanvasVersions(root: workspace.workspaceDirectory)}
        catch {statusMessage="Canvas History unavailable: "+error.localizedDescription}
        let historyHostID=UUID()
        self.historyHostID=historyHostID
        let sourceAccess=workspace.sourceAccess
        captureExperimentInputs={ [weak workspace,weak owner] paths,directory in
            guard let workspace,let owner else {throw WorkflowFileLeaseError.unsafePath}
            let configs=DataFlowSourceOperators.fileConfigurations(in:owner.document).filter {paths[$0.key] != nil}
            let limit=try workspace.executionPolicy(for:owner.projectID).storageLimitBytes
            return try await sourceAccess.captureExperimentInputs(paths:paths,directory:directory,fileConfigurations:configs,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,storageLimitBytes:limit)
        }
        canvas = try WorkflowCanvasAdapter(owner: owner, modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry)
        bindings = try WorkflowPrivateSourceBindings(owner: owner, directory: workspace.workspaceDirectory,ephemeral:workspace.scheduledExecutionID != nil)
        checkpointBindings = try WorkflowPrivateCheckpointBindings(owner: owner, directory: workspace.workspaceDirectory,ephemeral:workspace.scheduledExecutionID != nil)
        cacheBindings = try WorkflowPrivateCacheBindings(owner: owner, directory: workspace.workspaceDirectory,ephemeral:workspace.scheduledExecutionID != nil)
        frozenBindings = try WorkflowPrivateFrozenModelBindings(owner:owner,directory:workspace.workspaceDirectory,ephemeral:workspace.scheduledExecutionID != nil)
        modelLoader = try ModelLoaderStore(owner:owner,root:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,ephemeralSelections:workspace.scheduledExecutionID != nil)
        modelDestinations = try WorkflowPrivateModelDestinations(owner:owner,directory:workspace.workspaceDirectory)
        // Saved results outlive individual graph edits. Retain host identity,
        // while checking the owner's current lifetime rather than an old edit lease.
        evaluationHistory=WorkflowEvaluationHistoryStore(root:workspace.workspaceDirectory,projectID:owner.projectID,exposureRoot:workspace.applicationDirectory,isCurrent:{[weak workspace,weak owner] in
            guard let workspace,let owner,let host=workspace.workflowSession else {return false}
            return host.historyHostID==historyHostID && host.owner === owner && !host.isRetired && workspace.activeProjectID==owner.projectID && owner.isCurrent(owner.lease())
        })
        backendHistory = BackendFitHistoryStore(root: workspace.workspaceDirectory, projectID: owner.projectID)
        evaluationHistory.workspace=workspace
        let accountingRoot = workspace.applicationDirectory
        captureCheckpoint = { reference, root, limit in
            try await WorkflowHistoryVerificationWork.run {try WorkflowHistoryReader.stageCheckpoint(reference, root: root, accountingRoot: accountingRoot, storageLimitBytes: limit)}
        }

        outputDestinations=try DataOutputDestinationProvider(workspace:workspace,owner:owner)
        dataPreparationHistory=WorkflowDataPreparationHistory(root:workspace.workspaceDirectory,projectID:owner.projectID)
        sourceProvider = try WorkflowSourceProvider(workspace: workspace, owner: owner, bindings: bindings)
        sourceProvider.outputDestinations=outputDestinations
        sourceProvider.capturedPredictionSchema={ [weak self] endpoint in
            guard let self,self.predictionOutputIsCurrent(endpoint) else {return nil}
            if let record=self.modelLoader.currentEvaluation,record.address==endpoint.address,self.modelLoader.evaluationIsCurrent(record),let metadata=self.modelLoader.evaluationTableMetadata {return DataObservedSchemaProjection.columns(.object(metadata),isCurrent:true)}
            if let metadata=self.backendFit?.predictionMetadata(for:endpoint) {return DataObservedSchemaProjection.columns(.object(metadata),isCurrent:true)}
            if endpoint.port=="table",let metadata=self.generation?.outputMetadata {return DataObservedSchemaProjection.columns(.object(metadata),isCurrent:true)}
            guard let evidence=self.predictionEvidence(for:endpoint) else {return nil}
            return DataObservedSchemaProjection.columns(evidence.outputMetadata,isCurrent:true)
        }
        sourceProvider.capturedSchema = { [weak self] endpoint in
            guard let self,let capture=self.snapshot(for:.init(domain:.data,localID:endpoint.node)),
                  let value=capture.ports[endpoint.port] else {return nil}
            return DataObservedSchemaProjection.columns(value,isCurrent:capture.matchesCurrentConstruction && capture.matchesCurrentBindings)
        }
        captureRunSources = { [weak self,weak workspace] paths, root, permissions, limit in
            let fileConfigurations=(self.map {DataFlowSourceOperators.fileConfigurations(in:$0.owner.document)} ?? [:]).filter {paths[$0.key] != nil}
            guard let self,let workspace,let stagingID=self.runStagingID else {
                return try await WorkflowRunSourceCapture.capture(paths:paths,workspace:root,accountingRoot:accountingRoot,sourcePermissions:permissions,storageLimitBytes:limit,fileConfigurations:fileConfigurations)
            }
            let revision=self.owner.document.revision,bindingRevision=self.bindings.revision
            self.sourceCaptureIdentity=(stagingID,revision,bindingRevision)
            self.sourceCaptureProgress = .init(context:stagingID,allowedNodes:Set(paths.keys),edges:[])
            return try await WorkflowRunSourceCapture.capture(paths:paths,workspace:root,accountingRoot:accountingRoot,sourcePermissions:permissions,storageLimitBytes:limit,fileConfigurations:fileConfigurations,progress:{ [weak self,weak workspace] event in
                Task { @MainActor in
                    guard let self,let workspace,!self.isRetired,workspace.workflowSession === self,
                          workspace.activeProjectID == self.owner.projectID,self.owner.isCurrent(self.owner.lease()),
                          self.runStagingID == stagingID,self.runStagingTask?.isCancelled == false,self.owner.document.revision == revision,self.bindings.revision == bindingRevision,
                          var feed=self.sourceCaptureProgress else {return}
                    if event.finished {_=feed.complete(event.address,disposition:.captured,context:stagingID,sequence:event.sequence)}
                    else {_=feed.start(event.address,stage:.capturing,context:stagingID,sequence:event.sequence)}
                    self.sourceCaptureProgress=feed;self.canvas.objectWillChange.send()
                }
            })
        }

        modelLoader.onBindingChange = { [weak self] in self?.canvas.objectWillChange.send();self?.objectWillChange.send();self?.workspace?.objectWillChange.send() }
        observations = [owner.objectWillChange, canvas.objectWillChange, bindings.objectWillChange, cacheBindings.objectWillChange, checkpointBindings.objectWillChange, frozenBindings.objectWillChange, modelDestinations.objectWillChange, outputDestinations.objectWillChange, dataPreparationHistory.objectWillChange, backendHistory.objectWillChange, evaluationHistory.objectWillChange, sourceProvider.objectWillChange].map { publisher in
            publisher.sink { [weak self] _ in self?.objectWillChange.send() }
        }
        canvas.outputProvider = self
        canvas.componentAvailability = { [weak workspace] id in
            !EmbeddingBagRuntimeCapabilities.operations.contains(id) || workspace?.embeddingBagRuntimeCapabilities.isComplete == true
        }
        // Observe the underlying publishers, not self: canvas notifications already
        // forward into the host and subscribing back would create a feedback loop.
        observations.append(bindings.objectWillChange.sink { [weak self] _ in self?.canvas.objectWillChange.send() })
        observations.append(workspace.$embeddingBagRuntimeCapabilities.dropFirst().sink { [weak self] _ in self?.canvas.objectWillChange.send() })
        canvas.onSelectSource = { [weak self] address, lease in
            guard let self, !self.isRetired else { return }
            _ = self.sourceProvider.chooseSource(for: address, lease: lease)
        }
        for publisher in [bindings.$revision.eraseToAnyPublisher(),checkpointBindings.$revision.eraseToAnyPublisher(),cacheBindings.$revision.eraseToAnyPublisher()] {
            observations.append(publisher.dropFirst().sink {[weak self] _ in self?.invalidateTrainOutputs()})
        }

    }

    private var evaluationHostIsCurrent:Bool {
        guard let workspace else {return false}
        return !isRetired && workspace.workflowSession === self && workspace.activeProjectID==owner.projectID && owner.isCurrent(owner.lease())
    }
    // Trained artifact branches retain one exact completed run. Consumers never
    // discover an arbitrary newest checkpoint from History.
    private func directModelConsumers(of train:WorkflowNodeAddress)->[WorkflowNodeAddress] {
        Set(owner.document.edges.filter {$0.from == .init(address:train,port:"model") && $0.to.port=="model" && $0.kind == .frozenModel}.map(\.to.address))
            .filter {address in owner.document.nodes.contains {node in
                guard node.address==address else {return false}
                switch node.payload {case .evaluateTest,.saveModel,.generateText:return true;case .dataOperator(let id,_) where id==TypedDecisionConfiguration.operatorID:return true;default:return false}
            }}.sorted {$0.localID<$1.localID}
    }
    private func invalidateTrainOutputs() {
        guard !owner.isBusy else {modelArtifactTask?.cancel();return}
        for node in owner.document.nodes where FineTuningConfiguration.isExternal(node) {
            if modelLoader.bindings.reference(for:node.address) != nil {_=modelLoader.bindings.setReference(nil,for:node.address,lease:owner.lease(),expectedRevision:modelLoader.bindings.revision)}
        }
        for node in owner.document.nodes {
            guard case .train=node.payload,frozenBindings.reference(for:node.address) != nil else {continue}
            if !frozenBindings.setReference(nil,for:node.address,lease:owner.lease(),expectedRevision:frozenBindings.revision) {
                statusMessage=frozenBindings.error ?? "Could not invalidate the previous trained model."
            }
        }
    }
    private func clearModelFlow() {pendingModelConsumers=[];modelFlowID=nil}
    private func prepareModelConsumers(of train:WorkflowNodeAddress)throws {
        if modelFlowID == nil {pendingModelConsumers=directModelConsumers(of:train);modelFlowID=UUID()}
        for consumer in pendingModelConsumers {
            if let node=owner.document.nodes.first(where:{$0.address==consumer}),case .saveModel=node.payload,modelDestinations.path(for:consumer)==nil {
                clearModelFlow();throw DataFlowContractError("Choose Save model’s destination before running this branch.")
            }
        }
    }
    /// Only the captured parent flow may admit another consumer or its final data sweep.
    /// Direct inspector/MCP consumers finish at their own result, even when other
    /// captured tables and authorized output destinations are available on the canvas.
    private func finishModelConsumer(_ success:Bool,continuingOperation:Bool,flowID:UUID?) {
        guard continuingOperation,let flowID,modelFlowID==flowID else{return}
        if success && evaluationHostIsCurrent {advanceModelConsumers()}else{clearModelFlow()}
    }
    private func finishImportedModelOperation(_ success:Bool,continuingOperation:Bool,flowID:UUID?) {
        guard evaluationHostIsCurrent else {finishModelConsumer(false,continuingOperation:continuingOperation,flowID:flowID);return}
        statusMessage=modelLoader.status;canvas.objectWillChange.send()
        finishModelConsumer(success,continuingOperation:continuingOperation,flowID:flowID)
    }
    private func startModelConsumer(_ address:WorkflowNodeAddress,forceTraining:Bool)->Bool {
        do {
            let source=try WorkflowTrainedModelSource.resolve(in:owner.document,consumer:address)
            if selfPlayRecords.contains(where:{$0.address==source.address && $0.completed}) || selfPlayBindings[source.address] != nil {
                if let node=owner.document.nodes.first(where:{$0.address==address}),case .saveModel=node.payload {return saveModel(address:address,continuingOperation:true)}
            }
            if source.isTrain && (forceTraining || (frozenBindings.reference(for:source.address)==nil && modelLoader.reference(source.address)==nil)) {
                pendingModelConsumers=[address];modelFlowID=UUID()
                guard run(train:source.address,continuingOperation:true) else {clearModelFlow();return false}
                return true
            }
            guard let node=owner.document.nodes.first(where:{$0.address==address}) else {return false}
            if case .evaluateTest=node.payload {return evaluate(address:address,continuingOperation:true)}
            if case .generateText=node.payload {return generateText(address:address,continuingOperation:true)}
            if case .saveModel=node.payload {return saveModel(address:address,continuingOperation:true)}
            if case .dataOperator(let id,_)=node.payload,id==TypedDecisionConfiguration.operatorID {return try startDecisionPreparation(address)}
            return false
        } catch {statusMessage=error.localizedDescription;clearModelFlow();return false}
    }
    private func advanceModelConsumers() {
        guard evaluationHostIsCurrent,modelFlowID != nil,!pendingModelConsumers.isEmpty else {clearModelFlow();schedulePredictionPreparation();return}
        let next=pendingModelConsumers.removeFirst(),id=modelFlowID
        Task { [weak self] in
            guard let self,let workspace=self.workspace,self.evaluationHostIsCurrent,self.modelFlowID==id else {return}
            // Results can acquire a bounded table-read reservation as a model
            // finishes. Let it release before admitting the queued consumer;
            // never bypass admission, rerun training or retry a write.
            for _ in 0..<50 where workspace.isBusy || self.owner.isBusy {
                do {try await Task.sleep(for:.milliseconds(100))} catch {return}
                guard !Task.isCancelled,self.evaluationHostIsCurrent,self.modelFlowID==id else {return}
            }
            guard !Task.isCancelled,self.evaluationHostIsCurrent,self.modelFlowID==id else {return}
            if !self.startModelConsumer(next,forceTraining:false) {self.clearModelFlow()}
        }
    }
    @discardableResult private func startDecisionPreparation(_ address:WorkflowNodeAddress)throws->Bool {
        guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy,let runtime=workspace.runtime else {throw DataFlowContractError("Wait for the current operation before preparing decisions.")}
        let plan=try WorkflowDecisionPreparation(document:owner.document,consumer:address)
        let session=try WorkflowDataPreparationSession(workspace:workspace,owner:owner,bindings:bindings,destinations:outputDestinations,history:dataPreparationHistory,runtime:runtime,policy:workspace.executionPolicy(for:owner.projectID),launcher:executionLaunch,acceptedBudgetWindow:currentAPIBudgetWindow,allowTrainingTerminals:true,preparationDocument:plan.document)
        dataPreparation=session
        dataPreparationObservations=[session.objectWillChange.sink {[weak self,weak session] _ in
            guard let self,let session,self.dataPreparation === session,!self.isRetired else {return};self.objectWillChange.send();self.canvas.objectWillChange.send()
        },session.$statusMessage.sink {[weak self] in self?.statusMessage=$0}]
        session.onTerminal={ [weak self] success in
            guard let self else {return}
            if success,self.modelFlowID != nil {self.advanceModelConsumers()} else if !success {self.clearModelFlow()}
        }
        try session.start();latestRunID=session.runID;return true
    }
    private func schedulePredictionPreparation(ownedBy identity:UUID?=nil) {
        guard !predictionPreparationScheduled,!WorkflowPredictionPreparation.dependentData(in:owner.document).isEmpty else {return}
        predictionPreparationScheduled=true
        Task { [weak self] in
            guard let self else {return};defer {self.predictionPreparationScheduled=false}
            do {
                guard let workspace=self.workspace,self.evaluationHostIsCurrent,identity == nil || self.operationIdentity == identity else {return}
                // A newly visible prediction table may already own a short query
                // reservation when completion schedules its downstream consumers.
                // Let that bounded read finish instead of reporting a false failure.
                for _ in 0..<50 where workspace.isBusy || self.owner.isBusy {
                    try await Task.sleep(nanoseconds:100_000_000)
                    guard self.evaluationHostIsCurrent else {return}
                }
                guard !workspace.isBusy,!self.owner.isBusy,identity == nil || self.operationIdentity == identity else {return}
                // An independently evaluated saved checkpoint does not recreate
                // another provider's transient capture. Explicit View runs still
                // explain missing providers; automatic completion stays quiet.
                guard self.owner.document.edges.filter({$0.from.address.domain == .control && ["predictions","profiles","confusion","routing","routingSample","table","summary","episodes","moves","iterations","evaluation","campaign","opponents","performance"].contains($0.from.port) && $0.kind == .table}).allSatisfy({self.predictionOutputIsCurrent($0.from)}) else {return}
                _ = try self.startPredictionPreparation(policy:workspace.executionPolicy(for:self.owner.projectID))
            } catch {self.statusMessage=error.localizedDescription}
        }
    }
    private func currentCapturedTableArtifact(_ endpoint:WorkflowEndpoint)->TableViewQuerySession.Artifact? {
        guard predictionOutputIsCurrent(endpoint) else {return nil}
        if let session=selfPlay,session.address==endpoint.address {return session.artifacts[endpoint.port]}
        if let session=gameEpisodes,session.address==endpoint.address {return session.artifacts[endpoint.port]}
        if let artifact=backendFit?.predictionArtifact(for:endpoint) {return artifact}
        if let artifact=modelLoader.outputArtifact(endpoint) {return artifact}
        if let execution=currentExecution,execution.request.plan.train==endpoint.address {return execution.outputTableArtifact(endpoint.port)}
        if let evaluation,evaluation.request.plan.evaluate==endpoint.address {return evaluation.predictionTableArtifact}
        if let generation,generation.address==endpoint.address {return generation.artifact}
        return nil
    }
    func scopedDataPlan(at target:WorkflowNodeAddress,explainMissingCapture:Bool=false)throws->WorkflowScopedDataPreparation {
        try WorkflowScopedDataPreparation(document:owner.document,target:target,registry:owner.dataRegistry,capturedPin:{[self] endpoint in
            guard let artifact=currentCapturedTableArtifact(endpoint) else {
                if explainMissingCapture {throw DataFlowContractError(currentCaptureGuidance(for:endpoint.address))}
                throw WorkflowCapturedTableRequirement()
            }
            return artifact.sourceSHA256
        })
    }
    private func currentCaptureGuidance(for address:WorkflowNodeAddress)->String {
        let action:String
        switch owner.document.nodes.first(where:{$0.address==address})?.payload {
        case .estimator(let config):
            if PrebuiltModelConfiguration.isPrebuilt(config) {
                switch config["operation"]?.stringValue {
                case "predict":action=PrebuiltModelConfiguration.isCluster(config) ? "Prebuilt models’ Assign new rows":"Prebuilt models’ Predict rows"
                case "evaluate":action="Prebuilt models’ Evaluate model"
                default:action=PrebuiltModelConfiguration.isCluster(config) ? "Prebuilt models’ Fit population":"Prebuilt models’ Fit model"
                }
            } else {action="the connected model’s run action"}
        case .train:action="the connected Train component"
        case .evaluateTest:action="the connected Evaluate component"
        case .generateText:action="the connected Generate component"
        case .gameSession:action="the connected Game session"
        default:action="the connected producer"
        }
        return "Current captured rows are unavailable. Run \(action) explicitly, then run this selected data component. No producer was rerun."
    }
    @discardableResult private func startPredictionPreparation(policy:ResolvedExecutionPreferences)throws->Bool {
        guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy,let runtime=workspace.runtime else {throw DataFlowContractError("Wait for the current operation before preparing predictions.")}
        var artifacts:[WorkflowEndpoint:TableViewQuerySession.Artifact]=[:]
        for edge in owner.document.edges where ["predictions","profiles","confusion","routing","routingSample","table","summary","episodes","moves","iterations","evaluation","campaign","opponents","performance"].contains(edge.from.port) && edge.from.address.domain == .control && edge.kind == .table {
            guard predictionOutputIsCurrent(edge.from) else {throw DataFlowContractError(currentCaptureGuidance(for:edge.from.address))}
            if let session=selfPlay,session.address==edge.from.address,let artifact=session.artifacts[edge.from.port] {artifacts[edge.from]=artifact}
            else if let session=gameEpisodes,session.address==edge.from.address,let artifact=session.artifacts[edge.from.port] {artifacts[edge.from]=artifact}
            else if let artifact=backendFit?.predictionArtifact(for:edge.from) {artifacts[edge.from]=artifact}
            else if let artifact=modelLoader.outputArtifact(edge.from) {artifacts[edge.from]=artifact}
            else if let execution=currentExecution,execution.request.plan.train==edge.from.address,let artifact=execution.outputTableArtifact(edge.from.port) {artifacts[edge.from]=artifact}
            else if let evaluation,evaluation.request.plan.evaluate==edge.from.address,let artifact=evaluation.predictionTableArtifact {artifacts[edge.from]=artifact}
            else if let generation,generation.address==edge.from.address,let artifact=generation.artifact {artifacts[edge.from]=artifact}
            else {throw DataFlowContractError("Prediction-row capture is unavailable. Enable prediction capture and rerun its component.")}
        }
        let pins=artifacts.mapValues(\.sourceSHA256)
        let plan=try WorkflowPredictionPreparation(document:owner.document,pins:pins)
        let inputs=Dictionary(uniqueKeysWithValues:plan.sources.map {($0.key,artifacts[$0.value]!)})
        let session=try WorkflowDataPreparationSession(workspace:workspace,owner:owner,bindings:bindings,destinations:outputDestinations,history:dataPreparationHistory,runtime:runtime,policy:policy,launcher:executionLaunch,acceptedBudgetWindow:currentAPIBudgetWindow,predictionPlan:plan,predictionInputs:inputs)
        predictionPreparation=session;predictionPreparationPins=pins
        predictionPreparationObservations=[session.objectWillChange.sink {[weak self,weak session] _ in
            guard let self,let session,self.predictionPreparation === session,!self.isRetired else {return};self.objectWillChange.send();self.canvas.objectWillChange.send()
        },session.$statusMessage.sink {[weak self] in self?.statusMessage=$0}]
        try session.start();return true
    }
    func ownsPreparation(_ candidate:WorkflowDataPreparationSession)->Bool {
        currentDataPreparation === candidate || (predictionPreparation === candidate && predictionPreparationIsCurrent)
    }
    private var predictionPreparationIsCurrent:Bool {
        if let preparation=predictionPreparation {
            let realIDs=Set(owner.document.nodes.filter {$0.address.domain == .data}.map { $0.address.localID })
            guard sourceProvider.observedSourcesMatch(preparation.preparationOutputs.filter {realIDs.contains($0.key)}) else {return false}
        }
        return !predictionPreparationPins.isEmpty && predictionPreparationPins.allSatisfy {endpoint,pin in
            guard predictionOutputIsCurrent(endpoint) else {return false}
            if let session=selfPlay,session.address==endpoint.address {return session.artifacts[endpoint.port]?.sourceSHA256==pin}
            if let session=gameEpisodes,session.address==endpoint.address {return session.artifacts[endpoint.port]?.sourceSHA256==pin}
            if backendFit?.predictionArtifact(for:endpoint)?.sourceSHA256==pin && predictionOutputIsCurrent(endpoint) {return true}
            if modelLoader.outputArtifact(endpoint)?.sourceSHA256==pin {return true}
            if let execution=currentExecution,execution.request.plan.train==endpoint.address {return execution.outputTableArtifact(endpoint.port)?.sourceSHA256==pin}
            if endpoint.port=="table" {return generation?.artifact?.sourceSHA256==pin}
            return evaluation?.predictionTableArtifact?.sourceSHA256==pin
        }
    }
    private func publishTrainedOutput(_ outcome:WorkflowExecutionOutcome) {
        guard let workspace,evaluationHostIsCurrent,owner.document.schemaVersion>=17 else {clearModelFlow();return}
        guard outcome.status == .completed,outcome.recordError == nil,
              let directoryID=UUID(uuidString:outcome.directory.lastPathComponent),
              outcome.request.experiment == nil else {clearModelFlow();return}
        let train=outcome.request.plan.train,root=workspace.workspaceDirectory
        let sourceRevision=bindings.revision,checkpointRevision=checkpointBindings.revision,cacheRevision=cacheBindings.revision
        let id=UUID();modelArtifactID=id;isManagingTrainedArtifacts = !pendingModelConsumers.isEmpty
        statusMessage="Verifying Train’s completed model…"
        modelArtifactTask=Task { [self] in
            defer {if modelArtifactID==id {modelArtifactTask=nil;modelArtifactID=nil;isManagingTrainedArtifacts=false}}
            do {
                // Visible prepared or prediction tables can acquire bounded reads
                // at completion before this task starts. Join only current reads;
                // retain all idle/source/draft checks for model publication.
                for _ in 0..<50 where owner.isBusy && (predictionQueries.values.contains(where:{$0.1.isBusy}) || (capturedQueryRunID==outcome.runID && capturedQueries.values.contains(where:{$0.isBusy}))) {
                    try await Task.sleep(nanoseconds:100_000_000)
                    try Task.checkCancellation()
                    guard evaluationHostIsCurrent,modelArtifactID==id,currentExecution?.runID==outcome.runID else {throw CancellationError()}
                }
                let reference=try await WorkflowTrainedArtifactAuthority.acceptCompletedTrain(history:.init(projectID:owner.projectID,directoryID:directoryID),train:train,owner:owner,bindings:frozenBindings,root:root)
                try Task.checkCancellation()
                guard evaluationHostIsCurrent,modelArtifactID==id,bindings.revision==sourceRevision,
                      checkpointBindings.revision==checkpointRevision,cacheBindings.revision==cacheRevision else {throw DataFlowContractError("Private training inputs changed before model publication.")}
                guard frozenBindings.setReference(reference,for:train,lease:owner.lease(),expectedRevision:frozenBindings.revision) else {throw DataFlowContractError(frozenBindings.error ?? "Could not publish trained model.")}
                isManagingTrainedArtifacts=false
                statusMessage="Trained model ready · \(outcome.runID.uuidString.prefix(12))."
                advanceModelConsumers()
            } catch {clearModelFlow();if evaluationHostIsCurrent {statusMessage="Trained model unavailable: "+error.localizedDescription;schedulePredictionPreparation()}}
        }
    }
    @discardableResult func selectSelfPlayModel(_ record:SelfPlayRecord,for address:WorkflowNodeAddress)->Bool {
        do {
            guard let workspace,evaluationHostIsCurrent,!owner.isBusy,!workspace.isBusy,record.completed,record.document.projectID==owner.projectID,owner.document.nodes.contains(where:{if $0.address==address,case .frozenModel(let c)=$0.payload{return c.isEmpty};return false})else{throw WorkflowArchiveError("Select a completed paired model in the idle current project.")}
            var next=selfPlayBindings;next[address]=record.id
            try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(next),root:workspace.workspaceDirectory,parts:["self-play-bindings",owner.projectID,"bindings.json"])
            _=frozenBindings.setReference(nil,for:address,lease:owner.lease(),expectedRevision:frozenBindings.revision)
            selfPlayBindings=next;statusMessage="Paired policy/value checkpoint selected · "+String(record.id.prefix(8));return true
        }catch{statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func clearSelfPlayModel(_ address:WorkflowNodeAddress)->Bool {
        do{guard let workspace,!owner.isBusy else{return false};var next=selfPlayBindings;next[address]=nil;try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(next),root:workspace.workspaceDirectory,parts:["self-play-bindings",owner.projectID,"bindings.json"]);selfPlayBindings=next;return true}catch{statusMessage=error.localizedDescription;return false}
    }
    func isSelfPlaySource(_ consumer:WorkflowNodeAddress)->Bool {
        guard let source=try? WorkflowTrainedModelSource.resolve(in:owner.document,consumer:consumer)else{return false}
        return owner.document.nodes.contains{$0.address==source.address && SelfPlayConfiguration.isSelfPlay($0)} || selfPlayBindings[source.address] != nil || selfPlayRecords.contains{$0.address==source.address && $0.completed}
    }
    private func saveSelfPlayModel(_ record:SelfPlayRecord,address:WorkflowNodeAddress)throws->Bool {
        guard let workspace,let node=owner.document.nodes.first(where:{$0.address==address}),case .saveModel(let config)=node.payload,try WorkflowSaveModelConfiguration.decode(config).format == .package else {throw WorkflowArchiveError("Game policy/value exports use Portable package: paired architecture, weights and provenance; no optimizer resume.")}
        let access=try modelDestinations.acquire(for:address,lease:owner.lease()),lease=owner.lease(),root=workspace.workspaceDirectory,revision=modelDestinations.revision
        workspace.isPreparingData=true;owner.setExecutionBusy(true);isManagingTrainedArtifacts=true;let token=workspace.dataPreparationReservationID
        statusMessage="Verifying paired inference export…"
        modelArtifactTask=Task {[self] in
            defer{access.close();isManagingTrainedArtifacts=false;modelArtifactTask=nil;if workspace.dataPreparationReservationID==token {workspace.isPreparingData=false;owner.setExecutionBusy(false)}}
            do {
                let pending=try await Task.detached(priority:.userInitiated){()->WorkflowPendingExportPublication in
                    try access.check();return try WorkflowExportPublication.stageSelfPlay(record:record,root:root,to:access.destination)
                }.value
                defer{pending.close()};guard evaluationHostIsCurrent,owner.isCurrent(lease),modelDestinations.revision==revision else{throw WorkflowArchiveError("Source or destination changed before export.")};try access.check();try pending.commit()
                externalModelSaveReceipts[address]=access.destination;statusMessage="Paired inference package saved · nonresumable. Required resume state remains in History."
            }catch{statusMessage=error.localizedDescription}
        };return true
    }

    func reloadSelfPlayRecords() {
        guard let workspace else{return};let root=workspace.workspaceDirectory,project=owner.projectID
        Task {let records=(try? await WorkflowHistoryVerificationWork.run{try SelfPlayRecord.list(root:root,projectID:project)}) ?? [];guard evaluationHostIsCurrent else{return};selfPlayRecords=records;if let raw=try? DiagnosticFileIO.read(root:root,parts:["self-play-bindings",project,"bindings.json"],limit:1_000_000),let bindings=try? JSONDecoder().decode([WorkflowNodeAddress:String].self,from:raw){selfPlayBindings=bindings}}
    }
    @discardableResult func startSelfPlay(_ address:WorkflowNodeAddress,override:[String:JSONValue]?=nil,validateOnly:Bool=false,resume:SelfPlayRecord?=nil,campaignAction:String?=nil,continuingCampaign:Bool=false)->SelfPlaySession? {
        do {
            guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy,let runtime=workspace.runtime,workspace.allowExecution() else {throw WorkflowArchiveError("Use an idle workspace with its runtime available.")}
            guard validateOnly || !owner.hasPendingInspectorEdits else{throw WorkflowArchiveError("Apply or revert pending edits before Self-play.")}
            var document=owner.document
            if let override,let index=document.nodes.firstIndex(where:{$0.address==address}) {document.nodes[index].payload = .train(config:["selfPlay":.object(override)])}
            let env=SelfPlaySession.Environment(root:workspace.workspaceDirectory,accounting:workspace.applicationDirectory,runtime:runtime,policy:try workspace.executionPolicy(for:owner.projectID))
            let lease=owner.lease()
            let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                guard let self,let workspace,self.evaluationHostIsCurrent,!workspace.isBusy,!self.owner.isBusy else{return nil};workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return workspace.dataPreparationReservationID
            },release:{[weak self,weak workspace] token in
                guard let workspace,workspace.dataPreparationReservationID==token else{return};workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
            })
            if !validateOnly && !continuingCampaign {operationIdentity=UUID()}
            let session=SelfPlaySession(document:document,address:address,validateOnly:validateOnly,environment:env,reservation:reservation)
            if validateOnly {selfPlayValidation=session;selfPlayValidationObservation=session.objectWillChange.sink{[weak self] _ in self?.objectWillChange.send()}}
            else {selfPlayQueries.values.forEach{$0.invalidate()};selfPlayQueries=[:];selfPlay=session;selfPlayObservation=session.objectWillChange.sink{[weak self] _ in self?.objectWillChange.send()}}
            workspace.cancelDataPreparation={[weak session] in session?.cancel()}
            session.start(modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,resume:resume,campaignAction:campaignAction,isCurrent:{[weak self] in self?.evaluationHostIsCurrent==true && self?.owner.isCurrent(lease)==true},onFinish:{[weak self,weak session] in
                guard let self,let session else{return};self.statusMessage=session.message;self.reloadSelfPlayRecords()
                if !validateOnly,!session.artifacts.isEmpty {
                    if let campaign=session.campaign,campaign["complete"] == .bool(false),!session.pauseRequested,let root=self.workspace?.workspaceDirectory {
                        // Preserve the externally owned job identity across serial slices. Never overlap writers.
                        self.campaignContinuationTask=Task { [weak self] in
                            guard let self else{return};defer{self.campaignContinuationTask=nil}
                            guard !Task.isCancelled,self.evaluationHostIsCurrent,self.owner.isCurrent(lease) else{return}
                            do {
                                let record=try await WorkflowHistoryVerificationWork.run{try SelfPlayRecord.list(root:root,projectID:document.projectID).first(where:{$0.id==session.id.uuidString})}
                                guard let record else{throw WorkflowArchiveError("Committed campaign slice record is missing.")}
                                // A visible Results query may already own a brief reservation. Wait only
                                // for admission; never retry training, updates, or publication.
                                for _ in 0..<100 {
                                    guard !Task.isCancelled,self.evaluationHostIsCurrent,self.owner.isCurrent(lease)else{return}
                                    if self.workspace?.isBusy==false && !self.owner.isBusy{break}
                                    try await Task.sleep(for:.milliseconds(50))
                                }
                                guard self.workspace?.isBusy==false,!self.owner.isBusy else{throw WorkflowArchiveError("Campaign paused: another workspace operation is still active. Resume the committed boundary when idle.")}
                                _=self.startSelfPlay(address,resume:record,continuingCampaign:true)
                            }catch{self.statusMessage=error.localizedDescription}
                        }
                    }else{self.schedulePredictionPreparation()}
                }
            })
            statusMessage=validateOnly ? "Validating Self-play…":"Running Self-play…";return session
        }catch{statusMessage=error.localizedDescription;return nil}
    }
    func selfPlayOutputIsCurrent(_ endpoint:WorkflowEndpoint)->Bool {
        guard campaignContinuationTask == nil,let session=selfPlay,!session.isBusy,session.address==endpoint.address,session.artifacts[endpoint.port] != nil else{return false}
        return WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:endpoint.address)
    }
    func selfPlayTableViewSession(for endpoint:WorkflowEndpoint)->TableViewQuerySession? {
        guard let workspace,let session=selfPlay,selfPlayOutputIsCurrent(endpoint),let artifact=session.artifacts[endpoint.port]else{return nil}
        if let query=selfPlayQueries[endpoint]{return query}
        let query=workspace.makeTableViewQuerySession(artifact:artifact,projectID:owner.projectID,accepts:{[weak self] in self?.evaluationHostIsCurrent==true && self?.selfPlayOutputIsCurrent(endpoint)==true})
        selfPlayQueries[endpoint]=query;return query
    }

    func reloadGameRecords() {
        guard let workspace else{return};let root=workspace.workspaceDirectory,project=owner.projectID
        Task {let records=(try? await WorkflowHistoryVerificationWork.run{try GameEpisodeRecord.list(root:root,projectID:project)}) ?? [];guard evaluationHostIsCurrent else{return};gameRecords=records}
    }
    @discardableResult func startGameSession(_ address:WorkflowNodeAddress,override:[String:JSONValue]?=nil,validateOnly:Bool=false,continuingOperation:Bool=false)->GameEpisodeSession? {
        guard continuingOperation || !hasActiveOperation else {statusMessage="Wait for the active operation or cancel it.";return nil}
        do {
            guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy,let runtime=workspace.runtime,workspace.allowExecution() else {throw WorkflowArchiveError("Use an idle workspace with its Python runtime available.")}
            var document=owner.document
            if let override,let index=document.nodes.firstIndex(where:{$0.address==address}) {document.nodes[index].payload = .gameSession(config:override)}
            guard let target=document.nodes.first(where:{$0.address==address}),case .gameSession(let settings)=target.payload else {throw DataFlowContractError("Select a Game session.")}
            _=try GameSessionConfiguration(settings)
            let env=GameEpisodeSession.Environment(root:workspace.workspaceDirectory,accounting:workspace.applicationDirectory,runtime:runtime,policy:try workspace.executionPolicy(for:owner.projectID))
            let lease=owner.lease()
            let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                guard let self,let workspace,self.evaluationHostIsCurrent,!workspace.isBusy,!self.owner.isBusy else{return nil};workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return workspace.dataPreparationReservationID
            },release:{[weak self,weak workspace] token in
                guard let workspace,workspace.dataPreparationReservationID==token else{return};workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
            })
            let session=GameEpisodeSession(document:document,address:address,validateOnly:validateOnly,environment:env,reservation:reservation)
            if !validateOnly && !continuingOperation {clearModelFlow();operationIdentity=UUID();apiRunBudget=nil}
            let admittedIdentity=operationIdentity
            if validateOnly {gameValidation=session;gameValidationObservation=session.objectWillChange.sink{[weak self] _ in self?.objectWillChange.send()}}
            else {gameQueries.values.forEach{$0.invalidate()};gameQueries=[:];gameEpisodes=session;gameObservation=session.objectWillChange.sink{[weak self] _ in self?.objectWillChange.send()}}
            workspace.cancelDataPreparation={[weak session] in session?.cancel()}
            var references:[WorkflowNodeAddress:WorkflowFrozenModelReference]=[:]
            for node in document.nodes {if let ref=frozenBindings.reference(for:node.address){references[node.address]=ref}}
            var paired:[WorkflowNodeAddress:SelfPlayRecord]=[:]
            for record in selfPlayRecords.reversed() where record.completed && document.nodes.contains(where:{$0.address==record.address && SelfPlayConfiguration.isSelfPlay($0)}) {paired[record.address]=record}
            for (address,id) in selfPlayBindings where frozenBindings.reference(for:address)==nil {if let record=selfPlayRecords.first(where:{$0.id==id && $0.completed}){paired[address]=record}}
            session.start(modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,references:references,paired:paired,isCurrent:{[weak self] in self?.evaluationHostIsCurrent==true && self?.owner.isCurrent(lease)==true},onFinish:{[weak self,weak session] in
                guard let self,let session,self.operationIdentity==admittedIdentity,self.owner.isCurrent(lease),self.evaluationHostIsCurrent else{return};self.statusMessage=session.message;if !validateOnly {self.reloadGameRecords()}
                if !validateOnly,!session.artifacts.isEmpty,self.owner.document.matchesCapturedConstruction(session.document),WorkflowPredictionPreparation.dependentData(in:self.owner.document).count>0 {
                    // Explicit Run Flow includes its downstream data consumers; validation never does.
                    self.schedulePredictionPreparation(ownedBy:admittedIdentity)
                }
            })
            statusMessage=validateOnly ? "Validating game bindings…":"Running game session…";return session
        } catch {statusMessage=error.localizedDescription;return nil}
    }
    func gameOutputIsCurrent(_ endpoint:WorkflowEndpoint)->Bool {
        guard evaluationHostIsCurrent,let session=gameEpisodes,!session.isBusy,session.address==endpoint.address,session.artifacts[endpoint.port] != nil else{return false}
        return session.boundReferences.allSatisfy{frozenBindings.reference(for:$0.key)==$0.value} && WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:endpoint.address)
    }
    func gameTableViewSession(_ endpoint:WorkflowEndpoint)->TableViewQuerySession? {
        guard let workspace,let session=gameEpisodes,gameOutputIsCurrent(endpoint),let artifact=session.artifacts[endpoint.port] else{return nil}
        if let query=gameQueries[endpoint]{return query}
        let accepts={ [weak self] in self?.gameOutputIsCurrent(endpoint)==true }
        let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
            guard let self,let workspace,accepts(),!workspace.isBusy,!self.owner.isBusy else{return nil};workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return workspace.dataPreparationReservationID
        },release:{[weak self,weak workspace] token in guard let workspace,workspace.dataPreparationReservationID==token else{return};workspace.isPreparingData=false;self?.owner.setExecutionBusy(false)})
        let query=TableViewQuerySession(artifact:artifact,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,projectID:owner.projectID,runtime:session.environment.runtime,storageLimitBytes:session.environment.policy.storageLimitBytes,reservation:reservation,accepts:accepts)
        gameQueries[endpoint]=query;return query
    }
    func makeMixtureValidationSession()->MixtureValidationSession {
        MixtureValidationSession(environment:{[weak self,weak workspace] in
            guard let self,let workspace,self.evaluationHostIsCurrent,let runtime=workspace.runtime,let policy=try? workspace.executionPolicy(for:self.owner.projectID) else{return nil}
            return .init(root:workspace.workspaceDirectory,accounting:workspace.applicationDirectory,projectID:self.owner.projectID,runtime:runtime,policy:policy)
        },acquire:{[weak self,weak workspace] cancel in
            guard let self,let workspace,self.evaluationHostIsCurrent,!workspace.isBusy,!self.owner.isBusy else{return nil}
            workspace.cancelDataPreparation=cancel;let token=workspace.dataPreparationReservationID
            workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return token
        },release:{[weak self,weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID==token else{return}
            workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
        })
    }
    func makeGameRuleValidationSession()->GameRuleValidationSession {
        GameRuleValidationSession(environment:{[weak self,weak workspace] in
            guard let self,let workspace,self.evaluationHostIsCurrent,let runtime=workspace.runtime,let policy=try? workspace.executionPolicy(for:self.owner.projectID) else{return nil}
            return .init(root:workspace.workspaceDirectory,accounting:workspace.applicationDirectory,projectID:self.owner.projectID,runtime:runtime,policy:policy)
        },acquire:{[weak self,weak workspace] cancel in
            guard let self,let workspace,self.evaluationHostIsCurrent,!workspace.isBusy,!self.owner.isBusy else{return nil}
            workspace.cancelDataPreparation=cancel;let token=workspace.dataPreparationReservationID
            workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return token
        },release:{[weak self,weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID==token else{return}
            workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
        })
    }
    private func modelLoaderContext()throws->(RuntimeLocation,ResolvedExecutionPreferences) {
        guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy,let runtime=workspace.runtime else {throw WorkflowArchiveError("Use an idle current workspace with an available Python runtime.")}
        return (runtime,try workspace.executionPolicy(for:owner.projectID))
    }
    func chooseLocalModel(_ address:WorkflowNodeAddress) {do {let (runtime,policy)=try modelLoaderContext();modelLoader.chooseLocal(address,runtime:runtime,policy:policy)} catch {modelLoader.show(error.localizedDescription,for:address)}}
    func resolveModelHub(_ address:WorkflowNodeAddress,credential:String?) {do {let (_,policy)=try modelLoaderContext();modelLoader.resolve(address,bearer:try credential.map{try ModelHubCredentials.read($0)},maximumSeconds:min(45,Double(policy.budget.maximumRunSeconds)))} catch {modelLoader.show(error.localizedDescription,for:address)}}
    func downloadModelHub(_ address:WorkflowNodeAddress,credential:String?) {do {let (runtime,policy)=try modelLoaderContext();modelLoader.download(address,bearer:try credential.map{try ModelHubCredentials.read($0)},runtime:runtime,policy:policy)} catch {modelLoader.show(error.localizedDescription,for:address)}}
    func previewModelTokens(_ address:WorkflowNodeAddress,text:String) {do {let (runtime,policy)=try modelLoaderContext();modelLoader.preview(address,text:text,runtime:runtime,policy:policy)} catch {modelLoader.show(error.localizedDescription,for:address)}}
    func modelDestinationSummary(for address:WorkflowNodeAddress)->String {modelDestinations.summary(for:address)}
    @discardableResult func selectSaveModelDestination(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {
        guard evaluationHostIsCurrent,let workspace,!workspace.isBusy,owner.isCurrent(lease),!owner.hasPendingInspectorEdits,
              let node=owner.document.nodes.first(where:{$0.address==address}),case .saveModel(let values)=node.payload,
              let config=try? WorkflowSaveModelConfiguration.decode(values) else {return false}
        // Every export stages and publishes beneath a parent directory. A save-panel
        // leaf grant cannot authorize that directory in the signed sandbox.
        let panel = WorkflowExportParentChooser.panel(title: "Choose model parent folder", message: "Choose the folder that will contain the new model export.")
        guard panel.runModal() == .OK,let parent=panel.url else {return false}
        let checkpoint=config.format == .checkpoint
        let alert=NSAlert();alert.messageText=checkpoint ? "New checkpoint file":"New model folder"
        alert.informativeText="Choose a new name inside the selected folder. Existing files and folders will not be replaced."
        let name=NSTextField(string:checkpoint ? "model.pt":"KwipML model");name.frame=NSRect(x:0,y:0,width:280,height:24);alert.accessoryView=name;alert.addButton(withTitle:"Choose destination");alert.addButton(withTitle:"Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else {return false}
        do {try WorkflowExportPublication.validateExclusiveLeaf(name.stringValue)} catch {statusMessage=error.localizedDescription;return false}
        let accepted=modelDestinations.set(destination:parent.appendingPathComponent(name.stringValue),for:address,lease:lease)
        statusMessage=modelDestinations.statusMessage;return accepted
    }

    @discardableResult func clearSaveModelDestination(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {
        guard evaluationHostIsCurrent,let workspace,!workspace.isBusy else {return false}
        let accepted=modelDestinations.clear(for:address,lease:lease);statusMessage=modelDestinations.statusMessage;return accepted
    }
    func reloadGenerationRecords() {
        if let runtime=workspace?.runtime {modelLoader.restoreGeneration(runtime:runtime)}
        guard let workspace,!isRetired else {return}
        generationRefreshTask?.cancel()
        let root=workspace.workspaceDirectory,project=owner.projectID
        generationRefreshTask=Task { [weak self] in
            do {
                let records=try await Task.detached(priority:.utility) {try WorkflowGenerationRecord.list(root:root,projectID:project)}.value
                guard let self,!Task.isCancelled,!self.isRetired else {return}
                self.generationRecords=records
                if self.generation==nil,let runtime=workspace.runtime,let record=records.first(where:{WorkflowOutputConstruction.matches(self.owner.document,captured:$0.document,provider:$0.address)}) {
                    let verified=try await WorkflowHistoryVerificationWork.run {try record.verify(root:root)}
                    guard !Task.isCancelled,self.generation==nil,WorkflowOutputConstruction.matches(self.owner.document,captured:record.document,provider:record.address) else {return}
                    self.generation = .restored(record,directory:record.directory(root:root),runtime:runtime,verified:verified)
                    self.canvas.objectWillChange.send()
                }
            } catch {self?.statusMessage="Captured generation could not be restored: "+error.localizedDescription}
        }
    }
    func generationModelMetadata(for address:WorkflowNodeAddress)async throws->WorkflowGenerationModelMetadata {
        guard let workspace,evaluationHostIsCurrent,!owner.isBusy else {throw WorkflowArchiveError("Wait for the current operation.")}
        let source=try WorkflowTrainedModelSource.resolve(in:owner.document,consumer:address)
        if let imported=modelLoader.reference(source.address) {
            let special:[String:JSONValue]
            if case .object(let values)=imported.metadata["specialTokens"] {special=values} else {special=[:]}
            return .init(context:Int(imported.metadata["maximumTokens"]?.integerValue ?? 0),vocabularySize:Int(imported.metadata["vocabularySize"]?.integerValue ?? 0),hasUnknown:special["unk_token"]?.integerValue != nil,hasEOS:special["eos_token"]?.integerValue != nil)
        }
        let reference=try await WorkflowTrainedArtifactAuthority.resolve(source:source,owner:owner,bindings:frozenBindings,root:workspace.workspaceDirectory)
        let key=reference.runID+reference.modelManifestSHA256+reference.checkpointSHA256
        if let metadata=generationMetadataCache[key] {return metadata}
        let root=workspace.workspaceDirectory,accounting=workspace.applicationDirectory,limit=try workspace.executionPolicy(for:owner.projectID).storageLimitBytes
        let metadata=try await WorkflowHistoryVerificationWork.run {
            guard let id=UUID(uuidString:reference.directoryID) else {throw WorkflowFileLeaseError.unsafePath}
            let package=try WorkflowHistoryReader.stageTrainedExport(.init(projectID:reference.projectID,directoryID:id),root:root,accountingRoot:accounting,storageLimitBytes:limit)
            defer {package.remove()}
            guard package.runID==reference.runID,package.requestIdentity==reference.requestIdentity,package.modelManifestSHA256==reference.modelManifestSHA256,package.files.first(where:{$0.path=="checkpoint.pt"})?.sha256==reference.checkpointSHA256 else {throw WorkflowFileLeaseError.unsafePath}
            return try WorkflowGenerationModelMetadata.read(package)
        }
        try Task.checkCancellation()
        guard evaluationHostIsCurrent,frozenBindings.reference(for:source.address)==reference else {throw CancellationError()}
        if generationMetadataCache.count>=8 {generationMetadataCache.removeAll()}
        generationMetadataCache[key]=metadata;return metadata
    }
    func generationPromptColumns(for address:WorkflowNodeAddress)async->[String] {
        guard let edge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="prompts"}),let artifact=currentDataPreparation?.generationPromptArtifact(for:edge.from) ?? capturedTableContext?.tables.artifacts[edge.from],artifact.inputKind == "table" else {return []}
        return (try? await Task.detached(priority:.userInitiated) {
            let bytes=try DiagnosticFileIO.read(root:artifact.indexURL.deletingLastPathComponent(),parts:[artifact.indexURL.lastPathComponent],limit:8_000_000)
            guard WorkflowArchiveCodec.hash(bytes)==artifact.sourceSHA256,case .object(let table)=try JSONDecoder().decode(JSONValue.self,from:bytes),case .array(let columns)=table["columns"] else {throw WorkflowFileLeaseError.unsafePath}
            return columns.compactMap{value -> String? in guard case .object(let c)=value,c["dtype"] == .string("string") else {return nil};return c["name"]?.stringValue}
        }.value) ?? []
    }
    @discardableResult func generateText(address:WorkflowNodeAddress,continuingOperation:Bool=false)->Bool {
        guard continuingOperation || !hasActiveOperation else {statusMessage="Wait for the active operation or cancel it.";return false}
        guard let workspace,evaluationHostIsCurrent,!isStagingEvaluation,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,let runtime=workspace.runtime else {statusMessage="Use the idle current workflow and apply pending edits.";return false}
        do {
            guard workspace.allowExecution() else {statusMessage=workspace.statusMessage;return false}
            let source=try WorkflowTrainedModelSource.resolve(in:owner.document,consumer:address)
            guard let node=owner.document.nodes.first(where:{$0.address==address}),case .generateText(let values)=node.payload else {return false}
            if !continuingOperation {clearModelFlow();operationIdentity=UUID()}
            if modelLoader.reference(source.address) != nil {
                let configuration=try TextGenerationConfiguration(values),policy=try workspace.executionPolicy(for:owner.projectID),lease=owner.lease()
                let promptEdge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="prompts"})
                let prompts=promptEdge.flatMap{currentDataPreparation?.generationPromptArtifact(for:$0.from) ?? capturedTableContext?.tables.artifacts[$0.from]}
                generationQuery?.invalidate();generationQuery=nil
                let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                    guard let self,let workspace,self.owner.isCurrent(lease),!workspace.isBusy,!self.owner.isBusy else {return nil}
                    workspace.isPreparingData=true;self.owner.setExecutionBusy(true);workspace.cancelDataPreparation={self.modelLoader.cancel()};return workspace.dataPreparationReservationID
                },release:{[weak self,weak workspace] token in
                    guard let workspace,workspace.dataPreparationReservationID==token else {return};workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
                })
                let started=modelLoader.generate(address:address,sourceAddress:source.address,configuration:configuration,promptArtifact:prompts,runtime:runtime,policy:policy,reservation:reservation,onTerminal:{[weak self,flowID=modelFlowID] success in self?.finishImportedModelOperation(success,continuingOperation:continuingOperation,flowID:flowID)})
                if !started {statusMessage=modelLoader.status;clearModelFlow()}
                return started
            }
            let configuration=try TextGenerationConfiguration(values),document=owner.document,root=workspace.workspaceDirectory
            let policy=try workspace.executionPolicy(for:owner.projectID),lease=owner.lease(),binding=frozenBindings.revision
            let promptEdge=document.edges.first(where:{$0.to.address==address && $0.to.port=="prompts"})
            let prompts=promptEdge.flatMap{currentDataPreparation?.generationPromptArtifact(for:$0.from) ?? capturedTableContext?.tables.artifacts[$0.from]}
            isManagingTrainedArtifacts=true
            modelArtifactTask=Task { [self,weak workspace] in
                defer {modelArtifactTask=nil;isManagingTrainedArtifacts=false}
                do {
                    let reference=try await WorkflowTrainedArtifactAuthority.resolve(source:source,owner:owner,bindings:frozenBindings,root:root)
                    guard let workspace,evaluationHostIsCurrent,owner.isCurrent(lease),frozenBindings.revision==binding else {throw CancellationError()}
                    let next=WorkflowGenerationSession(address:address,document:document,runtime:runtime)
                    let current={ [weak self] in guard let self else {return false};return self.evaluationHostIsCurrent && self.owner.document.matchesCapturedConstruction(document) && self.frozenBindings.revision==binding}
                    let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                        guard let self,let workspace,current(),!workspace.isPreparingData,!self.owner.isBusy else {return nil}
                        workspace.isPreparingData=true;self.owner.setExecutionBusy(true)
                        workspace.cancelDataPreparation={next.cancel()};return workspace.dataPreparationReservationID
                    },release:{[weak self,weak workspace] token in
                        guard let workspace,workspace.dataPreparationReservationID==token else {return}
                        workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
                    })
                    generationQuery?.invalidate();generationQuery=nil;generation=next;generationIsLatest=true
                    generationObservation=next.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send();self?.canvas.objectWillChange.send()}
                    try next.start(reference:reference,configuration:configuration,prompts:prompts,workspace:root,accountingRoot:workspace.applicationDirectory,policy:policy,reservation:reservation,isCurrent:current,onTerminal:{[weak self,weak next,flowID=modelFlowID] in
                        guard let self else {return}
                        self.statusMessage=next?.statusMessage ?? "Generation ended";self.reloadGenerationRecords()
                        self.finishModelConsumer(next?.record?.status == "completed",continuingOperation:continuingOperation,flowID:flowID)
                    })
                } catch {statusMessage=error.localizedDescription}
            }
            return true
        } catch {statusMessage=error.localizedDescription;return false}
    }
    func generationOutputIsCurrent(_ endpoint:WorkflowEndpoint)->Bool {
        if evaluationHostIsCurrent,modelLoader.generationIsCurrent(endpoint) {return true}
        guard evaluationHostIsCurrent,let generation,generation.address==endpoint.address,endpoint.port=="table",!generation.isBusy,generation.artifact != nil,let record=generation.record,let source=try? WorkflowTrainedModelSource.resolve(in:owner.document,consumer:endpoint.address),frozenBindings.reference(for:source.address)==record.reference else {return false}
        return WorkflowOutputConstruction.matches(owner.document,captured:generation.document,provider:endpoint.address)
    }
    func generationTableViewSession(for endpoint:WorkflowEndpoint,usesSharedSession:Bool=true)->TableViewQuerySession? {
        if let workspace,modelLoader.generationIsCurrent(endpoint),let record=modelLoader.generationRecord,let runtime=modelLoader.generationRuntime,let table=record.files.first(where:{$0.path=="output/table.json"}) {
            if usesSharedSession,let generationQuery {return generationQuery}
            let artifact=TableViewQuerySession.Artifact(indexURL:record.directory(workspace.workspaceDirectory).appendingPathComponent(table.path),indexSHA256:table.sha256,sourceSHA256:table.sha256,inputKind:"table")
            let accepts={ [weak self] in self?.evaluationHostIsCurrent==true && self?.modelLoader.generationIsCurrent(endpoint)==true && self?.modelLoader.generationRecord?.operationID==record.operationID}
            let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                guard let self,let workspace,accepts(),!workspace.isBusy,!self.owner.isBusy else {return nil};workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return workspace.dataPreparationReservationID
            },release:{[weak self,weak workspace] token in guard let workspace,workspace.dataPreparationReservationID==token else {return};workspace.isPreparingData=false;self?.owner.setExecutionBusy(false)})
            let query=TableViewQuerySession(artifact:artifact,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,projectID:owner.projectID,runtime:runtime,storageLimitBytes:(try? workspace.executionPolicy(for:owner.projectID).storageLimitBytes) ?? nil,reservation:reservation,accepts:accepts)
            if usesSharedSession {generationQuery=query};return query
        }
        guard let workspace,let generation,generationOutputIsCurrent(endpoint),let artifact=generation.artifact else {return nil}
        if usesSharedSession,let generationQuery {return generationQuery}
        final class GenerationQueryCancellation {weak var session:TableViewQuerySession?}
        let cancellation=GenerationQueryCancellation()
        let accepts={ [weak self,weak generation] in guard let self,let generation else {return false};return self.generationOutputIsCurrent(endpoint) && generation.artifact?.sourceSHA256==artifact.sourceSHA256}
        let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
            guard let self,let workspace,accepts(),!workspace.isBusy,!self.owner.isBusy else {return nil}
            workspace.isPreparingData=true;self.owner.setExecutionBusy(true);workspace.cancelDataPreparation={cancellation.session?.cancel()};return workspace.dataPreparationReservationID
        },release:{[weak self,weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID==token else {return};workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
        })
        let query=TableViewQuerySession(artifact:artifact,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,projectID:owner.projectID,runtime:generation.runtime,storageLimitBytes:(try? workspace.executionPolicy(for:owner.projectID).storageLimitBytes) ?? nil,reservation:reservation,accepts:accepts)
        cancellation.session=query;if usesSharedSession{generationQuery=query};return query
    }
    func learnedPreparationAvailable(for save:WorkflowNodeAddress)->Bool {
        guard let source=try? WorkflowTrainedModelSource.resolve(in:owner.document,consumer:save),let fit=completedDocumentFits[source.address] ?? backendFit,fit.plan.estimator==source.address,fit.phase == .completed,!fit.plan.configuration.isPrediction,WorkflowOutputConstruction.matches(owner.document,captured:fit.document,provider:source.address),bindings.revision==fit.bindingRevision,case .object=fit.manifest?["learnedPreparation"] else{return false}
        return true
    }
    private func saveDocumentClassifier(address:WorkflowNodeAddress,plan:WorkflowSaveModelPlan)throws->Bool {
        guard let workspace,plan.configuration.format == .package,let session=completedDocumentFits[plan.modelSource.address] ?? backendFit,session.plan.estimator==plan.modelSource.address,session.plan.configuration.hasTableOutputs,!session.plan.configuration.isPrediction,
              session.phase == .completed,WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:session.plan.estimator),bindings.revision==session.bindingRevision else {throw WorkflowArchiveError("Complete the current model fit, then choose Portable package. Saving never retrains.")}
        let lease=owner.lease(),revision=modelDestinations.revision,access=try modelDestinations.acquire(for:address,lease:lease),id=UUID()
        isManagingTrainedArtifacts=true;modelArtifactID=id;statusMessage="Verifying the model inference package…"
        modelArtifactTask=Task { [self] in
            defer {access.close();if modelArtifactID==id {modelArtifactID=nil;modelArtifactTask=nil;isManagingTrainedArtifacts=false}}
            do {
                let output=try await session.verifiedOutput()
                let learned=plan.configuration.preparation == .learned
                guard !learned || session.manifest?["learnedPreparation"] != nil else {throw WorkflowArchiveError("This accepted fit has no complete supported learned preparation chain. Save prepared inputs only, or complete a supported Preprocess fit.")}
                let source=output.appendingPathComponent(learned ? "portable-prepared":"portable"),destination=access.destination
                let worker=Task.detached(priority:.userInitiated) {()->WorkflowPendingExportPublication in
                    try access.check()
                    let raw=try DiagnosticFileIO.read(root:source,parts:["manifest.json"],limit:2_000_000)
                    struct Inventory:Decodable {let artifacts:[WorkflowArchiveFile]}
                    let files=try JSONDecoder().decode(Inventory.self,from:raw).artifacts + [.init(path:"manifest.json",bytes:raw.count,sha256:WorkflowArchiveCodec.hash(raw))]
                    guard files.count<=16,files.reduce(0,{$0+$1.bytes})<=60_000_000,Set(files.map(\.path)).count==files.count,Set(["state.json","inference.py","manifest.json"]).isSubset(of:Set(files.map(\.path))) && (files.contains(where:{$0.path=="document_classification.py"}) || files.contains(where:{$0.path=="prebuilt_models.py"})) else {throw WorkflowArchiveError("Invalid classifier export inventory.")}
                    if learned {
                        guard files.contains(where:{$0.path=="preparation.json"}) else {throw WorkflowArchiveError("The learned preparation package is incomplete.")}
                        _=try PrebuiltPreparationAttachment.decode(DiagnosticFileIO.read(root:source,parts:["preparation.json"],limit:PrebuiltPreparationAttachment.maximumBytes),modelBytes:DiagnosticFileIO.read(root:source,parts:["state.json"],limit:32_000_000))
                    } else if files.contains(where:{$0.path=="preparation.json"}) {throw WorkflowArchiveError("Prepared-input export must not change its input contract.")}
                    let pending=try WorkflowPendingExportPublication(destination:destination,directory:true)
                    do {for file in files {try pending.writeClassifierFile(file,from:source)};try access.check();return pending} catch {pending.close();throw error}
                }
                let publication=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                defer {publication.close()}
                guard evaluationHostIsCurrent,owner.isCurrent(lease),modelDestinations.revision==revision,completedDocumentFits[session.plan.estimator] === session,WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:session.plan.estimator) else {throw WorkflowArchiveError("Classifier or destination changed while saving.")}
                try access.check();try publication.commit()
                externalModelSaveReceipts[address]=destination;statusMessage="Saved verified model package. See its input contract and feature/class order; no training rows included."
            } catch {statusMessage=error.localizedDescription}
        }
        return true
    }
    @discardableResult func saveModel(address:WorkflowNodeAddress,continuingOperation:Bool=false)->Bool {
        guard continuingOperation || !hasActiveOperation else {statusMessage="Wait for the active operation or cancel it.";return false}
        guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits else {statusMessage="Use the idle current workflow to save a model.";return false}
        do {
            let plan=try WorkflowSaveModelPlan.compile(owner.document,save:address)
            // A direct inspector/MCP export owns a new operation; queued consumers
            // remain part of their parent Train/Evaluate operation for cancellation.
            if !continuingOperation {clearModelFlow();operationIdentity=UUID()}
            if let record=selfPlayRecords.first(where:{$0.id==selfPlayBindings[plan.modelSource.address] || ($0.address==plan.modelSource.address && $0.completed)}) {return try saveSelfPlayModel(record,address:address)}
            if plan.modelSource.isEstimator {return try saveDocumentClassifier(address:address,plan:plan)}
            if let reference=modelLoader.reference(plan.modelSource.address) {
                guard plan.configuration.format == .huggingFace else {throw WorkflowArchiveError("Choose Hugging Face folder to save this imported model.")}
                let content=plan.configuration.effectiveContent(isLoRA:reference.metadata["fineTuningMode"] == .string("lora"))
                if content == .merged,reference.metadata["fineTuningMode"] != .string("lora") {throw WorkflowArchiveError("Merged output requires a completed LoRA result.")}
                let policy=try workspace.executionPolicy(for:owner.projectID)
                guard let runtime=workspace.runtime else {throw WorkflowArchiveError("Python runtime unavailable.")}
                let access=try modelDestinations.acquire(for:address,lease:owner.lease()),lease=owner.lease(),revision=modelDestinations.revision
                if content == .adapter || content == .continuation {
                    guard reference.metadata["trainingOperation"] == .bool(true) else {access.close();throw WorkflowArchiveError("These contents require a completed fine-tuning result.")}
                    let started=modelLoader.exportTraining(reference,address:address,content:content,access:access,isCurrent:{[weak self] in self?.evaluationHostIsCurrent==true && self?.owner.isCurrent(lease)==true && self?.modelDestinations.revision==revision},onTerminal:{[weak self,flowID=modelFlowID] success in self?.finishImportedModelOperation(success,continuingOperation:continuingOperation,flowID:flowID)},onSaved:{[weak self] url in self?.externalModelSaveReceipts[address]=url})
                    if !started {statusMessage=modelLoader.status;clearModelFlow()};return started
                }
                let started=modelLoader.exportModel(reference,address:address,shardBytes:plan.configuration.shardBytes,access:access,runtime:runtime,policy:policy,isCurrent:{[weak self] in self?.evaluationHostIsCurrent==true && self?.owner.isCurrent(lease)==true && self?.modelDestinations.revision==revision},onTerminal:{[weak self,flowID=modelFlowID] success in self?.finishImportedModelOperation(success,continuingOperation:continuingOperation,flowID:flowID)},onSaved:{[weak self] url in self?.externalModelSaveReceipts[address]=url})
                if !started {statusMessage=modelLoader.status;clearModelFlow()};return started
            }
            if plan.configuration.format == .huggingFace {throw WorkflowArchiveError("This canvas model has no Hugging Face export adapter. Use Portable package or Checkpoint.")}
            guard modelDestinations.path(for:address) != nil else {throw DataFlowContractError("Choose Save model’s destination before running.")}
            if plan.modelSource.isTrain,frozenBindings.reference(for:plan.modelSource.address)==nil {return startModelConsumer(address,forceTraining:true)}
            let policy=try workspace.executionPolicy(for:owner.projectID),lease=owner.lease(),root=workspace.workspaceDirectory
            let destinationRevision=modelDestinations.revision,id=UUID();modelArtifactID=id;isManagingTrainedArtifacts=true
            modelArtifactTask=Task { [self,weak workspace] in
                defer {if modelArtifactID==id {modelArtifactID=nil;modelArtifactTask=nil;isManagingTrainedArtifacts=false}}
                do {
                    let reference=try await WorkflowTrainedArtifactAuthority.resolve(source:plan.modelSource,owner:owner,bindings:frozenBindings,root:root)
                    try Task.checkCancellation()
                    guard let workspace,evaluationHostIsCurrent,modelArtifactID==id,owner.isCurrent(lease),modelDestinations.revision==destinationRevision else {throw DataFlowContractError("Model or destination changed before saving.")}
                    let access=try modelDestinations.acquire(for:address,lease:lease)
                    let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                        guard let self,let workspace,self.evaluationHostIsCurrent,!workspace.isRunning,!workspace.isPreparingData,!workspace.isPreviewing,!workspace.isInspectingData,!self.owner.isBusy,self.owner.isCurrent(lease) else {return nil}
                        workspace.isPreparingData=true;workspace.cancelDataPreparation={ [weak self] in self?.cancel() }
                        self.owner.setExecutionBusy(true);return workspace.dataPreparationReservationID
                    },release:{[weak self,weak workspace] token in
                        guard let self,let workspace,workspace.dataPreparationReservationID==token else {return}
                        self.owner.setExecutionBusy(false);workspace.isPreparingData=false;workspace.cancelDataPreparation=nil
                    })
                    let session=WorkflowSaveModelSession(reference:reference,format:plan.configuration.format,root:root,accountingRoot:workspace.applicationDirectory,storageLimitBytes:policy.storageLimitBytes ?? Int64.max,reservation:reservation,isCurrent:{[weak self] in
                        guard let self else {return false}
                        return self.evaluationHostIsCurrent && self.owner.isCurrent(lease) && self.modelDestinations.revision==destinationRevision && self.frozenBindings.reference(for:plan.modelSource.address)==reference
                    },onSaved:{[weak self,flowID=modelFlowID] receipt in
                        guard let self else {return};self.modelSaveReceipts.insert(receipt,at:0)
                        self.modelSaveReceiptsByAddress[address]=receipt
                        if self.modelSaveReceipts.count>100 {self.modelSaveReceipts.removeLast()}
                        self.finishModelConsumer(true,continuingOperation:continuingOperation,flowID:flowID)
                    },onFailed:{[weak self] in self?.clearModelFlow()})
                    modelSave=session
                    modelSaveObservations=[session.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()},session.$statusMessage.sink {[weak self] message in self?.statusMessage=message}]
                    isManagingTrainedArtifacts=false
                    if !session.save(using:access) {access.close();clearModelFlow();statusMessage=session.statusMessage}
                } catch {clearModelFlow();if evaluationHostIsCurrent {statusMessage="Could not save model: "+error.localizedDescription}}
            }
            return true
        } catch {statusMessage=error.localizedDescription;clearModelFlow();return false}
    }

    @discardableResult func selectFrozenModel(_ reference:WorkflowHistoryReference,for address:WorkflowNodeAddress)async->Bool {
        guard let workspace,evaluationHostIsCurrent,!isStagingEvaluation,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,
              let node=owner.document.nodes.first(where:{$0.address==address}),case .frozenModel=node.payload else {return false}
        let lease=owner.lease(),revision=frozenBindings.revision,id=UUID(),select=selectFrozenReference,root=workspace.workspaceDirectory
        evaluationStagingID=id;isStagingEvaluation=true;retainedHost=self;retainedWorkspace=workspace
        statusMessage="Verifying frozen model selection…"
        let task=Task {try await select(reference,root)}
        frozenSelectionTask=task
        defer {if evaluationStagingID==id {evaluationStagingID=nil;isStagingEvaluation=false;frozenSelectionTask=nil;retainedHost=nil;retainedWorkspace=nil}}
        do {
            let selected=try await withTaskCancellationHandler(operation:{try await task.value},onCancel:{task.cancel()})
            try Task.checkCancellation()
            guard !task.isCancelled,evaluationHostIsCurrent,evaluationStagingID==id,selected.projectID==reference.projectID,selected.directoryID==reference.directoryID.uuidString,owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits,frozenBindings.revision==revision else {throw WorkflowArchiveError("Frozen selection changed while verification was pending.")}
            let accepted=frozenBindings.setReference(selected,for:address,lease:lease,expectedRevision:revision)
            if accepted {_=clearSelfPlayModel(address)}
            statusMessage=accepted ? "Verified frozen model selected.":(frozenBindings.error ?? "Frozen selection could not be saved.")
            return accepted
        } catch {if evaluationHostIsCurrent {statusMessage=error.localizedDescription};return false}
    }
    @discardableResult func clearFrozenModel(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {
        guard let workspace,evaluationHostIsCurrent,!workspace.isBusy,!owner.isBusy else {return false}
        let accepted=frozenBindings.setReference(nil,for:address,lease:lease,expectedRevision:frozenBindings.revision)
        statusMessage=accepted ? "Frozen model selection cleared.":(frozenBindings.error ?? "Frozen selection could not be cleared.")
        return accepted
    }
    @discardableResult func evaluate(address:WorkflowNodeAddress,continuingOperation:Bool=false)->Bool {
        guard continuingOperation || !hasActiveOperation else {statusMessage="Wait for the active operation or cancel it.";return false}
        guard let workspace,evaluationHostIsCurrent,!isStagingEvaluation,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,
              let runtime=workspace.runtime else {statusMessage="Use an idle current workflow with a matching runtime to evaluate.";return false}
        if !continuingOperation {clearModelFlow();operationIdentity=UUID()}
        do {
            let source=try WorkflowTrainedModelSource.resolve(in:owner.document,consumer:address)
            if source.isTrain,owner.document.nodes.contains(where:{$0.address==source.address && FineTuningConfiguration.isExternal($0)}) {
                if modelLoader.reference(source.address)==nil{return startModelConsumer(address,forceTraining:true)}
                guard workspace.allowExecution()else{statusMessage=workspace.statusMessage;return false}
                return try startCausalEvaluation(address,policy:workspace.executionPolicy(for:owner.projectID),continuingOperation:continuingOperation)
            }
            if source.isTrain,frozenBindings.reference(for:source.address)==nil {return startModelConsumer(address,forceTraining:true)}
        } catch {statusMessage=error.localizedDescription;clearModelFlow();return false}
        let policy: ResolvedExecutionPreferences
        do {
            guard workspace.allowExecution() else {statusMessage=workspace.statusMessage;return false}
            policy = try workspace.executionPolicy(for: owner.projectID)
        }
        catch { statusMessage = error.localizedDescription; return false }
        let id=UUID(),stage=evaluationStage,root=workspace.workspaceDirectory
        evaluationStagingID=id;isStagingEvaluation=true;retainedHost=self;retainedWorkspace=workspace
        statusMessage="Capturing frozen evaluation inputs…"
        evaluationStagingTask=Task {
            do {
                let request=try await WorkflowEvaluationRequest.capture(from:owner,bindings:frozenBindings,evaluate:address,runtime:runtime,root:root,accountingRoot:workspace.applicationDirectory,stage:stage,executionBudget:policy.budget,storageLimitBytes:policy.storageLimitBytes)
                var transferred=false
                defer {if !transferred {request.removeSnapshot()}}
                try Task.checkCancellation()
                guard evaluationHostIsCurrent,evaluationStagingID==id,request.isCurrent(in:owner,bindings:frozenBindings) else {throw WorkflowArchiveError("Evaluation capture no longer belongs to this workflow.")}
                let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                    guard let self,let workspace,self.evaluationHostIsCurrent,self.evaluationStagingID==id,self.isStagingEvaluation,request.isCurrent(in:self.owner,bindings:self.frozenBindings),!workspace.isPreparingData,!workspace.isRunning,!workspace.isPreviewing,!workspace.isInspectingData,!workspace.isAssessingLearning else {return nil}
                    workspace.cancelDataPreparation={[weak self] in self?.evaluation?.cancel()}
                    let token=workspace.dataPreparationReservationID
                    workspace.isPreparingData=true
                    self.isStagingEvaluation=false
                    return token
                },release:{[weak workspace] token in
                    guard let workspace,workspace.dataPreparationReservationID==token else {return}
                    workspace.isPreparingData=false;workspace.cancelDataPreparation=nil
                })
                let next=WorkflowEvaluationSession(request:request,owner:owner,bindings:frozenBindings,workspace:root,exposureRoot:workspace.applicationDirectory,reservation:reservation,launch:evaluationLaunch,onTerminal:{[weak self,flowID=modelFlowID] result in
                    request.removeSnapshot()
                    guard let self else {return}
                    defer {self.retainedHost=nil;self.retainedWorkspace=nil}
                    self.evaluationOutcomes.insert(result,at:0)
                    if self.evaluationOutcomes.count>100 {self.evaluationOutcomes.removeLast()}
                    if self.evaluationHostIsCurrent {
                        self.statusMessage=result.recordError ?? result.exposureError ?? result.outcome.error ?? "Evaluation finished. Open its saved result to inspect it."
                        self.evaluationHistory.refresh()
                        self.finishModelConsumer(result.outcome.status == .completed && result.recordError == nil,continuingOperation:continuingOperation,flowID:flowID)
                    }
                })
                evaluation=next
                evaluationObservations=[next.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()},next.$statusMessage.sink {[weak self] value in if let self,self.evaluationHostIsCurrent {self.statusMessage=value}}]
                transferred=next.start()
                evaluationStagingID=nil;evaluationStagingTask=nil;isStagingEvaluation=false
                if !transferred {retainedHost=nil;retainedWorkspace=nil;statusMessage=next.statusMessage;clearModelFlow()}
            } catch {
                if evaluationStagingID==id {evaluationStagingID=nil;isStagingEvaluation=false;evaluationStagingTask=nil;retainedHost=nil;retainedWorkspace=nil}
                if evaluationHostIsCurrent {statusMessage=error.localizedDescription};clearModelFlow()
            }
        }
        return true
    }

    @discardableResult func advanceGuidedCourse(expected:WorkflowGuidedBuildProgress)->Bool {
        guard let workspace,workspace.workflowSession === self,!isRetired,!workspace.isBusy,!canvas.isBusy else {return false}
        guard !owner.hasPendingInspectorEdits else {
            canvas.statusMessage="Apply or discard unfinished inspector values before placing the next course component."
            return false
        }
        let accepted=canvas.apply(.advanceGuidedBuild(expected:expected),lease:owner.lease())
        if accepted {canvas.revealSelection();workspace.refreshProjects()}
        return accepted
    }

    /// Rendering may inspect this list without compiling or opening local sources.
    var trainTerminals: [WorkflowNodeAddress] {
        owner.document.nodes.compactMap { node in
            if case .train = node.payload { return node.address }; return nil
        }
    }
    private func selectedTrain(_ explicit: WorkflowNodeAddress?) throws -> WorkflowNodeAddress {
        let terminals = trainTerminals
        if let explicit {
            guard terminals.contains(explicit) else { throw DataFlowContractError("Choose a Train terminal from this workflow.") }
            return explicit
        }
        let selected = owner.selection.nodeAddresses.intersection(Set(terminals))
        if selected.count == 1, let value = selected.first { return value }
        if terminals.count == 1, let value = terminals.first { return value }
        throw DataFlowContractError(terminals.isEmpty ? "Add and connect a Train terminal before running." : "Select one Train terminal; this workflow contains multiple training paths.")
    }
    struct CheckpointDetails:Equatable {
        let device:String
        let completedEpoch:Int
    }
    func checkpointDetails(_ reference:WorkflowCheckpointReference) async throws -> CheckpointDetails {
        guard let workspace,!isRetired,reference.projectID == owner.projectID else {
            throw DataFlowContractError("Checkpoint details require the active project.")
        }
        let root=workspace.workspaceDirectory
        let policy = try workspace.executionPolicy(for: owner.projectID)
        let details=try await WorkflowHistoryVerificationWork.run {
            let snapshot=try WorkflowHistoryReader.stageCheckpoint(reference,root:root,accountingRoot:workspace.applicationDirectory,storageLimitBytes:policy.storageLimitBytes)
            defer {snapshot.remove()}
            let record=try snapshot.parentRecord()
            return CheckpointDetails(device:record.device,completedEpoch:record.evidence.completedEpoch)
        }
        try Task.checkCancellation()
        return details
    }
    @discardableResult func selectCheckpointParent(reference:WorkflowHistoryReference,
        for train:WorkflowNodeAddress,lease:WorkflowEditorLease) async -> Bool {
        do {
            guard let workspace,!isRetired,!workspace.isBusy,!owner.isBusy,
                  workspace.activeProjectID == owner.projectID,owner.isCurrent(lease),
                  reference.projectID == owner.projectID,trainTerminals.contains(train) else {
                throw DataFlowContractError("Select a completed checkpoint for an active Train component in this project while idle.")
            }
            try Task.checkCancellation()
            let revision=checkpointBindings.revision,root=workspace.workspaceDirectory
            let selected=try await WorkflowHistoryVerificationWork.run {
                guard case .verified(let value)=WorkflowHistoryReader.load(reference,root:root),
                      value.outcome.status == .completed,value.modelContent != nil,
                      let pin=value.outcome.modelManifestSHA256 else {
                    throw DataFlowContractError("Choose a completed run with verified model artifacts.")
                }
                return WorkflowCheckpointReference(projectID:reference.projectID,directoryID:reference.directoryID.uuidString,
                    runID:value.outcome.runID,requestIdentity:value.outcome.requestIdentity,modelManifestSHA256:pin)
            }
            try Task.checkCancellation()
            guard !isRetired,!workspace.isBusy,!owner.isBusy,workspace.activeProjectID == owner.projectID,
                  owner.isCurrent(lease),checkpointBindings.setReference(selected,for:train,lease:lease,expectedRevision:revision) else {
                throw DataFlowContractError("Workflow or checkpoint selection changed during verification. Select the parent again.")
            }
            statusMessage="Checkpoint selected. Its artifacts will be verified again before continuation."
            return true
        } catch {statusMessage=error.localizedDescription;return false}
    }
    @discardableResult func clearCheckpointParent(for train:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {
        guard let workspace,!isRetired,!workspace.isBusy,workspace.activeProjectID == owner.projectID else {return false}
        let accepted=checkpointBindings.setReference(nil,for:train,lease:lease,expectedRevision:checkpointBindings.revision)
        statusMessage=accepted ? "Checkpoint selection cleared.":(checkpointBindings.error ?? "Checkpoint selection could not be cleared.")
        return accepted
    }
    @discardableResult func selectPreparationCache(reference: WorkflowHistoryReference,
        for train: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool {
        do {
            guard let workspace, !isRetired, !workspace.isBusy, !owner.isBusy,
                  workspace.activeProjectID == owner.projectID, owner.isCurrent(lease),
                  reference.projectID == owner.projectID, trainTerminals.contains(train) else {
                throw DataFlowContractError("Select a preparation result for an active Train component in this project while idle.")
            }
            let revision = cacheBindings.revision
            let parts = ["workflow-runs", reference.projectID, reference.directoryID.uuidString, "outcome.json"]
            let bytes = try DiagnosticFileIO.read(root: workspace.workspaceDirectory, parts: parts, limit: 256_000)
            let receipt = try WorkflowArchiveCodec.decodeOutcome(bytes)
            guard let pin = receipt.preparationManifestSHA256 else {
                throw DataFlowContractError("This result has no completed preparation cache. Materialize a preparation first.")
            }
            let managed = WorkflowArchiveCacheReference(projectID: reference.projectID,
                directoryID: reference.directoryID.uuidString, preparationManifestSHA256: pin)
            _ = try WorkflowCapturedCache.capture(reference: managed, workspaceDirectory: workspace.workspaceDirectory)
            guard cacheBindings.setReference(managed, for: train, lease: lease, expectedRevision: revision) else {
                throw DataFlowContractError(cacheBindings.error ?? "The cache selection could not be saved.")
            }
            statusMessage = "Prior preparation selected. Sources and requested cache artifacts will be verified again when running."
            return true
        } catch { statusMessage = error.localizedDescription; return false }
    }
    @discardableResult func clearPreparationCache(for train: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool {
        guard let workspace, !isRetired, !workspace.isBusy, workspace.activeProjectID == owner.projectID else { return false }
        let accepted = cacheBindings.setReference(nil, for: train, lease: lease, expectedRevision: cacheBindings.revision)
        statusMessage = accepted ? "Prior preparation selection cleared." : (cacheBindings.error ?? "Cache selection could not be cleared.")
        return accepted
    }
    /// Outcome belongs to this admission, never to an older training session.
    private(set) weak var scopedPreparation:WorkflowDataPreparationSession?
    private(set) var scopedDataOperation:(identity:UUID,target:WorkflowNodeAddress,state:String)?
    @discardableResult func prepareData(at target:WorkflowNodeAddress,explainMissingCapture:Bool=false,authorize:@escaping ()throws->Void = {})->Bool {
        guard !hasActiveOperation else {statusMessage="Wait for the active operation or cancel it.";return false}
        var admittedIdentity:UUID?
        do {
            guard !isRetired,let workspace,workspace.workflowSession === self,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,let runtime=workspace.runtime else {throw DataFlowContractError("Use the idle current canvas with a matching runtime and apply pending edits.")}
            guard workspace.allowExecution() else {statusMessage=workspace.statusMessage;return false}
            try authorize()
            let plan=try scopedDataPlan(at:target,explainMissingCapture:explainMissingCapture)
            let policy=try workspace.executionPolicy(for:owner.projectID)
            let budget=try ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
            let lease=owner.lease(),captured=owner.document,identity=UUID()
            operationIdentity=identity;admittedIdentity=identity;scopedDataOperation=(identity,target,"running")
            func start() throws {
                try authorize()
                guard owner.isCurrent(lease),owner.document==captured,!isRetired,workspace.workflowSession === self else {throw CancellationError()}
                var inputs:[WorkflowNodeAddress:TableViewQuerySession.Artifact]=[:],pins:[WorkflowEndpoint:String]=[:]
                for (source,endpoint) in plan.capturedSources {
                    guard let artifact=currentCapturedTableArtifact(endpoint),
                          plan.document.nodes.contains(where:{$0.address==source && $0.payload == .dataOperator(operatorID:DataFlowSourceOperators.capturedTable,config:["sha256":.string(artifact.sourceSHA256)] )}) else {throw WorkflowCapturedTableRequirement()}
                    inputs[source]=artifact;pins[endpoint]=artifact.sourceSHA256
                }
                let session=try WorkflowDataPreparationSession(workspace:workspace,owner:owner,bindings:bindings,destinations:outputDestinations,history:dataPreparationHistory,runtime:runtime,policy:policy,launcher:executionLaunch,acceptedBudgetWindow:budget,predictionInputs:inputs,allowTrainingTerminals:true,preparationDocument:plan.document)
                scopedPreparation=session
                if inputs.isEmpty {dataPreparation=session} else {predictionPreparation=session;predictionPreparationPins=pins}
                let observations=[session.objectWillChange.sink {[weak self,weak session] _ in
                    guard let self,let session,!self.isRetired,self.scopedPreparation === session else{return};self.objectWillChange.send();self.canvas.objectWillChange.send()
                },session.$statusMessage.sink {[weak self,weak session] message in
                    guard let self,let session,!self.isRetired,self.scopedPreparation === session else{return};self.statusMessage=message
                }]
                if inputs.isEmpty {dataPreparationObservations=observations} else {predictionPreparationObservations=observations}
                session.onTerminal={[weak self,weak session] _ in
                    guard let self,let session,self.operationIdentity==identity else{return}
                    self.scopedDataOperation=(identity,target,session.phase == .completed ? "completed":session.phase == .cancelled ? "cancelled":"failed")
                }
                try session.start();if inputs.isEmpty {latestRunID=session.runID}
            }
            if plan.apiSources.isEmpty {try start()}
            else {
                statusMessage="Fetching selected API sources…"
                apiRunTask=Task {[self] in
                    defer {apiRunTask=nil}
                    do {
                        try await sourceProvider.refreshAPISnapshotsForRun(runID:identity,budget:budget,sourceAddresses:plan.apiSources,acceptedPolicy:policy)
                        try Task.checkCancellation();try start()
                    } catch {
                        scopedDataOperation=(identity,target,Task.isCancelled || error is CancellationError ? "cancelled":"failed")
                        statusMessage="Selected data preparation stopped; check source diagnostics."
                    }
                }
                workspace.cancelDataPreparation={[weak self] in self?.apiRunTask?.cancel()}
            }
            return true
        } catch {if let admittedIdentity,let active=scopedDataOperation,active.identity==admittedIdentity {scopedDataOperation=(active.identity,active.target,"failed")};statusMessage=error.localizedDescription;return false}
    }
    private var apiRunBudget:(operation:UUID,window:ExecutionBudgetWindow)?
    private var currentAPIBudgetWindow:ExecutionBudgetWindow? {apiRunBudget?.operation == operationIdentity ? apiRunBudget?.window:nil}
    private var apiRunTask:Task<Void,Never>?
    var isFetchingAPISources:Bool {apiRunTask != nil}
    @discardableResult func run(train:WorkflowNodeAddress?=nil, useSelection:Bool=true, continuingOperation:Bool=false, apiSnapshotsRefreshed:Bool=false)->Bool {
        guard continuingOperation || apiSnapshotsRefreshed || !hasActiveOperation else {statusMessage="Wait for the active operation or cancel it.";return false}
        // A selected captured-data consumer is an explicit scoped operation, not
        // permission to refresh unrelated API sources or replay other producers.
        if train==nil,useSelection,!continuingOperation,!apiSnapshotsRefreshed,
           let selected=owner.selection.nodeAddresses.first,
           WorkflowPredictionPreparation.dependentData(in:owner.document).contains(selected) {
            return prepareData(at:selected,explainMissingCapture:true)
        }
        do {
            guard !modelLoader.isBusy else {throw DataFlowContractError("Wait for the current model operation or cancel it.")}
            guard let workspace, runStagingID == nil else { throw DataFlowContractError("Run capture is already in progress.") }
            guard workspace.allowExecution() else {statusMessage=workspace.statusMessage;return false}
            if !continuingOperation && !apiSnapshotsRefreshed {operationIdentity=UUID();apiRunBudget=nil}
            if !apiSnapshotsRefreshed,!continuingOperation,owner.document.nodes.contains(where:{if case .dataOperator(let id,let c)=$0.payload {return id==DataFlowSourceOperators.tableInput && c["format"] == .string("api")};return false}) {
                guard apiRunTask==nil,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits else {throw WorkflowArchiveError("Wait for the active operation and apply pending edits.")}
                let lease=owner.lease()
                let policy=try workspace.executionPolicy(for:owner.projectID)
                let apiWindow=try ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
                apiRunBudget=(operationIdentity,apiWindow)
                statusMessage="Fetching API sources…"
                apiRunTask=Task { [self] in
                    defer {apiRunTask=nil}
                    do {try await sourceProvider.refreshAPISnapshotsForRun(runID:operationIdentity,budget:apiWindow);try Task.checkCancellation();guard owner.isCurrent(lease),!isRetired else {throw CancellationError()};apiRunTask=nil;_ = run(train:train,useSelection:useSelection,continuingOperation:continuingOperation,apiSnapshotsRefreshed:true)}
                    catch {statusMessage=Task.isCancelled ? "API capture cancelled.":"API capture failed; no output request was sent. Check Table Input’s Diagnostics and saved settings."}
                }
                workspace.cancelDataPreparation={ [weak self] in self?.apiRunTask?.cancel() }
                return true
            }
            let policy = try workspace.executionPolicy(for: owner.projectID)
            let selected = train ?? (useSelection ? owner.selection.nodeAddresses.first:nil)
            let learners=owner.document.nodes.filter(SelfPlayConfiguration.isSelfPlay)
            if let learner=learners.first(where:{$0.address==selected}) ?? (selected==nil && learners.count==1 ? learners.first:nil) {return startSelfPlay(learner.address) != nil}
            if let selected,owner.document.nodes.contains(where:{node in guard node.address==selected else{return false};if case .gameSession=node.payload{return true};return false}) {return startGameSession(selected,continuingOperation:true) != nil}
            if selected==nil {
                let games=owner.document.nodes.filter{if case .gameSession=$0.payload{return true};return false}
                if games.count==1 {return startGameSession(games[0].address,continuingOperation:true) != nil}
            }
            if train == nil,useSelection,let selected,let node=owner.document.nodes.first(where:{$0.address==selected}) {
                switch node.payload {case .evaluateTest,.saveModel,.generateText:return startModelConsumer(selected,forceTraining:true);default:break}
            }
            if train==nil,useSelection,let selected,let decision=WorkflowDecisionPreparation.consumer(in:owner.document,for:selected) {
                return startModelConsumer(decision,forceTraining:false)
            }
            if train==nil,useSelection,let selected,WorkflowPredictionPreparation.dependentData(in:owner.document).contains(selected) {
                return try startPredictionPreparation(policy:policy)
            }
            let estimators = owner.document.nodes.filter { if case .estimator = $0.payload { return true }; return false }
            if let address = selected.flatMap({ selected in estimators.first(where: { $0.address == selected })?.address }) ?? (estimators.count == 1 && !owner.document.nodes.contains(where: { if case .train = $0.payload { return true }; return false }) ? estimators.first?.address : nil) {
                guard let runtime = workspace.runtime else { throw WorkflowArchiveError("A matching runtime is required.") }
                let plan = try WorkflowEstimatorCompiler.compile(owner.document, estimator: address, modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry)
                if let readiness = workspace.estimatorRuntimeReadinessMessage(backend: plan.configuration.backendID) {
                    throw WorkflowArchiveError(readiness)
                }
                let session = BackendFitSession(workspace: workspace, owner: owner, bindings: bindings, history: backendHistory, plan: plan, runtime: runtime, policy: policy, launcher: executionLaunch,acceptedBudgetWindow:currentAPIBudgetWindow,documentModel:plan.documentModelSource.flatMap{modelLoader.reference($0)})
                // Explicit producer requests finish at their captured tables. Only
                // the native whole-flow owner may schedule downstream data writes.
                observeBackendFit(session,prepareDescendants:train == nil || continuingOperation)
                try session.start(); statusMessage = "Backend fit started."; return true
            }
            if trainTerminals.isEmpty,estimators.isEmpty {
                guard !isRetired,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,let runtime=workspace.runtime else {throw DataFlowContractError("Use the idle current canvas with a matching runtime and apply or discard pending edits.")}
                let session=try WorkflowDataPreparationSession(workspace:workspace,owner:owner,bindings:bindings,destinations:outputDestinations,history:dataPreparationHistory,runtime:runtime,policy:policy,launcher:executionLaunch,acceptedBudgetWindow:currentAPIBudgetWindow)
                dataPreparation=session
                dataPreparationObservations=[session.objectWillChange.sink {[weak self,weak session] _ in
                    guard let self,let session,!self.isRetired,self.dataPreparation === session else {return};self.objectWillChange.send();self.canvas.objectWillChange.send()
                },session.$statusMessage.sink {[weak self,weak session] message in
                    guard let self,let session,!self.isRetired,self.dataPreparation === session else {return};self.statusMessage=message
                }]
                try session.start();latestRunID=session.runID;return true
            }
            if let terminal=try? selectedTrain(train),let node=owner.document.nodes.first(where:{$0.address==terminal}),FineTuningConfiguration.isExternal(node) {return try startImportedTraining(terminal,policy:policy)}
            return try runCaptured(train: train, policy: policy)
        } catch {statusMessage=error.localizedDescription;clearModelFlow();return false}
    }
    func chooseFineTuningResumePackage(_ address:WorkflowNodeAddress) {
        guard let workspace else{return};do{modelLoader.chooseResumePackage(address:address,policy:try workspace.executionPolicy(for:owner.projectID))}catch{statusMessage=error.localizedDescription}
    }
    private var fineTuningInspectionAddress:WorkflowNodeAddress?
    func isInspectingFineTuning(_ address:WorkflowNodeAddress)->Bool{fineTuningInspectionAddress==address}
    func cancelFineTuningInspection(_ address:WorkflowNodeAddress){guard fineTuningInspectionAddress==address else{return};fineTuningInspectionAddress=nil;cancel()}
    @discardableResult func inspectFineTuning(_ address:WorkflowNodeAddress,draft:[String:JSONValue])->Bool {
        guard let workspace,!hasActiveOperation,workspace.allowExecution() else{return false}
        fineTuningInspectionAddress=address
        do{let started=try startImportedTraining(address,policy:workspace.executionPolicy(for:owner.projectID),inspection:draft);if !started{fineTuningInspectionAddress=nil};return started}catch{fineTuningInspectionAddress=nil;statusMessage=error.localizedDescription;return false}
    }

    private func startImportedTraining(_ address:WorkflowNodeAddress,policy:ResolvedExecutionPreferences,inspection:[String:JSONValue]?=nil)throws->Bool {
        guard let workspace,evaluationHostIsCurrent,!isStagingEvaluation,!workspace.isBusy,!owner.isBusy,!owner.hasPendingInspectorEdits,let runtime=workspace.runtime else {throw WorkflowArchiveError("Use the idle current canvas and apply pending edits.")}
        let window=try currentAPIBudgetWindow ?? ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
        let taskPlan=try WorkflowFineTuningPlan.compile(owner.document,train:address,dataRegistry:owner.dataRegistry)
        let links=owner.document.edges.filter{$0.to.address==address},lease=owner.lease()
        guard let training=links.first(where:{$0.to.port=="training"}),let validation=links.first(where:{$0.to.port=="validation"}),training.kind == .table,validation.kind == .table else {throw WorkflowArchiveError("Connect Training and Validation tables from disjoint partitions.")}
        if inspection == nil {try prepareModelConsumers(of:address)}
        func launch(_ prepared:WorkflowDataPreparationSession)throws {
            guard let train=prepared.generationPromptArtifact(for:training.from),let valid=prepared.generationPromptArtifact(for:validation.from) else {throw WorkflowArchiveError("Preparation did not retain complete Training and Validation tables. Enable required table capture and rerun.")}
            let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                guard let self,let workspace,self.owner.isCurrent(lease),!workspace.isBusy,!self.owner.isBusy else {return nil};workspace.isPreparingData=true;self.owner.setExecutionBusy(true);workspace.cancelDataPreparation={self.modelLoader.cancel()};return workspace.dataPreparationReservationID
            },release:{[weak self,weak workspace] token in guard let workspace,workspace.dataPreparationReservationID==token else {return};workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)})
            if let inspection {
                guard fineTuningInspectionAddress==address else{throw CancellationError()}
                modelLoader.inspectFineTuning(address:address,draft:inspection,training:train,validation:valid,runtime:runtime,policy:policy,reservation:reservation,onTerminal:{[weak self] _ in self?.fineTuningInspectionAddress=nil;self?.clearModelFlow()})
                return
            }
            modelLoader.train(address:address,training:train,validation:valid,runtime:runtime,policy:policy,reservation:reservation,budgetWindow:window,onTerminal:{[weak self,flowID=modelFlowID] success in
                self?.finishImportedModelOperation(success,continuingOperation:true,flowID:flowID)
            })
        }
        if let prepared=currentDataPreparation,prepared.generationPromptArtifact(for:training.from) != nil,prepared.generationPromptArtifact(for:validation.from) != nil {try launch(prepared);return true}
        let session=try WorkflowDataPreparationSession(workspace:workspace,owner:owner,bindings:bindings,destinations:outputDestinations,history:dataPreparationHistory,runtime:runtime,policy:policy,launcher:executionLaunch,acceptedBudgetWindow:currentAPIBudgetWindow,allowTrainingTerminals:true,requiredOutputNodes:Set([training.from.address.localID,validation.from.address.localID]),preparationDocument:taskPlan.preparationDocument(owner.document))
        dataPreparation=session
        dataPreparationObservations=[session.$statusMessage.sink{[weak self] in self?.statusMessage=$0}]
        session.onTerminal={ [weak self,weak session] success in
            guard let self,let session else{return}
            guard success else{self.fineTuningInspectionAddress=nil;self.clearModelFlow();return}
            Task { @MainActor in
                guard self.evaluationHostIsCurrent,self.owner.isCurrent(lease) else {return}
                do {try launch(session)}catch {self.fineTuningInspectionAddress=nil;self.statusMessage=error.localizedDescription;self.clearModelFlow()}
            }
        }
        try session.start();latestRunID=session.runID;return true
    }
    private func startCausalEvaluation(_ address:WorkflowNodeAddress,policy:ResolvedExecutionPreferences,continuingOperation:Bool)throws->Bool {
        guard let workspace,let runtime=workspace.runtime else{throw WorkflowArchiveError("A matching runtime is required.")}
        let plan=try WorkflowEvaluationCompiler.compile(owner.document,evaluate:address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        guard let reference=modelLoader.reference(plan.modelSource.address),let record=modelLoader.trainingRecords.first(where:{$0.id==reference.directoryID}),let input=try? WorkflowToolRequest.decode(DiagnosticFileIO.read(root:record.directory(workspace.workspaceDirectory),parts:["request.json"],limit:256_000)),let trainFile=input.inputFiles.first(where:{$0.path=="input/training.table.json"}) else{throw WorkflowArchiveError("Retain the original fine-tuning Training population before evaluating this model.")}
        // The archived table envelope has the same separately pinned dependencies
        // as live preparation. Keep them when reconstructing its captured input.
        let trainingSegments=input.inputFiles.filter {ModelAssetToolContract.isSegment($0.path)}.map {WorkflowArchiveFile(path:String($0.path.dropFirst(6)),bytes:$0.bytes,sha256:$0.sha256)}
        let training=TableViewQuerySession.Artifact(indexURL:record.directory(workspace.workspaceDirectory).appendingPathComponent(trainFile.path),indexSHA256:trainFile.sha256,sourceSHA256:trainFile.sha256,inputKind:"table",retainedSegments:trainingSegments,workloadPolicy:input.executionBudget.resources?.workloadPolicy),endpoint=WorkflowEndpoint(address:.init(domain:.data,localID:plan.testEndpoint.node),port:plan.testEndpoint.port),populationEndpoint=WorkflowEndpoint(address:.init(domain:.data,localID:plan.canonicalTestEndpoint.node),port:plan.canonicalTestEndpoint.port),lease=owner.lease()
        func launch(_ prepared:WorkflowDataPreparationSession)throws {
            guard let table=prepared.generationPromptArtifact(for:endpoint),let population=prepared.generationPromptArtifact(for:populationEndpoint)else{throw WorkflowArchiveError("Prepare the complete declared evaluation population.")}
            let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
                guard let self,let workspace,self.owner.isCurrent(lease),!workspace.isBusy,!self.owner.isBusy else{return nil};workspace.isPreparingData=true;self.owner.setExecutionBusy(true);workspace.cancelDataPreparation={self.modelLoader.cancel()};return workspace.dataPreparationReservationID
            },release:{[weak self,weak workspace] token in guard let workspace,workspace.dataPreparationReservationID==token else{return};workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)})
            modelLoader.evaluateCausal(address:address,training:training,evaluation:table,population:population,runtime:runtime,policy:policy,exposureRoot:workspace.applicationDirectory,reservation:reservation,onTerminal:{[weak self,flowID=modelFlowID] in self?.finishImportedModelOperation($0,continuingOperation:continuingOperation,flowID:flowID)})
        }
        if let prepared=currentDataPreparation,prepared.generationPromptArtifact(for:endpoint) != nil {try launch(prepared);return true}
        var document=owner.document,needed:Set<WorkflowNodeAddress>=[],pending=[endpoint.address]
        while let node=pending.popLast(){guard needed.insert(node).inserted else{continue};guard node.domain == .data else{throw WorkflowArchiveError("Prepare evaluation from ordinary data sources.")};pending += document.edges.filter{$0.to.address==node}.map(\.from.address)}
        document.nodes.removeAll{!needed.contains($0.address)};document.edges.removeAll{!needed.contains($0.to.address) || !needed.contains($0.from.address)};document.definitions=[];document.drafts=[];document.modelInputs=nil;document.namedModelInputs=nil;document.modelSourceVersions=nil;document.guidedBuild=nil
        let session=try WorkflowDataPreparationSession(workspace:workspace,owner:owner,bindings:bindings,destinations:outputDestinations,history:dataPreparationHistory,runtime:runtime,policy:policy,launcher:executionLaunch,allowTrainingTerminals:true,requiredOutputNodes:[endpoint.address.localID,populationEndpoint.address.localID],preparationDocument:document)
        dataPreparation=session;dataPreparationObservations=[session.$statusMessage.sink{[weak self] in self?.statusMessage=$0}]
        // Preparation releases its reservation before the asynchronous model launch.
        // Keep the accepted operation visible through that handoff, including cancellation.
        let stagingID=UUID()
        evaluationStagingID=stagingID
        session.onTerminal={ [weak self,weak session] success in
            guard let self,self.evaluationStagingID==stagingID else{return}
            guard let session,success else {
                self.evaluationStagingID=nil;self.isStagingEvaluation=false;return
            }
            self.evaluationStagingTask=Task { @MainActor in
                defer {
                    if self.evaluationStagingID==stagingID {
                        self.evaluationStagingID=nil;self.isStagingEvaluation=false;self.evaluationStagingTask=nil
                    }
                }
                guard !Task.isCancelled,self.evaluationHostIsCurrent,self.owner.isCurrent(lease),
                      self.evaluationStagingID==stagingID else{return}
                do {try launch(session)}
                catch {self.statusMessage=error.localizedDescription;self.clearModelFlow()}
            }
        }
        do {try session.start();isStagingEvaluation=true}
        catch {evaluationStagingID=nil;isStagingEvaluation=false;throw error}
        latestRunID=session.runID;return true
    }
    /// Forward the underlying backend publisher into the canvas. Canvas changes
    /// already forward to this host; observing the host itself would form a cycle.
    func observeBackendFit(_ session: BackendFitSession,prepareDescendants:Bool=true) {
        backendFit = session;latestRunID=session.runID
        backendStatusObservation=session.$status.sink { [weak self,weak session] status in
            guard let self,let session,self.backendFit === session,self.evaluationHostIsCurrent else{return}
            self.statusMessage=status
        }
        session.onCompleted = { [weak self,weak session] in
            guard let self,let session,self.backendFit === session,self.evaluationHostIsCurrent else {return}
            if session.plan.configuration.hasTableOutputs && !session.plan.configuration.isPrediction {self.completedDocumentFits[session.plan.estimator]=session}
            if prepareDescendants,session.plan.configuration.hasTableOutputs {self.schedulePredictionPreparation()}
        }
        backendObservation = WorkflowExecutionObservation.connect(
            changes: session.objectWillChange.eraseToAnyPublisher(), phases: session.$phase.eraseToAnyPublisher(),
            isCurrent: { [weak self, weak session] in
                guard let self, let session else { return false }
                return self.evaluationHostIsCurrent && self.backendFit === session
            }, publish: { [weak self] in self?.canvas.objectWillChange.send() })
    }
    /// Check the selected compiled model, not unrelated primitives elsewhere on the canvas.
    /// Queue dispatch repeats admission because runtime discovery can change while waiting.
    private func requireReviewedTrainingRuntime(for plan:WorkflowExecutionPlan,workspace:WorkspaceStore)throws {
        if plan.datasetRecipe != nil, !workspace.dataLoadingRuntimeCapabilities.contains("dataset_source") {throw DataFlowContractError("Dataset is unavailable in this runtime. Refresh runtime support before running.")}
        if plan.loaderRecipe != nil, !workspace.dataLoadingRuntimeCapabilities.contains("data_loader") {throw DataFlowContractError("DataLoader is unavailable in this runtime. Refresh runtime support before running.")}
        if plan.trainingMode != .fresh, !workspace.lifecycleRuntimeCapability.isAvailable {
            throw DataFlowContractError(Self.checkpointRuntimeMessage)
        }
        if let recipe = plan.initializationRecipe,
           !workspace.initializationRuntimeCapabilities.contains(recipe.configuration.componentID) {
            throw DataFlowContractError("This initialization recipe is unavailable in the current runtime. Check runtime support before running.")
        }
        if let recipe = plan.gradientPolicyRecipe,
           !workspace.gradientControlRuntimeCapabilities.contains(recipe.configuration.componentID) {
            throw DataFlowContractError("This gradient policy is unavailable in the current runtime. Refresh runtime support before running.")
        }
        if let recipe = plan.trainabilityRecipe,
           !workspace.gradientControlRuntimeCapabilities.contains(recipe.configuration.componentID) {
            throw DataFlowContractError("This trainability policy is unavailable in the current runtime. Refresh runtime support before running.")
        }
        if let recipe=plan.lossRecipe,!workspace.lossRuntimeCapabilities.contains(recipe.configuration.componentID) {throw DataFlowContractError("This objective recipe is unavailable in the current runtime. Check runtime support before running.")}
        if let recipe=plan.schedulerRecipe,!workspace.schedulerRuntimeCapabilities.contains(recipe.configuration.componentID) {throw DataFlowContractError("This scheduler recipe is unavailable in the current runtime. Check runtime support before running.")}
        if let recipe=plan.optimizerRecipe,!workspace.optimizerRuntimeCapabilities.contains(recipe.configuration.componentID) {
            throw DataFlowContractError("This optimizer recipe is unavailable in the current runtime. Check runtime support before running.")
        }
    }
    private func requireReviewedActivationRuntime(for graph: GraphDocument, workspace: WorkspaceStore, context: GraphValidationContext = .preparedWorkflow) throws {
        guard graph.nodes.contains(where: {
            ["sigmoid", "gelu", "softmax", "layer_norm", "bilinear", "lazy_linear"].contains($0.type)
                || EmbeddingBagRuntimeCapabilities.operations.contains($0.type)
                || ConfigurableActivationRuntimeCapabilities.operations.contains($0.type)
                || RemainingActivationRuntimeCapabilities.operations.contains($0.type)
                || TensorRoutingRuntimeCapabilities.operations.contains($0.type)
                || StatefulNormalizationRuntimeCapabilities.operations.contains($0.type)
                || RecurrentConfiguration.componentIDs.contains($0.type)
                || AttentionConfiguration.componentIDs.contains($0.type)
                || SpatialPatchRuntimeCapabilities.operations.contains($0.type)
                || TensorLayoutRuntimeCapability.operations.contains($0.type)
                || BundledTensorCompositionComponents.entries[$0.type] != nil
        }) else { return }
        if let issue = workspace.capabilityBoundaryError(for: graph, context: context) {
            throw DataFlowContractError(issue)
        }
    }
    @discardableResult func runExperiments(train:WorkflowNodeAddress?=nil)->Bool {
        do {
            guard let workspace,!isRetired,workspace.workflowSession === self,
                  workspace.activeProjectID == owner.projectID,!workspace.isBusy,!owner.isBusy,
                  experimentRunner?.isRunning != true else {throw DataFlowContractError("Use the active idle workflow to start experiments.")}
            guard workspace.allowExecution() else {statusMessage=workspace.statusMessage;return false}
            guard let runtime=workspace.runtime else {throw DataFlowContractError("A matching runtime is required.")}
            let policy = try workspace.executionPolicy(for: owner.projectID)
            let terminal=try selectedTrain(train)
            let compiled=try WorkflowCompiler.compile(owner.document,train:terminal,modelRegistry:owner.modelRegistry,
                dataRegistry:owner.dataRegistry,contractVersion:max(2,WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:owner.document.schemaVersion)))
            try requireReviewedTrainingRuntime(for:compiled,workspace:workspace)
            try requireReviewedActivationRuntime(for: compiled.model.graph, workspace: workspace, context: compiled.lossRecipe == nil ? .preparedWorkflow : .explicitObjective)
            let sourceRevision=bindings.revision,cacheRevision=cacheBindings.revision,checkpointRevision=checkpointBindings.revision
            var priorCache:WorkflowCapturedCache?
            if compiled.preparationCachePolicy == .reuse {
                guard let reference=cacheBindings.reference(for:terminal) else {throw DataFlowContractError("Select a preparation cache before running this group.")}
                priorCache=try WorkflowCapturedCache.capture(reference:reference,workspaceDirectory:workspace.workspaceDirectory)
            }
            let base=try WorkflowExecutionRequest.capture(from:owner,train:terminal,runtime:runtime,priorCache:priorCache,
                supportedGradientDevices:workspace.gradientCapture.supportedDevices,executionBudget:policy.budget,sampleCapture:policy.sampleCapture, preparationCacheLimitBytes: policy.preparationCacheLimitBytes, derivativeCapture: policy.derivativeCapture, moduleCompositionEvidence: workspace.moduleCompositionDiscoveryEvidence)
            let runAuthority=try WorkflowTrainRunSetup.resolve(in:owner.document,train:terminal)
            guard let value=runAuthority.configuration["experiment"] else {
                throw DataFlowContractError("Save experiment combinations in Train first.")
            }
            let spec=try WorkflowExperimentSpecification.decode(value)
            let planned=try WorkflowExperimentPlan.build(base.document,train:terminal,axes:spec.axes,budget:spec.budget,
                modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
            var paths:[WorkflowNodeAddress:String]=[:]
            for address in base.plan.sourceAddresses {
                guard let path=bindings.path(for:address) else {throw DataFlowContractError("Choose a local file for source \(address.localID).")}
                paths[address]=path
            }
            let fileConfigurations=DataFlowSourceOperators.fileConfigurations(in:base.document).filter {paths[$0.key] != nil}
            let groupID = UUID().uuidString
            let root = try canonicalWorkspace(workspace.workspaceDirectory), permissions = workspace.sourceAccess.directory, projectID = owner.projectID
            workspace.cancelDataPreparation = {[weak self] in self?.cancel()}
            let token=workspace.dataPreparationReservationID
            workspace.isPreparingData=true
            retainedHost=self;retainedWorkspace=workspace
            isStagingExperiment=true
            statusMessage="Capturing shared experiment inputs…"
            experimentStagingTask=Task { [self,workspace] in
                var ownershipTransferred=false
                var capturedOwner: WorkflowGroupOwner?, storageAdmission: WorkflowExperimentStorageAdmission?
                defer {
                    if !ownershipTransferred {capturedOwner?.close();storageAdmission?.close()}
                    experimentStagingTask=nil;isStagingExperiment=false
                }
                do {
                    try Task.checkCancellation()
                    let admission = Task.detached(priority: .userInitiated) { () -> (URL, WorkflowGroupOwner, WorkflowExperimentStorageAdmission?) in
                        var storage: WorkflowExperimentStorageAdmission?
                        var owner: WorkflowGroupOwner?
                        var retained = false
                        defer { if !retained { owner?.close(); storage?.close() } }
                        try Task.checkCancellation()
                        if let limit = policy.storageLimitBytes {
                            let access = SourceAccess(directory: permissions)
                            var total: Int64 = 0
                            for (address,path) in paths where fileConfigurations[address]==nil {
                                try Task.checkCancellation()
                                let count = try access.withAccess(to: path) { url in
                                    try WorkflowPreviewCapture.byteCountBound(inputs: [
                                        .file(path: "input/data/" + url.lastPathComponent, source: url, expectedSHA256: nil)
                                    ], purpose: .sourceInspection)
                                }
                                total += Int64(count)
                                guard total <= 256_000_000 else { throw WorkflowArchiveError("Experiment source capture exceeds its byte allowance.") }
                            }
                            storage = try WorkflowExperimentStorageAdmission(workspace: root, accountingRoot: workspace.applicationDirectory, projectID: projectID, groupID: groupID,
                                // The existing copy seam enforces 10MB/file and 256MB aggregate,
                                // not this earlier stat size. Fund growth up to that enforced bound.
                                sourceBytes: fileConfigurations.isEmpty ? min(Int64(paths.count) * 10_000_000, 256_000_000):268_435_456,
                                variantCount: planned.variants.count,hasFileCollections:!fileConfigurations.isEmpty, limitBytes: limit)
                        }
                        try Task.checkCancellation()
                        let created = try WorkflowGroupOwnership.createGroup(workspace: root, projectID: projectID, groupID: groupID)
                        owner = created.owner
                        try storage?.bind(to: created.owner)
                        try Task.checkCancellation()
                        retained = true
                        return (created.directory, created.owner, storage)
                    }
                    let (directory, groupOwner, storage) = try await withTaskCancellationHandler(operation: { try await admission.value }, onCancel: { admission.cancel() })
                    capturedOwner = groupOwner; storageAdmission = storage
                    try Task.checkCancellation()
                    try groupOwner.check()
                    let inputs=try await captureExperimentInputs(paths,directory)
                    try groupOwner.validateDirectory(inputs.directory)
                    let authority:()->Bool = {[weak self,weak workspace] in
                        guard let self,let workspace else {return false}
                        return !self.isRetired && workspace.workflowSession === self && workspace.activeProjectID == self.owner.projectID
                            && base.isCurrent(in:self.owner) && self.bindings.revision == sourceRevision
                            && self.cacheBindings.revision == cacheRevision && self.checkpointBindings.revision == checkpointRevision
                    }
                    try Task.checkCancellation()
                    guard authority(),workspace.dataPreparationReservationID == token,workspace.isPreparingData,!owner.isBusy else {throw DataFlowContractError("Workflow or private bindings changed during experiment capture.")}
                    let requests=try await base.experimentVariantsInBackground(inputs:inputs,owner:owner,
                        supportedGradientDevices:workspace.gradientCapture.supportedDevices)
                    try Task.checkCancellation()
                    try groupOwner.check()
                    guard authority(),workspace.dataPreparationReservationID == token,workspace.isPreparingData else {
                        throw DataFlowContractError("Workflow or private bindings changed during experiment verification.")
                    }
                    let group=try await WorkflowExperimentGroupStore.createInBackground(requests:requests,directory:inputs.directory,groupOwner:groupOwner)
                    try Task.checkCancellation()
                    try groupOwner.check()
                    guard authority(),workspace.dataPreparationReservationID == token,workspace.isPreparingData else {
                        throw DataFlowContractError("Workflow or private bindings changed while saving the experiment group.")
                    }
                    // One workspace reservation spans the entire group, including gaps between sessions.
                    let reservation=WorkflowExecutionReservation(acquire:{[weak workspace] in
                        guard let workspace,authority(),workspace.dataPreparationReservationID == token,workspace.isPreparingData else {return nil}
                        return token
                    },release:{_ in /* The group terminal callback releases the workspace. */})
                    let runner:WorkflowExperimentRunner
                    do {
                        runner=try WorkflowExperimentRunner(requests:requests,owner:owner,group:group,groupOwner:groupOwner,workspaceDirectory:workspace.workspaceDirectory,
                            reservation:reservation,storageAdmission:storageAdmission,launch:executionLaunch,canContinue:authority,onTerminal:{[weak self,weak workspace] in
                                guard let self else {return}
                                if let runner=self.experimentRunner {
                                    self.outcomes.insert(contentsOf:runner.outcomes.reversed(),at:0)
                                    if self.outcomes.count>100 {self.outcomes.removeLast(self.outcomes.count-100)}
                                    self.statusMessage=runner.error ?? (runner.group.record.variants.allSatisfy {$0.status == .completed}
                                        ? "Experiment group completed; inspect its captured runs in History.":"Experiment group stopped; completed runs were retained.")
                                }
                                if let workspace,workspace.dataPreparationReservationID == token {
                                    workspace.isPreparingData=false;workspace.cancelDataPreparation=nil
                                }
                                if let id = self.queuedRequestID, let runner = self.experimentRunner {
                                    let state: WorkflowExecutionQueue.State = runner.error != nil ? .failed : runner.group.record.variants.allSatisfy { $0.status == .completed } ? .completed : .cancelled
                                    self.executionQueue.finish(id, state: state, message: self.statusMessage)
                                    self.queuedRequestID = nil
                                }
                                self.retainedHost=nil;self.retainedWorkspace=nil
                            })
                    } catch {
                        try? group.cancelPending();throw error
                    }
                    experimentRunner=runner;experimentObservations.removeAll()
                    experimentObservations.append(runner.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()})
                    experimentObservations.append(runner.$activeSession.sink {[weak self] session in
                        guard let self,let session else {return}
                        self.observeExecution(session,variantCount:requests.count)
                    })
                    executionBindingRevision=sourceRevision
                    retainedHost=self;retainedWorkspace=workspace;workspace.isPreparingData=true
                    let capture = try WorkflowQueueCapture(projectID: owner.projectID, title: owner.document.name, kind: .experiments,
                        requestIdentities: requests.map(\.requestIdentity), document: base.authoringDocument,
                        sourceHashes: inputs.sources.values.map(\.sha256).sorted(), settings: requests.map { $0.plan.settings }, budget: base.executionBudget, externalLocationID: workspace.externalLocation?.id)
                    do {
                        queuedRequestID = try executionQueue.enqueue(capture: capture, start: {
                            guard authority() else { throw WorkflowArchiveError("Project or private bindings changed after experiment capture. Review the project and submit a new request.") }
                            for request in requests { try self.requireReviewedTrainingRuntime(for:request.plan,workspace:workspace); try self.requireReviewedActivationRuntime(for: request.plan.model.graph, workspace: workspace, context: request.plan.lossRecipe == nil ? .preparedWorkflow : .explicitObjective) }
                            guard runner.start() else { throw WorkflowArchiveError(runner.error ?? "The captured experiment group could not start.") }
                        }, cancel: { runner.cancel() })
                    } catch { runner.cancel(); throw error }
                    statusMessage="Queued \(requests.count) captured experiment variants; inspect the training queue in History."
                    ownershipTransferred=true
                } catch {
                    if workspace.dataPreparationReservationID == token {
                        workspace.isPreparingData=false;workspace.cancelDataPreparation=nil
                    }
                    retainedHost=nil;retainedWorkspace=nil
                    if !isRetired {statusMessage=error is CancellationError
                        ? "Experiment capture cancelled; no worker started.":error.localizedDescription}
                }
            }
            return true
        } catch {statusMessage=error.localizedDescription;return false}
    }

    private static let checkpointRuntimeMessage = "Checkpoint restore is unavailable in the current runtime. Check runtime status in Settings before resuming or fine-tuning."
    var checkpointRestoreUnavailableReason: String? {
        workspace?.lifecycleRuntimeCapability.isAvailable == true ? nil : Self.checkpointRuntimeMessage
    }

    private func canonicalWorkspace(_ directory: URL) throws -> URL {
        guard let resolved = realpath(directory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(resolved) }
        return URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
    }

    private func runCaptured(train: WorkflowNodeAddress?, policy: ResolvedExecutionPreferences) throws -> Bool {
        guard let workspace, !isRetired, owner.isCurrent(owner.lease()), workspace.activeProjectID == owner.projectID,
              !workspace.isBusy, !owner.isBusy, runStagingID == nil else {
            throw DataFlowContractError("Use the active idle workflow before running.")
        }
        guard let runtime = workspace.runtime else { throw DataFlowContractError(workspace.runtimeError ?? "A matching worker runtime is required.") }
        let acceptedWindow = try currentAPIBudgetWindow ?? ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let terminal = try selectedTrain(train)
        modelArtifactTask?.cancel();modelArtifactID=nil;isManagingTrainedArtifacts=false
        try prepareModelConsumers(of:terminal)
        guard frozenBindings.setReference(nil,for:terminal,lease:owner.lease(),expectedRevision:frozenBindings.revision) else {clearModelFlow();throw DataFlowContractError(frozenBindings.error ?? "Could not clear the previous Train output.")}
        let compiled = try WorkflowCompiler.compile(owner.document, train: terminal,
            modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry,
            contractVersion: max(2, WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: owner.document.schemaVersion)))
        try requireReviewedTrainingRuntime(for:compiled,workspace:workspace)
            try requireReviewedActivationRuntime(for: compiled.model.graph, workspace: workspace, context: compiled.lossRecipe == nil ? .preparedWorkflow : .explicitObjective)
        let parent = compiled.trainingMode == .fresh ? nil : checkpointBindings.reference(for: terminal)
        let checkpointRevision = checkpointBindings.revision, cacheRevision = cacheBindings.revision, bindingRevision = bindings.revision
        var priorCache: WorkflowCapturedCache?
        if compiled.preparationCachePolicy == .reuse {
            guard let reference = cacheBindings.reference(for: terminal) else { throw DataFlowContractError("Select a prior materialized preparation result before reusing cache.") }
            priorCache = try WorkflowCapturedCache.capture(reference: reference, workspaceDirectory: workspace.workspaceDirectory)
        }
        let destinationRevision=outputDestinations.revision
        let databaseNodes=compiled.dataflow.nodes.filter {$0.operatorID=="kwip.data.outputDelimited" && APITableSettings.isExternal($0.config)}
        let databaseStore=outputDestinations.databaseOutputs
        var databaseAuthority:[String:DatabaseOutputStore.Authority]=[:]
        for node in databaseNodes {databaseAuthority[node.id]=try databaseStore.captureAuthority(node)}
        let destinationNodes=Set(compiled.dataflow.nodes.filter {$0.operatorID=="kwip.data.outputDelimited" && DatabaseOutputSettings.requiresLocalFolder($0.config)}.map {WorkflowNodeAddress(domain:.data,localID:$0.id)})
        let destinationAccess=try outputDestinations.acquire(destinationNodes)
        var destinationHandedOff=false
        defer {if !destinationHandedOff {destinationAccess.close()}}
        let request = try WorkflowExecutionRequest.capture(from: owner, train: terminal, runtime: runtime, priorCache: priorCache,
            supportedGradientDevices: workspace.gradientCapture.supportedDevices, checkpointParent: parent, executionBudget: policy.budget, sampleCapture: policy.sampleCapture, preparationCacheLimitBytes: policy.preparationCacheLimitBytes, derivativeCapture: policy.derivativeCapture, moduleCompositionEvidence: workspace.moduleCompositionDiscoveryEvidence, outputDestinationPaths:destinationAccess.paths.filter {id,_ in !databaseNodes.contains(where:{$0.id==id})})
        var paths: [WorkflowNodeAddress: String] = [:]
        for address in request.plan.sourceAddresses {
            guard let path = bindings.path(for: address) else { throw DataFlowContractError("Choose a local CSV for source \(address.localID) before running.") }
            paths[address] = path
        }
        let id = UUID()
        let root = try canonicalWorkspace(workspace.workspaceDirectory), permissions = workspace.sourceAccess.directory
        let sourceCapture = captureRunSources, checkpointCapture = captureCheckpoint
        runStagingID = id; isStagingCheckpoint = parent != nil
        workspace.isPreparingData = true; workspace.cancelDataPreparation = { [weak self] in self?.cancel() }
        let token = workspace.dataPreparationReservationID
        retainedHost = self; retainedWorkspace = workspace
        statusMessage = parent == nil ? "Capturing workflow sources…" : "Verifying selected checkpoint…"
        destinationHandedOff=true
        runStagingTask = Task { [self, workspace] in
            var snapshot: WorkflowCheckpointSnapshot?, sources: WorkflowRunSourceCapture?, transferred = false
            defer {
                if runStagingID == id { runStagingID = nil; runStagingTask = nil; isStagingCheckpoint = false; sourceCaptureProgress?.clear();sourceCaptureIdentity=nil;canvas.objectWillChange.send() }
                if !transferred {
                    sources?.remove(); snapshot?.remove(); destinationAccess.close()
                    if workspace.dataPreparationReservationID == token { workspace.isPreparingData = false; workspace.cancelDataPreparation = nil }
                    retainedHost = nil; retainedWorkspace = nil
                }
            }
            @MainActor func current() -> Bool {
                !isRetired && workspace.activeProjectID == owner.projectID && request.isCurrent(in: owner)
                    && bindings.revision == bindingRevision && cacheBindings.revision == cacheRevision
                    && checkpointBindings.revision == checkpointRevision && outputDestinations.revision == destinationRevision && workspace.dataPreparationReservationID == token
                    && workspace.isPreparingData && runStagingID == id
            }
            do {
                try Task.checkCancellation()
                _ = try acceptedWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                if let parent { snapshot = try await checkpointCapture(parent, root, policy.storageLimitBytes) }
                try Task.checkCancellation()
                _ = try acceptedWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                guard current() else { throw DataFlowContractError("Workflow or private inputs changed during checkpoint capture.") }
                isStagingCheckpoint = false
                statusMessage = "Capturing workflow sources…"
                _ = try acceptedWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                sources = try await sourceCapture(paths, root, permissions, policy.storageLimitBytes)
                try Task.checkCancellation()
                _ = try acceptedWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                guard current(), let sourceSnapshot = sources else { throw DataFlowContractError("Workflow or private inputs changed during source capture.") }
                let checkpointSnapshot = snapshot
                let reservation = WorkflowExecutionReservation(acquire: { [weak self, weak workspace] in
                    guard let self, let workspace, !self.isRetired, workspace.activeProjectID == self.owner.projectID,
                          request.isCurrent(in: self.owner), self.bindings.revision == bindingRevision,
                          self.cacheBindings.revision == cacheRevision, self.checkpointBindings.revision == checkpointRevision, self.outputDestinations.revision == destinationRevision,
                          workspace.dataPreparationReservationID == token, workspace.isPreparingData, !self.owner.isBusy else { return nil }
                    return token
                }, release: { [weak workspace] activeToken in
                    guard let workspace, workspace.dataPreparationReservationID == activeToken else { return }
                    workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
                })
                let directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(UUID().uuidString)")
                let remaining = try acceptedWindow.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                let captureElapsed = Double(policy.budget.maximumRunSeconds) - remaining
                let capture = try WorkflowQueueCapture(projectID: owner.projectID, title: owner.document.name, kind: .training,
                    requestIdentities: [request.requestIdentity], document: request.authoringDocument,
                    sourceHashes: sourceSnapshot.sources.values.map(\.sha256).sorted(), settings: [request.plan.settings], budget: request.executionBudget, externalLocationID: workspace.externalLocation?.id)
                queuedRequestID = try executionQueue.enqueue(capture: capture, start: { [self, workspace] in
                    guard !isRetired, workspace.activeProjectID == owner.projectID,
                          request.isCurrent(in: owner), bindings.revision == bindingRevision,
                          cacheBindings.revision == cacheRevision, checkpointBindings.revision == checkpointRevision, outputDestinations.revision == destinationRevision,
                          workspace.dataPreparationReservationID == token, workspace.isPreparingData else {
                        throw WorkflowArchiveError("Project or private bindings changed after capture. Review the project and submit a new request; queued work never adopts later edits.")
                    }
                    try requireReviewedTrainingRuntime(for:request.plan,workspace:workspace)
                    try requireReviewedActivationRuntime(for: request.plan.model.graph, workspace: workspace, context: request.plan.lossRecipe == nil ? .preparedWorkflow : .explicitObjective)
                    // Capturing consumes the accepted allowance; waiting for the slot does not.
                    let dispatchWindow = try ExecutionBudgetWindow(policy: policy.budget,
                        startedAtUptime: ProcessInfo.processInfo.systemUptime - captureElapsed)
                    let next = WorkflowExecutionSession(request: request, owner: owner, sources: sourceSnapshot.sources,
                        directory: directory, reservation: reservation, outputDestinationAccess:destinationAccess, checkpointSnapshot: checkpointSnapshot,
                        storageLimitBytes: policy.storageLimitBytes, accountingRoot: workspace.applicationDirectory, acceptedBudgetWindow: dispatchWindow,
                        launch: executionLaunch, onTerminal: { [weak self] outcome in
                            sourceSnapshot.remove(); checkpointSnapshot?.remove(); destinationAccess.close()
                            guard let self else { return }
                            self.completed(outcome)
                            if let id = self.queuedRequestID {
                                let state: WorkflowExecutionQueue.State = outcome.status == .completed ? .completed : outcome.status == .cancelled ? .cancelled : .failed
                                self.executionQueue.finish(id, state: state, message: self.statusMessage)
                                self.queuedRequestID = nil
                            }
                        })
                    executionBindingRevision = bindingRevision
                    next.databaseWrite={node,artifact,id,window in
                        guard let authority=databaseAuthority[node.id] else {throw WorkflowArchiveError("Database authority was not captured for this run.")}
                        return try await databaseStore.write(node:node,artifact:artifact,preparationID:id,window:window,authority:authority)
                    }
                    observeExecution(next)
                    guard next.start() else { throw WorkflowArchiveError(next.statusMessage) }
                    statusMessage = next.statusMessage
                }, cancel: { [self, workspace] in
                    if let execution, execution.request.requestIdentity == request.requestIdentity,
                       ![WorkflowExecutionPhase.completed, .failed, .cancelled].contains(execution.phase) {
                        execution.cancel()
                    } else {
                        sourceSnapshot.remove(); checkpointSnapshot?.remove(); destinationAccess.close()
                        if workspace.dataPreparationReservationID == token { workspace.isPreparingData = false; workspace.cancelDataPreparation = nil }
                        queuedRequestID = nil; retainedHost = nil; retainedWorkspace = nil
                        clearModelFlow()
                        statusMessage = "Queued capture released; no waiting worker started."
                    }
                })
                transferred = true
                statusMessage = "Captured request queued; inspect the training queue in History."
            } catch {
                clearModelFlow()
                if !isRetired { statusMessage = error is CancellationError ? "Run capture cancelled; no worker started." : error.localizedDescription }
            }
        }
        return true
    }
    private func observeExecution(_ session:WorkflowExecutionSession,variantCount:Int?=nil) {
        execution=session;latestRunID=session.runID
        let failureBindingRevision=bindings.revision, failureDestinationRevision=outputDestinations.revision
        session.preparationFailureContextIsCurrent = { [weak self, weak session] in
            guard let self, let session, self.currentExecution === session,
                  self.bindings.revision == failureBindingRevision,
                  self.outputDestinations.revision == failureDestinationRevision else { return false }
            return self.sourceProvider.observedSourcesMatch(session.preparationOutputs)
        }
        canvasProgressObservation=session.$canvasProgress.combineLatest(session.$modelOperationProgress).dropFirst().receive(on:RunLoop.main).sink {[weak self,weak session] _ in
            guard let self,let session,!self.isRetired,self.execution === session,session.request.isCurrent(in:self.owner) else {return}
            self.canvas.objectWillChange.send()
        }
        executionObservation = WorkflowExecutionObservation.connect(
            changes: session.objectWillChange.eraseToAnyPublisher(), phases: session.$phase.eraseToAnyPublisher(),
            isCurrent: { [weak self, weak session] in
                guard let self, let session else { return false }
                return !self.isRetired && self.execution === session
            }, publish: { [weak self] in self?.canvas.objectWillChange.send() })
        executionStatusObservation=session.$statusMessage.sink {[weak self,weak session] message in
            guard let self,let session,!self.isRetired,self.execution === session else {return}
            if let count=variantCount,let ordinal=session.request.experiment?.ordinal {
                self.statusMessage="Variant \(ordinal) of \(count) · \(message)"
            } else {self.statusMessage=message}
        }
    }
    private func completed(_ outcome: WorkflowExecutionOutcome) {
        defer { retainedHost = nil; retainedWorkspace = nil }
        let archivedSources = Dictionary(uniqueKeysWithValues: outcome.sources.keys.sorted { $0.localID < $1.localID }.enumerated().map { index, address in
            let captured = outcome.sources[address]!
            return (address, WorkflowCapturedSource(url: outcome.directory.appendingPathComponent("inputs/\(WorkflowSourceFilename.archive(index: index, source: captured.url))"),
                sha256: captured.sha256, byteCount: captured.byteCount))
        })
        let retainedOutcome = WorkflowExecutionOutcome(runID: outcome.runID, request: outcome.request, status: outcome.status,
            failedPhase: outcome.failedPhase, sources: archivedSources, directory: outcome.directory,
            preparationManifestSHA256: outcome.preparationManifestSHA256, modelManifestSHA256: outcome.modelManifestSHA256,
            artifactHashes: outcome.artifactHashes, error: outcome.error, recordError: outcome.recordError)
        outcomes.insert(retainedOutcome, at: 0)
        if outcomes.count > 100 { outcomes.removeLast(outcomes.count - 100) }
        statusMessage = outcome.error ?? (outcome.status == .completed ? "Workflow completed; inspect component outputs or History." : "Workflow cancelled; no completed model result published.")
        if let failure = outcome.recordError { statusMessage += " Local outcome receipt could not be saved: " + failure }
        publishTrainedOutput(outcome)
        // Deliberately no activeProjectID, legacy graph/metrics or current history writes.
    }
    var projectID: String { owner.projectID }
    var sessionID: UUID { owner.sessionID }
    var changes: AnyPublisher<Void, Never> { objectWillChange.eraseToAnyPublisher() }
    /// Cached event evidence only. Matching construction/bindings does not assert
    /// that an external file still has its captured bytes.
    func terminalStatus(for address: WorkflowNodeAddress) -> String? {
        guard let workspace, workspace.workflowSession === self,
              workspace.activeProjectID == owner.projectID, !isRetired,
              owner.isCurrent(owner.lease()), address.domain == .control else { return nil }
        if let session=selfPlay,session.address==address {
            guard WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:address)else{return "Stale"}
            return session.isBusy ? "Running":session.artifacts.isEmpty ? "Failed":"Completed"
        }
        if let session=gameEpisodes,session.address==address {
            guard WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:address)else{return "Stale"}
            return session.isBusy ? "Running":session.artifacts.isEmpty ? "Failed":"Completed"
        }
        if let generation,generation.address==address {
            guard WorkflowOutputConstruction.matches(owner.document,captured:generation.document,provider:address) else {return "Stale"}
            return generation.isBusy ? "Running":generation.artifact != nil ? "Completed":"Failed"
        }
        if let execution=currentExecution,execution.request.plan.train==address {
            guard WorkflowOutputConstruction.matches(owner.document,captured:execution.request.authoringDocument,provider:address) else {return "Stale"}
            switch execution.phase {
            case .completed:return outcomes.first(where:{$0.runID==execution.runID})?.recordError == nil ? "Completed":"Failed"
            case .failed:return "Failed"
            case .cancelled:return "Cancelled"
            case .idle:return nil
            default:return "Running"
            }
        }
        if let evaluation,evaluation.request.plan.evaluate==address {
            guard WorkflowOutputConstruction.matches(owner.document,captured:evaluation.request.document,provider:address) else {return "Stale"}
            switch evaluation.phase {case .completed:return "Completed";case .failed:return "Failed";case .cancelled:return "Cancelled";case .idle:return nil;default:return "Running"}
        }
        guard let backendFit,backendFit.plan.estimator==address else {return nil}
        guard owner.document.matchesCapturedConstruction(backendFit.document), bindings.revision == backendFit.bindingRevision else { return "Stale" }
        switch backendFit.phase {
        case .idle: return nil
        case .preparing, .training: return "Running"
        case .completed: return "Completed"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        }
    }
    private func currentSourceEvidenceMatches(execution:WorkflowExecutionSession)->Bool {
        var outputs:[String:[String:JSONValue]]=[:]
        for address in execution.request.plan.sourceAddresses {
            if let value=execution.typedPreparationOutput(at:address) {outputs[address.localID]=value}
        }
        return sourceProvider.observedSourcesMatch(outputs)
    }
    func imageArtifactResolver(for address: WorkflowNodeAddress) -> FileImageArtifactResolver? {
        guard evaluationHostIsCurrent, address.domain == .data else { return nil }
        if let preparation=currentDataPreparation {return preparation.imageArtifactResolver(for:address)}
        guard snapshot(for:address) != nil else {return nil}
        return capturedTableContext?.tables.images
    }
    func fullTableViewSession(for address: WorkflowNodeAddress) -> TableViewQuerySession? {
        capturedTableViewSession(for:.init(address:address,port:"table"))
    }
    func formulaSampleSession(for consumer:WorkflowNodeAddress,configuration:[String:JSONValue])->TableViewQuerySession? {
        guard evaluationHostIsCurrent,
              let edge=owner.document.edges.first(where:{$0.to.address == consumer && $0.to.port == "table"}) else {return nil}
        if edge.from.port=="predictions",edge.from.address.domain == .control {return predictionTableViewSession(for:edge.from,formulaConsumer:consumer)?.formulaSample(configuration:configuration)}
        if predictionPreparationIsCurrent,let preparation=predictionPreparation,preparation.graph.nodes.contains(where:{$0.id==edge.from.address.localID}) {return preparation.capturedTableViewSession(for:edge.from,formulaConsumer:consumer)?.formulaSample(configuration:configuration)}
        if let preparation=currentDataPreparation {return preparation.capturedTableViewSession(for:edge.from,formulaConsumer:consumer)?.formulaSample(configuration:configuration)}
        return capturedQuery(for:edge.from,formulaConsumer:consumer)?.formulaSample(configuration:configuration)
    }
    /// The caller must already hold a captured-values grant. These fresh queries
    /// retain exact artifact/current-source/resource admission but do not borrow
    /// selection-owned UI query state or change the visible canvas selection.
    func automationTableViewSession(for endpoint:WorkflowEndpoint)->TableViewQuerySession? {
        guard evaluationHostIsCurrent else{return nil}
        if endpoint.address.domain == .control{return predictionTableViewSession(for:endpoint,selectionRequired:false)}
        if predictionPreparationIsCurrent,let preparation=predictionPreparation,preparation.graph.nodes.contains(where:{$0.id==endpoint.address.localID}){return preparation.capturedTableViewSession(for:endpoint,selectionRequired:false)}
        if let preparation=currentDataPreparation{return preparation.capturedTableViewSession(for:endpoint,selectionRequired:false)}
        return capturedQuery(for:endpoint,selectionRequired:false)
    }
    func capturedTableViewSession(for endpoint:WorkflowEndpoint) -> TableViewQuerySession? {
        if endpoint.address.domain == .control,let query=generationTableViewSession(for:endpoint) {return query}
        guard evaluationHostIsCurrent,endpoint.address.domain == .data else {return nil}
        if predictionPreparationIsCurrent,let preparation=predictionPreparation,preparation.graph.nodes.contains(where:{$0.id==endpoint.address.localID}) {return preparation.capturedTableViewSession(for:endpoint)}
        if let preparation=currentDataPreparation {return preparation.capturedTableViewSession(for:endpoint)}
        return capturedQuery(for:endpoint)
    }
    private var capturedTableContext:(runID:UUID,tables:WorkflowCapturedTables)? {
        switch currentResult {
        case .training(let execution):guard execution.phase == .completed,let tables=execution.capturedTables else {return nil};return (execution.runID,tables)
        case .backend(let backend):guard backend.phase == .completed,let tables=backend.capturedTables else {return nil};return (backend.runID,tables)
        default:return nil
        }
    }
    private func capturedQueryIsCurrent(_ endpoint:WorkflowEndpoint,runID:UUID,formulaConsumer:WorkflowNodeAddress?,selectionRequired:Bool=true)->Bool {
        guard evaluationHostIsCurrent,capturedTableContext?.runID==runID,
              let snapshot=snapshot(for:endpoint.address),snapshot.runID==runID,
              snapshot.matchesCurrentConstruction,snapshot.matchesCurrentBindings else {return false}
        if !selectionRequired{return true}
        if let consumer=formulaConsumer {return FormulaSampleScope.permits(consumer:consumer,input:endpoint,document:owner.document,selection:owner.selection)}
        if owner.selection.nodeAddresses == [endpoint.address],owner.selection.edgeIDs.isEmpty {return true}
        guard owner.selection.nodeAddresses.isEmpty,owner.selection.edgeIDs.count==1,
              let id=owner.selection.edgeIDs.first,let edge=owner.document.edges.first(where:{$0.id==id}) else {return false}
        return edge.from==endpoint
    }
    private func capturedQuery(for endpoint:WorkflowEndpoint,formulaConsumer:WorkflowNodeAddress?=nil,selectionRequired:Bool=true)->TableViewQuerySession? {
        guard let workspace,let context=capturedTableContext,
              capturedQueryIsCurrent(endpoint,runID:context.runID,formulaConsumer:formulaConsumer,selectionRequired:selectionRequired),
              let artifact=context.tables.artifacts[endpoint] else {return nil}
        if capturedQueryRunID != context.runID {
            capturedQueries.values.forEach {$0.invalidate()};capturedQueries=[:];capturedQueryRunID=context.runID
        }
        if selectionRequired,formulaConsumer == nil,let existing=capturedQueries[endpoint] {return existing}
        let accepts={ [weak self] in self?.capturedQueryIsCurrent(endpoint,runID:context.runID,formulaConsumer:formulaConsumer,selectionRequired:selectionRequired)==true }
        // Capture only a weak cancellation target; the query owns its reservation.
        final class CancellationTarget {weak var session:TableViewQuerySession?}
        let target=CancellationTarget()
        let reservation=WorkflowExecutionReservation(acquire:{[weak self,weak workspace] in
            guard let self,let workspace,accepts(),!workspace.isBusy,!self.owner.isBusy else {return nil}
            workspace.cancelDataPreparation={target.session?.cancel()}
            let token=workspace.dataPreparationReservationID;workspace.isPreparingData=true;self.owner.setExecutionBusy(true);return token
        },release:{[weak self,weak workspace] token in
            guard let workspace,workspace.dataPreparationReservationID==token else {return}
            workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;self?.owner.setExecutionBusy(false)
        })
        let query=TableViewQuerySession(artifact:artifact,workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,
            projectID:owner.projectID,runtime:context.tables.runtime,storageLimitBytes:context.tables.storageLimitBytes,
            reservation:reservation,imageResolver:context.tables.images,accepts:accepts)
        target.session=query
        if selectionRequired,formulaConsumer == nil {capturedQueries[endpoint]=query}
        return query
    }
    func snapshot(for address: WorkflowNodeAddress) -> WorkflowCapturedDataOutput? {
        snapshots(for:[address])[address]
    }
    func snapshots(for addresses:Set<WorkflowNodeAddress>)->[WorkflowNodeAddress:WorkflowCapturedDataOutput] {
        var result=capturedSnapshots(for:addresses,instancePath:[])
        if predictionPreparationIsCurrent,let preparation=predictionPreparation {result.merge(preparation.snapshots(for:addresses)) {_,new in new}}
        for address in addresses where address.domain == .model {if let value=modelSnapshot(for:address) {result[address]=value}}
        for address in addresses where address.domain == .control {
            if let session=selfPlay,session.address==address,!session.validateOnly {
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:session.id,requestIdentity:session.requestIdentity,capturedRevision:session.document.revision,bindingRevision:0,phase:session.isBusy ? .training:session.artifacts.isEmpty ? .failed:.completed,matchesCurrentConstruction:WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:address),matchesCurrentBindings:true,ports:session.ports,dataCompleteness:"Complete bounded Self-play outputs")
            }
            if let session=gameEpisodes,session.address==address,!session.validateOnly {
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:session.id,requestIdentity:session.requestIdentity,capturedRevision:session.document.revision,bindingRevision:0,phase:session.isBusy ? .preparing:session.artifacts.isEmpty ? .failed:.completed,matchesCurrentConstruction:WorkflowOutputConstruction.matches(owner.document,captured:session.document,provider:address),matchesCurrentBindings:true,ports:session.ports,dataCompleteness:"Complete bounded episode outputs")
            }

            if let record=modelLoader.currentEvaluation,record.address==address,modelLoader.evaluationIsCurrent(record),let file=record.files.first(where:{$0.path=="output/table.json"}),let id=UUID(uuidString:record.id) {
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:id,requestIdentity:file.sha256,capturedRevision:record.document.revision,bindingRevision:modelLoader.bindings.revision,phase:.completed,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:["predictions":.object(["kind":.string("table"),"preview":.object(modelLoader.evaluationTableMetadata ?? [:])])],dataCompleteness:"Complete causal evaluation · original observations exposure tracked",capturedAt:record.finishedAt)
                continue
            }
            if let reference=modelLoader.reference(address),reference.metadata["trainingOperation"] == .bool(true),let record=modelLoader.trainingRecords.first(where:{$0.id==reference.directoryID && $0.address==address}),let id=UUID(uuidString:record.id) {
                var ports:[String:JSONValue]=["model":.object(["kind":.string("trainedModel"),"runID":.string(record.id)])]
                if case .object(let summary)=reference.metadata["predictionSummary"] {ports["predictions"] = .object(["kind":.string("table"),"preview":.object(summary)])}
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:id,requestIdentity:reference.identity,capturedRevision:record.document.revision,bindingRevision:modelLoader.bindings.revision,phase:.completed,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:ports,dataCompleteness:"Verified imported-model training",capturedAt:record.finishedAt)
                continue
            }
            if let record=modelLoader.generationRecord,address==record.address,modelLoader.generationIsCurrent(.init(address:address,port:"table")),let metadata=modelLoader.generationMetadata,let id=UUID(uuidString:record.operationID) {
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:id,requestIdentity:record.reference.identity,capturedRevision:record.document.revision,bindingRevision:modelLoader.bindings.revision,phase:.completed,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:["table":.object(["kind":.string("table"),"preview":.object(metadata)])],dataCompleteness:"Complete imported-model generation",capturedAt:record.finishedAt)
                continue
            }
            if let generation,generation.address==address,let record=generation.record,let metadata=generation.outputMetadata,generationOutputIsCurrent(.init(address:address,port:"table")),let id=UUID(uuidString:record.id) {
                var preview=metadata
                if case .array(let rows)=generation.result?["rows"] {preview["rowCount"] = .integer(Int64(rows.count))}
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:id,requestIdentity:record.id,capturedRevision:record.document.revision,bindingRevision:bindings.revision,phase:.completed,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:["table":.object(["kind":.string("table"),"preview":.object(preview)])],dataCompleteness:"Complete generation",capturedAt:record.finishedAt)
                continue
            }
            let endpoint=WorkflowEndpoint(address:address,port:"predictions")
            guard predictionOutputIsCurrent(endpoint) else {continue}
            let runID:UUID,identity:String,revision:UInt64
            if let execution=currentExecution,execution.request.plan.train==address {runID=execution.runID;identity=execution.request.requestIdentity;revision=execution.request.document.revision}
            else if let evaluation {runID=evaluation.evaluationID;identity=evaluation.request.requestIdentity;revision=evaluation.request.document.revision}
            else {continue}
            var ports:[String:JSONValue]=[:]
            if let evidence=predictionEvidence(for:endpoint){ports["predictions"]=evidence.outputMetadata}
            if let execution=currentExecution,execution.request.plan.train==address{for (port,evidence) in execution.routingTables{ports[port]=evidence.outputMetadata}}
            guard !ports.isEmpty else{continue}
            if let reference=frozenBindings.reference(for:address) {ports["model"] = .object(["kind":.string("trainedModel"),"runID":.string(reference.runID)])}
            result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:runID,requestIdentity:identity,capturedRevision:revision,bindingRevision:bindings.revision,phase:.completed,matchesCurrentConstruction:true,matchesCurrentBindings:true,ports:ports,dataCompleteness:"Captured model outputs")
        }

        return result
    }
    func modelSnapshot(for address: WorkflowNodeAddress, instancePath: [String] = []) -> WorkflowCapturedDataOutput? {
        guard address.domain == .model else {return nil}
        if let member=try? WorkflowInlineContainer.parse(in:owner.document,address:address) {
            let actual=WorkflowNodeAddress(domain:.model,localID:member.childID)
            guard let value=capturedSnapshots(for:[actual],instancePath:instancePath+[member.root.localID])[actual] else {return nil}
            return .init(projectID:value.projectID,sessionID:value.sessionID,address:address,runID:value.runID,requestIdentity:value.requestIdentity,
                capturedRevision:value.capturedRevision,bindingRevision:value.bindingRevision,phase:value.phase,matchesCurrentConstruction:value.matchesCurrentConstruction,
                matchesCurrentBindings:value.matchesCurrentBindings,ports:value.ports,dataCompleteness:value.dataCompleteness,capturedAt:value.capturedAt)
        }
        return capturedSnapshots(for:[address],instancePath:instancePath)[address]
    }
    /// Cached observations only. Compute semantic/source freshness once, then project each qualified address.
    private func capturedSnapshots(for addresses:Set<WorkflowNodeAddress>,instancePath:[String])->[WorkflowNodeAddress:WorkflowCapturedDataOutput] {
        guard !addresses.isEmpty,evaluationHostIsCurrent,let current=currentResult else {return [:]}
        switch current {
        case .preparation(let preparation):
            let captured=preparation.snapshots(for:addresses)
            guard !captured.isEmpty else {return [:]}
            let sourceMatches=sourceProvider.observedSourcesMatch(preparation.preparationOutputs)
            return captured.mapValues { snapshot in
                .init(projectID:snapshot.projectID,sessionID:snapshot.sessionID,address:snapshot.address,runID:snapshot.runID,requestIdentity:snapshot.requestIdentity,capturedRevision:snapshot.capturedRevision,bindingRevision:snapshot.bindingRevision,phase:snapshot.phase,matchesCurrentConstruction:snapshot.matchesCurrentConstruction,matchesCurrentBindings:snapshot.matchesCurrentBindings && sourceMatches,ports:snapshot.ports,dataCompleteness:snapshot.dataCompleteness,capturedAt:snapshot.capturedAt)
            }
        case .backend(let backend):
            guard !isRetired,owner.isCurrent(owner.lease()) else {return [:]}
            let constructionMatches=owner.document.matchesCapturedConstruction(backend.document)
            let bindingsMatch=bindings.revision == backend.bindingRevision && sourceProvider.observedSourcesMatch(backend.preparationOutputs)
            return addresses.reduce(into:[:]) { result,address in
                guard address.domain == .data,let ports=backend.preparationOutputs[address.localID] else {return}
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:backend.runID,requestIdentity:backend.requestIdentity,capturedRevision:backend.document.revision,bindingRevision:backend.bindingRevision,phase:backend.phase,matchesCurrentConstruction:constructionMatches,matchesCurrentBindings:bindingsMatch,ports:ports)
            }
        case .training(let execution):
            guard let workspace,workspace.workflowSession === self,workspace.activeProjectID == owner.projectID,!isRetired,
                  owner.isCurrent(owner.lease()),let revision=executionBindingRevision,
                  execution.request.lease.projectID == owner.projectID,execution.request.lease.sessionID == owner.sessionID else {return [:]}
            let constructionMatches=owner.document.matchesCapturedConstruction(execution.request.document)
            let bindingsMatch=bindings.revision == revision
            // Model captures have their existing binding contract; data also requires observed source headers.
            let dataBindingsMatch=bindingsMatch && (!addresses.contains(where:{$0.domain == .data}) || currentSourceEvidenceMatches(execution:execution))
            return addresses.reduce(into:[:]) { result,address in
                let ports:[String:JSONValue]
                switch address.domain {
                case .data:
                    guard let captured=execution.typedPreparationOutput(at:address) else {return};ports=captured
                case .model:
                    guard execution.phase == .completed,let captured=execution.modelCaptures?.resolvedPorts(instancePath:instancePath,localNodeID:address.localID) else {return};ports=captured
                default:return
                }
                result[address] = .init(projectID:owner.projectID,sessionID:owner.sessionID,address:address,runID:execution.runID,requestIdentity:execution.request.requestIdentity,capturedRevision:execution.request.document.revision,bindingRevision:revision,phase:execution.phase,matchesCurrentConstruction:constructionMatches,matchesCurrentBindings:address.domain == .data ? dataBindingsMatch:bindingsMatch,ports:ports)
            }
        }
    }
    func cancel() {
        fineTuningInspectionAddress=nil
        campaignContinuationTask?.cancel();campaignContinuationTask=nil
        apiRunTask?.cancel()
        if selfPlay?.isBusy==true {selfPlay?.cancel();return}
        if selfPlayValidation?.isBusy==true {selfPlayValidation?.cancel();return}
        if gameEpisodes?.isBusy==true {gameEpisodes?.cancel();return}
        if gameValidation?.isBusy==true {gameValidation?.cancel();return}
        if modelLoader.isBusy {modelLoader.cancel();return}
        clearModelFlow();modelArtifactTask?.cancel();modelSave?.cancel();generation?.cancel()
        if predictionPreparation?.isRunning == true {predictionPreparation?.cancel();return}
        if dataPreparation?.isRunning == true {dataPreparation?.cancel();return}
        if let id = queuedRequestID { executionQueue.cancel(id); return }
        if backendFit?.isRunning == true { backendFit?.cancel(); return }
        if runStagingID != nil {
            sourceCaptureProgress?.clear();canvas.objectWillChange.send()
            runStagingTask?.cancel()
            statusMessage = "Cancelling run capture; waiting for staging to stop…"
            return
        }
        if isStagingEvaluation {
            evaluationStagingTask?.cancel();frozenSelectionTask?.cancel()
            statusMessage="Cancelling frozen evaluation verification…"
        } else if let evaluation,![WorkflowEvaluationPhase.completed,.failed,.cancelled].contains(evaluation.phase) {evaluation.cancel()}
        else if experimentRunner?.isRunning == true {experimentRunner?.cancel()}
        else if isStagingExperiment {
            experimentStagingTask?.cancel()
            statusMessage="Cancelling experiment capture…"
        }
        else {execution?.cancel()}
    }
    func retire() {
        guard !isRetired else { return }
        cancel()
        isRetired = true
        predictionQueries.values.forEach {$0.1.invalidate()};predictionQueries=[:]
        predictionPreparation?.cancel();predictionPreparationObservations.removeAll()
        capturedQueries.values.forEach {$0.invalidate()};capturedQueries=[:]
        executionObservation?.cancel(); canvasProgressObservation?.cancel(); executionStatusObservation?.cancel(); backendObservation?.cancel(); backendStatusObservation?.cancel()
        modelLoader.invalidate();modelSave?.invalidate();modelArtifactTask?.cancel()
        generationRefreshTask?.cancel();generationQuery?.invalidate();generation?.cancel()
        apiRunTask?.cancel();canvas.invalidate(); sourceProvider.invalidate();evaluationHistory.invalidate()
        execution?.cancel()
        owner.invalidate()
        observations.removeAll()
        statusMessage = "Workflow workspace retired. Active execution is cancelling."
    }
}


/// Coalesces presentation invalidations only. Protocol events and observations
/// remain owned by the execution session and are never filtered here.
@MainActor enum WorkflowExecutionObservation {
    static func connect(changes: AnyPublisher<Void, Never>, phases: AnyPublisher<WorkflowExecutionPhase, Never>,
                        isCurrent: @escaping () -> Bool, publish: @escaping () -> Void) -> AnyCancellable {
        var active = true
        var updates: AnyCancellable?
        updates = changes.throttle(for: .milliseconds(200), scheduler: RunLoop.main, latest: true).sink { _ in
            guard active, isCurrent() else { return }
            publish()
        }
        let transitions = phases.dropFirst().removeDuplicates().sink { phase in
            guard active, isCurrent() else { return }
            if [.completed, .failed, .cancelled].contains(phase) {
                updates?.cancel(); active = false
            }
            publish()
        }
        return AnyCancellable { active = false; updates?.cancel(); transitions.cancel() }
    }
}
