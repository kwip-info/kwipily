import SwiftUI
import Combine
import KwipMLCore

struct WorkflowCreationEntry: Identifiable {
    let id: String
    let component: ComponentDefinition
    let iconType: String
    let payload: WorkflowNodePayload
    var definition: CompositeDefinition? = nil
}

struct WorkflowPortInspection: Equatable {
    let endpoint: WorkflowEndpoint
    let direction: PortDirection
    let requestID: UUID
}

/// Captures the saved draft's editor generation so an old banner action cannot
/// open a replacement draft or borrow a replacement owner's authority.
struct WorkflowSavedDraftRecovery: Equatable {
    let address: WorkflowNodeAddress
    let title: String
    let lease: WorkflowEditorLease
}

private struct WorkflowCanvasTensorInference {
    let projections: [ModelTensorProjection]
    let configuredDefinitions: [WorkflowNodeAddress: ComponentDefinition]
    var issues: [WorkflowModelValidationIssue] = []
}

/// A rendering and interaction adapter, never an executable graph or a second authoring store.
@MainActor final class WorkflowCanvasAdapter: ObservableObject, CanvasWorkspace {
    let owner: WorkflowAuthoringStore
    let modelRegistry: ComponentRegistry
    let dataRegistry: DataFlowOperatorRegistry
    private let allCreationEntries: [WorkflowCreationEntry]
    var creationEntries: [WorkflowCreationEntry] { allCreationEntries.filter {componentAvailability($0.iconType)} }
    var componentAvailability: (String) -> Bool = { _ in true }
    private let creationByID: [String: WorkflowCreationEntry]
    @Published private(set) var projection: WorkflowCanvasProjection
    @Published private(set) var projectionRevision = UUID()
    // Camera packets repaint GraphCanvasView locally. Forwarding them through
    // objectWillChange also rebuilt the host, inspector, Results and ribbon.
    // A separate publisher preserves immediate sidecar scheduling and flushes.
    let canvasViewStateUpdates=CurrentValueSubject<CanvasViewState,Never>(.init())
    var canvasViewState = CanvasViewState() {
        willSet {
            guard newValue != canvasViewState else {return}
            if newValue.presentation != canvasViewState.presentation || newValue.overviewVisible != canvasViewState.overviewVisible
                || newValue.selectedNodeIDs != canvasViewState.selectedNodeIDs || newValue.selectedEdgeIDs != canvasViewState.selectedEdgeIDs {
                objectWillChange.send()
            }
            canvasViewStateUpdates.send(newValue)
        }
    }
    @Published var inlineContainers:[CanvasInlineContainer]=[]
    var inlineState=WorkflowInlinePresentation()
    var mixtureInlinePorts:[String:CanvasPortDescriptor]=[:]
    func inlinePortKey(_ address:WorkflowNodeAddress,_ port:String,_ direction:PortDirection)->String {address.localID+":"+port+":"+(direction == .input ? "in":"out")}
    @Published var canvasViewRevision = UUID()
    @Published private(set) var canvasRevealRequest:CanvasRevealRequest?
    @discardableResult func revealLessonNode(_ address:WorkflowNodeAddress)->Bool {
        guard active,allowsInspection,let id=projection.displayIDByAddress[address],graph.nodes.contains(where:{$0.id==id}) else {return false}
        canvasRevealRequest = .init(nodeIDs:[id])
        return true
    }
    func revealSelection() {
        guard !isBusy,!selectedNodeIDs.isEmpty else {return}
        canvasRevealRequest = .init(nodeIDs:selectedNodeIDs)
    }
    func fitSelection() {
        guard active else { return }
        var ids = selectedNodeIDs
        for edge in owner.document.edges where owner.selection.edgeIDs.contains(edge.id) {
            if let id = projection.displayIDByAddress[edge.from.address] { ids.insert(id) }
            if let id = projection.displayIDByAddress[edge.to.address] { ids.insert(id) }
        }
        guard !ids.isEmpty else { return }
        canvasRevealRequest = .init(nodeIDs: ids, fitsBounds: true)
    }
    func fitWorkflow() {
        guard active, !graph.nodes.isEmpty else { return }
        canvasRevealRequest = .init(nodeIDs: Set(graph.nodes.map(\.id)), fitsBounds: true)
    }
    @Published private(set) var portInspection: WorkflowPortInspection?
    func dismissPortInspection() { portInspection = nil }
    var currentPortInspection: WorkflowPortInspection? {
        guard active, owner.isCurrent(owner.lease()), let inspection = portInspection,
              projection.portDescriptor(address:inspection.endpoint.address,port:inspection.endpoint.port,direction:inspection.direction) != nil else { return nil }
        return inspection
    }
    var inspectedSourceEndpoints: [WorkflowEndpoint] {
        guard let inspection = currentPortInspection else { return [] }
        if inspection.direction == .output { return [inspection.endpoint] }
        return inlineInspectorEdges.filter { $0.to == inspection.endpoint }.sorted { $0.id < $1.id }.map(\.from)
    }
    @discardableResult func inspectPort(nodeID: String, port: String, direction: PortDirection) -> Bool {
        guard active, owner.isCurrent(owner.lease()), let address = projection.addressByDisplayID[nodeID],
              projection.portDescriptor(address:address,port:port,direction:direction) != nil else { return false }
        // Read-only presentation remains available while execution locks authoring.
        portInspection = .init(endpoint:.init(address:address,port:port),direction:direction,requestID:UUID())
        statusMessage = "Inspecting " + direction.rawValue + " " + port + ". Captured output is separate from the declared contract."
        return true
    }
    func canvasPortBadge(nodeID: String, port: String, direction: PortDirection) -> String {
        let descriptor = canvasPortDescriptor(nodeID:nodeID,port:port,direction:direction)
        if direction == .input { return port + " · declared " + descriptor.kind.rawValue }
        let snapshot=projection.addressByDisplayID[nodeID].flatMap {address in
            address.domain == .model ? (outputProvider as? WorkflowWorkspaceSession)?.modelSnapshot(for:address):outputProvider?.snapshot(for:address)
        }
        return WorkflowCanvasRenderEvidence.badge(port:port,kind:descriptor.kind,snapshot:snapshot)
    }
    @Published var pendingSourceID: String?
    @Published var statusMessage = "Authoring workflow · inspect captured execution in Results or History."
    private weak var childRequests: WorkflowChildWorkspace?
    private var childRequestObservation: AnyCancellable?
    var footerStatusMessage: String { childRequests?.openingError ?? statusMessage }
    func observeChildRequests(_ children: WorkflowChildWorkspace) {
        childRequests = children
        childRequestObservation = children.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    // Non-published diagnostic count: hovering must not itself trigger rendering.
    private(set) var connectionValidationCount = 0
    private struct ConnectionQuery:Hashable {
        let from:WorkflowEndpoint
        let to:WorkflowEndpoint
    }
    private var connectionCompatibility:[ConnectionQuery:Bool] = [:]
    private var feedbackProjectionRevision:UUID?
    private var feedbackConnectionIssues:[String:[ValidationIssue]] = [:]
    private var iconProjectionRevision:UUID?
    private var projectedIconTypes:[String:String] = [:]
    private(set) var feedbackConnectionValidationCount=0
    /// Split-view layout must not reconstruct container edges for every node on
    /// every frame. Projection revisions include document and inferred-port edits.
    func connectionFeedbackIssues(for nodeID:String)->[ValidationIssue] {
        if feedbackProjectionRevision != projectionRevision {
            feedbackConnectionIssues.removeAll(keepingCapacity:true)
            feedbackProjectionRevision=projectionRevision
        }
        if let issues=feedbackConnectionIssues[nodeID] {return issues}
        let issues=CanvasNodeFeedback.connectionIssues(nodeID:nodeID,document:cachedDocument,projection:projection)
        feedbackConnectionIssues[nodeID]=issues
        feedbackConnectionValidationCount+=1
        return issues
    }
    @Published private(set) var isInferringTensors=false
    private(set) var tensorInferenceCount=0
    private var tensorInput:WorkflowDocument?
    private var tensorProjections:[ModelTensorProjection]=[]
    private var modelValidationIssues:[WorkflowModelValidationIssue]=[]
    private var configuredModelDefinitions:[WorkflowNodeAddress:ComponentDefinition]=[:]
    private var tensorRequest=UUID()
    private var tensorWorker:Task<WorkflowCanvasTensorInference,Never>?
    private var tensorPublication:Task<Void,Never>?
    private var modelInputEndpoints=Set<WorkflowEndpoint>()
    private var cachedDocument: WorkflowDocument
    private var projectionError: String?
    private var active = true
    private var observations: [AnyCancellable] = []
    weak var outputProvider: (any WorkflowDataOutputs)?

    var onEditPrivateBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)?
    var onOpenBlock: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)?
    var onSelectSource: ((WorkflowNodeAddress, WorkflowEditorLease) -> Void)?

    init(owner: WorkflowAuthoringStore, modelRegistry: ComponentRegistry, dataRegistry: DataFlowOperatorRegistry) throws {
        guard modelRegistry == owner.modelRegistry, dataRegistry == owner.dataRegistry else {
            throw DataFlowContractError("Workflow display registries must match the authoritative editor registries. Reopen the editor with its original contracts.")
        }
        self.owner = owner; self.modelRegistry = modelRegistry; self.dataRegistry = dataRegistry
        cachedDocument = owner.document
        projection = try WorkflowCanvasProjector.project(owner.document, modelRegistry: modelRegistry, dataRegistry: dataRegistry, inlineTrainControls: true)
        allCreationEntries = try Self.catalog(modelRegistry: modelRegistry, dataRegistry: dataRegistry)
        creationByID = Dictionary(uniqueKeysWithValues: allCreationEntries.map { ($0.id, $0) })
        modelInputEndpoints=Set(cachedDocument.edges.filter {projection.displayIDByAddress[$0.from.address]==nil && $0.from.port=="out" && $0.kind == .tensor}.map(\.to))
        observations.append(owner.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() })
        // @Published delivers before owner.document changes; rebuild from the delivered value.
        observations.append(owner.$document.sink { [weak self] document in
            guard let self, document != self.cachedDocument else { return }
            self.connectionCompatibility.removeAll(keepingCapacity:true)
            do {
                let profile = InteractionProfiler.shared.begin(.projection)
                defer { InteractionProfiler.shared.end(profile) }
                self.refreshTensorInference(document)
                let next = try self.projection.updatingPresentation(document)
                    ?? WorkflowCanvasProjector.project(document, modelRegistry: self.modelRegistry, dataRegistry: self.dataRegistry,
                                                       configuredModelDefinitions: self.configuredModelDefinitions, inlineTrainControls: true)
                self.cachedDocument = document
                self.projectionError = nil
                self.projection = try self.makeInlineProjection(next,document:document)
                self.modelInputEndpoints=Set(document.edges.filter {next.displayIDByAddress[$0.from.address]==nil && $0.from.port=="out" && $0.kind == .tensor}.map(\.to))
                self.projectionRevision = UUID()
                if let pending = self.pendingSourceID, next.addressByDisplayID[pending] == nil { self.pendingSourceID = nil }
            } catch {
                self.projectionError = error.localizedDescription
                self.statusMessage = "Workflow display unavailable: " + error.localizedDescription
            }
        })
        inlineState=WorkflowInlinePresentation.load(projectID:owner.projectID)
        refreshInlinePresentation()
        refreshTensorInference(owner.document)
    }
    func refreshInlinePresentation() {
        do {
            let base=try WorkflowCanvasProjector.project(owner.document,modelRegistry:modelRegistry,dataRegistry:dataRegistry,
                configuredModelDefinitions:configuredModelDefinitions,inlineTrainControls:true)
            projection=try makeInlineProjection(base,document:owner.document)
            projectionRevision=UUID();connectionCompatibility.removeAll(keepingCapacity:true)
        } catch {statusMessage="Container display unavailable: "+error.localizedDescription}
    }
    func modelConfigurationIssue(for address:WorkflowNodeAddress)->String? {configurationIssue(for:address)}
    func configurationIssue(for address:WorkflowNodeAddress)->String? {
        let messages=(modelValidationIssues+sourceConfigurationIssues).filter {$0.address==address}.map(\.message)
        return messages.isEmpty ? nil:Array(Set(messages)).sorted().joined(separator:"\n")
    }
    func plannedInputLayout(for address:WorkflowNodeAddress,port:String="in")->BatchedTensorLayout? {
        plannedInputContract(for:address,port:port)?.layout
    }
    func plannedInputContract(for address:WorkflowNodeAddress,port:String="in")->TypedTensorContract? {
        guard !isInferringTensors else {return nil}
        let path:[String],localID:String
        if let child=try? WorkflowInlineContainer.parse(in:owner.document,address:address) {path=[child.root.localID];localID=child.childID}
        else {path=[];localID=address.localID}
        let relevant=tensorProjections.filter {$0.containsInput(instancePath:path,nodeID:localID,port:port)}
        let candidates=relevant.compactMap {$0.inputContract(instancePath:path,nodeID:localID,port:port)}
        guard !relevant.isEmpty,candidates.count==relevant.count,let first=candidates.first,candidates.allSatisfy({$0==first}) else {return nil}
        return first
    }
    private nonisolated static func modelValidationContext(_ document:WorkflowDocument,definition:WorkflowNodeAddress)->GraphValidationContext {
        // Follow only this architecture's definition aliases to its Train consumers.
        // An unrelated objective elsewhere cannot relax a binary model's contract.
        var visited=Set<WorkflowNodeAddress>(),pending=[definition]
        while let address=pending.popLast() {
            guard visited.insert(address).inserted else {continue}
            for edge in document.edges where edge.from.address==address && edge.kind == .modelDefinition {
                if let target=document.nodes.first(where:{$0.address==edge.to.address}),SelfPlayConfiguration.isSelfPlay(target) {return .policyValue}
                if let target=document.nodes.first(where:{$0.address==edge.to.address}),case .gameSession=target.payload {return .frozenPolicy}
                if edge.to.port=="model",let train=document.nodes.first(where:{$0.address==edge.to.address}),case .train(let c)=train.payload,
                   (try? WorkflowTrainRecipes(configuration:c,schemaVersion:document.schemaVersion).objective) != nil || document.edges.contains(where:{$0.to.address==edge.to.address && $0.to.port=="objective" && $0.kind == .objectiveRecipe}) {return .explicitObjective}
                pending.append(edge.to.address)
            }
        }
        return .preparedWorkflow
    }
    private func refreshTensorInference(_ document:WorkflowDocument) {
        var input=document
        input.revision=0;input.name=""
        for index in input.nodes.indices {input.nodes[index].name="";input.nodes[index].x=0;input.nodes[index].y=0}
        guard input != tensorInput else {return}
        tensorInput=input;tensorProjections=[];modelValidationIssues=[];configuredModelDefinitions=[:];tensorInferenceCount+=1
        tensorWorker?.cancel();tensorPublication?.cancel()
        let request=UUID();tensorRequest=request;isInferringTensors=true
        let models=modelRegistry,data=dataRegistry
        let worker=Task.detached(priority:.utility) {()->WorkflowCanvasTensorInference in
            let profile = InteractionProfiler.shared.begin(.tensorInference)
            defer { InteractionProfiler.shared.end(profile) }
            var results:[ModelTensorProjection]=[]
            var issues:[WorkflowModelValidationIssue]=[]
            var definitions:[WorkflowNodeAddress:ComponentDefinition]=[:]
            var conflicts=Set<WorkflowNodeAddress>()
            for node in input.nodes {
                guard !Task.isCancelled else {return .init(projections:[],configuredDefinitions:[:])}
                guard WorkflowModelBoundary.isVisibleDefinition(node,in:input) else {continue}
                do {
                    let context=Self.modelValidationContext(input,definition:node.address)
                    let snapshot=try WorkflowCompiler.modelSnapshot(input,definition:node.address,modelRegistry:models,dataRegistry:data,context:context)
                    results.append(ModelTensorProjection(snapshot:snapshot,registry:models,context:context))
                    let validation=GraphValidator.validate(snapshot.graph,registry:models,context:context)
                    for primitive in snapshot.graph.nodes where TensorRoutingConfiguration.componentIDs.contains(primitive.type) {
                        guard let origin=snapshot.origins[primitive.id],origin.instancePath.isEmpty,
                              let definition=validation.configuredDefinition(nodeID:primitive.id,graph:snapshot.graph,registry:models) else {continue}
                        let address=WorkflowNodeAddress(domain:.model,localID:origin.localNodeID)
                        if let previous=definitions[address],previous != definition {conflicts.insert(address)}
                        else {definitions[address]=definition}
                    }
                } catch let failure as WorkflowCompilerError {
                    if !failure.modelIssues.isEmpty { issues.append(contentsOf: failure.modelIssues) }
                    else if let address = failure.node { issues.append(.init(address: address, message: failure.message)) }
                } catch { issues.append(.init(address: node.address, message: error.localizedDescription)) }
            }
            if input.namedModelInputs?.isEmpty == false {
                for node in input.nodes {
                    guard !Task.isCancelled else {return .init(projections:[],configuredDefinitions:[:])}
                    guard case .train=node.payload else {continue}
                    do {if SelfPlayConfiguration.isSelfPlay(node) {_=try SelfPlayPlan.compile(input,address:node.address,modelRegistry:models,dataRegistry:data)}else if FineTuningConfiguration.isExternal(node) {_=try WorkflowFineTuningPlan.compile(input,train:node.address,dataRegistry:data)}else {_ = try WorkflowCompiler.compile(input,train:node.address,modelRegistry:models,dataRegistry:data,contractVersion:16)}}
                    catch let failure as WorkflowCompilerError {
                        let findings=failure.modelIssues.isEmpty ? failure.node.map {[WorkflowModelValidationIssue(address:$0,message:failure.message)]} ?? []:failure.modelIssues
                        for finding in findings where !issues.contains(finding) {issues.append(finding)}
                    } catch {issues.append(.init(address:node.address,message:error.localizedDescription))}
                }
            }
            // Shared nodes with disagreeing model contexts retain pending declarations;
            // choosing one interpretation would falsely grant a connectable endpoint.
            for address in conflicts {definitions.removeValue(forKey:address)}
            return .init(projections:results,configuredDefinitions:definitions,issues:issues)
        }
        tensorWorker=worker
        tensorPublication=Task { [weak self] in
            let result=await worker.value
            guard !Task.isCancelled,let self,self.active,self.tensorRequest == request else {return}
            self.tensorProjections=result.projections
            self.modelValidationIssues=result.issues
            self.configuredModelDefinitions=result.configuredDefinitions
            do {
                let base=try WorkflowCanvasProjector.project(self.cachedDocument,modelRegistry:self.modelRegistry,dataRegistry:self.dataRegistry,
                    configuredModelDefinitions:self.configuredModelDefinitions, inlineTrainControls: true)
                self.projection=try self.makeInlineProjection(base,document:self.cachedDocument)
                self.projectionRevision=UUID()
            } catch {
                self.projectionError=error.localizedDescription
                self.statusMessage="Workflow display unavailable: " + error.localizedDescription
            }
            self.connectionCompatibility.removeAll(keepingCapacity:true)
            self.isInferringTensors=false
            self.tensorWorker=nil;self.tensorPublication=nil
        }
    }
    func invalidate() {
        objectWillChange.send(); active = false; observations.removeAll()
        childRequestObservation = nil; childRequests = nil
        tensorRequest=UUID();tensorWorker?.cancel();tensorPublication?.cancel();tensorWorker=nil;tensorPublication=nil;tensorProjections=[];modelValidationIssues=[];configuredModelDefinitions=[:];isInferringTensors=false
        connectionCompatibility.removeAll()
        onOpenBlock = nil; onEditPrivateBlock = nil; onSelectSource = nil; pendingSourceID = nil
    }
    func canvasRenderEvidence(validation:ValidationResult)->WorkflowCanvasRenderEvidence {
        WorkflowCanvasRenderEvidence.make(adapter:self,active:active,validation:validation)
    }
    var graph: GraphDocument { projection.graph }
    var registry: ComponentRegistry { projection.registry }
    var canvasCatalog: ComponentRegistry { .init(components: creationEntries.map(\.component)) }
    var allowsInspection: Bool { active && owner.isCurrent(owner.lease()) && owner.document.drafts.isEmpty && projectionError == nil }
    var isBusy: Bool { !active || !owner.isCurrent(owner.lease()) || owner.isBusy || !owner.document.drafts.isEmpty || projectionError != nil }
    var savedDraftRecovery: WorkflowSavedDraftRecovery? {
        guard active, owner.isCurrent(owner.lease()),
              let root=owner.document.drafts.first?.instancePath.first,
              let node=owner.document.nodes.first(where:{$0.address.domain == .model && $0.address.localID == root}),
              case .modelBlock(let reference,_)=node.payload,
              let definition=owner.document.definitions.first(where:{$0.reference == reference}) else {return nil}
        return .init(address:node.address,title:node.name.isEmpty ? definition.title:node.name,lease:owner.lease())
    }
    func canOpenSavedDraft(_ recovery:WorkflowSavedDraftRecovery)->Bool {
        guard owner.isCurrent(recovery.lease),savedDraftRecovery == recovery,
              let display=projection.displayIDByAddress[recovery.address] else {return false}
        return canOpenBlock(display)
    }
    @discardableResult func openSavedDraft(_ recovery:WorkflowSavedDraftRecovery)->Bool {
        guard canOpenSavedDraft(recovery),let callback=onOpenBlock else {return false}
        callback(recovery.address,recovery.lease)
        return true
    }
    var selectedNodeIDs: Set<String> { Set(owner.selection.nodeAddresses.compactMap { projection.displayIDByAddress[$0] }) }
    var selectedEdgeID: String? { owner.selection.edgeIDs.sorted().compactMap { projection.displayIDByEdgeID[$0] }.first }
    /// Reuses asynchronously computed findings; never compiles a workflow during a render.
    /// Findings outside the chosen training path must not disable a valid independent path.
    var currentRunConfigurationFinding:WorkflowModelValidationIssue? {
        let terminals=cachedDocument.nodes.filter {switch $0.payload {case .train,.estimator:return true;default:return false}}
        let selected=terminals.filter {owner.selection.nodeAddresses.contains($0.address)}
        let terminal=selected.count==1 ? selected.first:terminals.count==1 ? terminals.first:nil
        guard let terminal else {return nil}
        var reachable:Set<WorkflowNodeAddress>=[terminal.address],queue=[terminal.address]
        while let next=queue.popLast() {
            for edge in cachedDocument.edges where edge.to.address==next {
                if reachable.insert(edge.from.address).inserted {queue.append(edge.from.address)}
            }
        }
        return (modelValidationIssues+sourceConfigurationIssues).first {reachable.contains($0.address)}
    }
    private var sourceFeedbackCache:(projection:UUID,bindings:UInt64,issues:[WorkflowModelValidationIssue])?
    private var sourceConfigurationIssues:[WorkflowModelValidationIssue] {
        guard active,let host=outputProvider as? WorkflowWorkspaceSession,host.canvas === self,!host.isRetired,host.owner === owner else {return []}
        if let cached=sourceFeedbackCache,cached.projection==projectionRevision,cached.bindings==host.bindings.revision {return cached.issues}
        let issues=host.bindings.configurationIssues
        sourceFeedbackCache=(projectionRevision,host.bindings.revision,issues)
        return issues
    }
    var validation: ValidationResult {
        var issues = projection.issues.enumerated().map { index, issue in
            ValidationIssue(id: "workflow-presentation-\(index)", message: issue.message, nodeID: projection.displayIDByAddress[issue.address])
        }
        for (index, issue) in modelValidationIssues.enumerated() {
            issues.append(.init(id: "workflow-model-\(index)", message: issue.message, nodeID: projection.displayIDByAddress[issue.address]))
        }
        for issue in sourceConfigurationIssues {
            issues.append(.init(id:"workflow-source-"+issue.address.localID,message:issue.message,nodeID:projection.displayIDByAddress[issue.address]))
        }
        if let projectionError { issues.append(.init(id: "workflow-projection", message: projectionError)) }
        return .init(issues: issues) // No inferred widths or executable topological order.
    }
    var previewWeightLabel: String { "Captured observations only · current sources not revalidated" }
    func previewStatus(for id: String) -> String {
        guard active, owner.isCurrent(owner.lease()), let provider=outputProvider,
              provider.projectID == owner.projectID, provider.sessionID == owner.sessionID,
              let address=projection.addressByDisplayID[id] else { return "Not evaluated" }
        if let host = provider as? WorkflowWorkspaceSession,
           host.attributedPreparationFailure?.currentCause(for: address) != nil { return "Failed" }
        if address.domain == .control { return provider.terminalStatus(for: address) ?? "Not evaluated" }
        guard [.data,.model].contains(address.domain),
              let capture=provider.snapshot(for:address), capture.address == address,
              capture.projectID == owner.projectID, capture.sessionID == owner.sessionID else {return "Not evaluated"}
        guard capture.matchesCurrentConstruction && capture.matchesCurrentBindings else {return "Stale"}
        return capture.phase == .completed ? "Captured" : "Partial capture"
    }
    func canvasEvidenceLabel(nodeID: String) -> String {
        switch previewStatus(for: nodeID) {
        case "Completed": return "Completed component · inspect its recorded evidence in Results or History."
        case "Running": return "Estimator execution in progress · inspect Results."
        case "Failed":
            if let address=projection.addressByDisplayID[nodeID], let host=outputProvider as? WorkflowWorkspaceSession,
               let cause=host.attributedPreparationFailure?.currentCause(for:address) { return "Run failed · " + cause }
            return "Execution failed · inspect Results → Diagnostics for the cause."
        case "Cancelled": return "Estimator execution cancelled · inspect Results for retained evidence."
        case "Captured": return "Captured output · current sources not revalidated."
        case "Partial capture": return "Partial captured output · inspect run status in Results."
        case "Stale": return "Captured output is stale · rerun to reflect current inputs."
        default: return "No captured output for this component."
        }
    }
    var historyContextID: String { "workflow:" + owner.projectID + ":" + owner.sessionID.uuidString }
    var historyContextIsActive: Bool { !isBusy }
    var canUndo: Bool { !isBusy && owner.canUndo }
    var canRedo: Bool { !isBusy && owner.canRedo }
    func undo() { let profile = InteractionProfiler.shared.begin(.undo); defer { InteractionProfiler.shared.end(profile) }; guard canUndo else { return }; report(owner.undo(lease: owner.lease())) }
    func redo() { let profile = InteractionProfiler.shared.begin(.redo); defer { InteractionProfiler.shared.end(profile) }; guard canRedo else { return }; report(owner.redo(lease: owner.lease())) }

    @discardableResult func apply(_ command: WorkflowCommand, lease: WorkflowEditorLease) -> Bool {
        guard !isBusy else { statusMessage = "This workflow editor is unavailable or busy."; return false }
        let accepted = owner.apply(command, lease: lease) != nil
        report(accepted); return accepted
    }
    @discardableResult func rename(nodeID: String, name: String, lease: WorkflowEditorLease) -> Bool {
        guard let address = projection.addressByDisplayID[nodeID] else { return false }
        return apply(.rename(address: address, name: name), lease: lease)
    }
    @discardableResult func configure(nodeID: String, config: [String: JSONValue], lease: WorkflowEditorLease) -> Bool {
        guard let address = projection.addressByDisplayID[nodeID] else { return false }
        return apply(.configure(address: address, config: config), lease: lease)
    }
    @discardableResult func insert(payload: WorkflowNodePayload, address: WorkflowNodeAddress,
                                   name: String = "", x: Double, y: Double, lease: WorkflowEditorLease) -> Bool {
        apply(.insert(.init(address: address, name: name, x: x, y: y, payload: payload)), lease: lease)
    }
    private func report(_ accepted: Bool) {
        statusMessage = accepted ? "Workflow source saved." : (owner.error ?? "Workflow change was not applied.")
    }
    func selectNode(_ id: String, extending: Bool) {
        let profile = InteractionProfiler.shared.begin(.selection)
        defer { InteractionProfiler.shared.end(profile) }
        guard allowsInspection, let address = projection.addressByDisplayID[id] else { return }
        portInspection = nil
        var next = extending ? owner.selection : WorkflowSelection()
        if extending && next.nodeAddresses.contains(address) { next.nodeAddresses.remove(address) }
        else { next.nodeAddresses.insert(address) }
        next.edgeIDs = []; _ = owner.select(next, lease: owner.lease())
    }
    func selectEdge(_ id: String) {
        guard allowsInspection, let edge = projection.edgeIDByDisplayID[id] else { return }
        portInspection = nil
        _ = owner.select(.init(edgeIDs: [edge]), lease: owner.lease())
    }
    func clearSelection() {
        guard allowsInspection else { return }; portInspection = nil; _ = owner.select(.init(), lease: owner.lease()); pendingSourceID = nil
    }
    func moveNodes(ids: Set<String>, dx: Double, dy: Double) {
        let profile = InteractionProfiler.shared.begin(.moveCommit)
        defer { InteractionProfiler.shared.end(profile) }
        if moveInlineNodes(ids:ids,dx:dx,dy:dy) {return}
        let addresses = Set(ids.compactMap { projection.addressByDisplayID[$0] })
        guard addresses.count == ids.count else { return }
        _ = apply(.move(addresses: addresses, offset: .init(x: dx, y: dy)), lease: owner.lease())
    }
    func duplicateSelection() { _ = apply(.duplicate(addresses: owner.selection.nodeAddresses, offset: .init(x: 30, y: 30)), lease: owner.lease()) }
    func deleteSelection() { _ = apply(.delete(owner.selection), lease: owner.lease()) }
    func canAddType(_ type: String) -> Bool {
        guard !isBusy,let entry=creationByID[type] else {return false}
        return componentAvailability(entry.iconType)
    }
    func addNode(type: String, x: Double, y: Double) {
        guard !isBusy, let entry = creationByID[type] else { statusMessage = "Choose a component from the workflow creation catalog."; return }
        guard componentAvailability(entry.iconType) else {
            statusMessage="This component requires verified paired bag runtime support. Refresh runtime status before placing it."
            return
        }
        let payload=entry.payload
        let address = WorkflowNodeAddress(domain: payload.domain, localID: UUID().uuidString)
        if case .modelPrimitive(let component,_,let config)=payload,let root=inlineDropRoot(at:CGPoint(x:x,y:y)),let parent=owner.document.nodes.first(where:{$0.address==root}) {
            let childID:String
            if let chains=try? WorkflowMixtureContainer.chains(in:owner.document,root:root){
                let experts=chains.filter{$0.key != "router"}
                let geometry=CanvasGeometrySnapshot(registry:registry,presentation:canvasViewState.presentation,graph:graph,validation:validation)
                let offset=geometry.projection.project(CGPoint(x:x-parent.x,y:y-parent.y))
                let index=min(max(Int(((offset.y-95)/225).rounded()),0),experts.count-1)
                let chain=experts[index]
                childID=WorkflowMixtureContainer.qualified(chain.id,address.localID)
            }else{childID=address.localID}
            let child=GraphNode(id:childID,type:component,name:entry.component.title,x:x-parent.x,y:y-parent.y,config:config)
            if apply(.inlineContainer(root:root,command:.insertNode(child)),lease:owner.lease()) {
                let selected=WorkflowInlineContainer.address(root:root,childID:child.id)
                inlineState.positions[selected.localID] = .init(x:child.x,y:child.y);inlineState.save(projectID:owner.projectID);refreshInlinePresentation()
                _=owner.select(.init(nodeAddresses:[selected]),lease:owner.lease())
            }
            return
        }
        if let definition = entry.definition {
            _ = apply(.insertBlock(.init(address: address, name: entry.component.title, x: x, y: y, payload: entry.payload), definition: definition), lease: owner.lease())
        } else {
            let name:String
            switch payload {
            case .datasetRecipe, .batchRecipe, .saveModel: name=entry.component.title
            default: name=""
            }
            _ = insert(payload: payload, address: address, name: name, x: x, y: y, lease: owner.lease())
        }
    }
    func canvasInsertionTypes(edgeID: String) -> [String] { [] }
    func insertNode(type: String, edgeID: String, x: Double, y: Double) {
        statusMessage = "Wire insertion is not yet supported. Place a component and connect its ports explicitly."
    }
    private func endpoints(source: String, sourcePort: String, target: String, targetPort: String) -> (WorkflowEndpoint, WorkflowEndpoint)? {
        guard let from = projection.addressByDisplayID[source], let to = projection.addressByDisplayID[target] else { return nil }
        if ["optimizer", "scheduler", "objective", "initialization", "gradientPolicy", "trainability", "dataset", "loader"].contains(targetPort), cachedDocument.nodes.contains(where: { guard $0.address == to else { return false }; if case .train = $0.payload { return true }; return false }) { return nil }
        if let definition=configuredModelDefinitions[from], !definition.outputs.contains(sourcePort) {return nil}
        if let definition=configuredModelDefinitions[to], !definition.inputs.contains(targetPort) {return nil}
        return (.init(address: from, port: sourcePort), .init(address: to, port: targetPort))
    }
    func canConnect(source: String, sourcePort: String, target: GraphNode, port: String) -> Bool {
        let profile = InteractionProfiler.shared.begin(.connectionCheck)
        defer { InteractionProfiler.shared.end(profile) }
        guard !isBusy, let (from, to) = endpoints(source: source, sourcePort: sourcePort, target: target.id, targetPort: port) else { return false }
        let query=ConnectionQuery(from:from,to:to)
        if let cached=connectionCompatibility[query] {return cached}
        connectionValidationCount += 1
        let compatible=(try? WorkflowEditing.apply(owner.authorizingInlineRetirement(owner.inlineCommand(.connect(from: from, to: to))), to: owner.document,
            modelRegistry: modelRegistry, dataRegistry: dataRegistry)) != nil
        // Bound transient rendering metadata even if clients query arbitrary port IDs.
        if connectionCompatibility.count >= 4096 {connectionCompatibility.removeAll(keepingCapacity:true)}
        connectionCompatibility[query]=compatible
        return compatible
    }
    func connect(source: String, sourcePort: String, target: String, targetPort: String) {
        let profile = InteractionProfiler.shared.begin(.connectionCommit)
        defer { InteractionProfiler.shared.end(profile) }
        guard let (from, to) = endpoints(source: source, sourcePort: sourcePort, target: target, targetPort: targetPort) else { return }
        if apply(.connect(from: from, to: to), lease: owner.lease()) { pendingSourceID = nil }
    }
    func reconnect(edgeID: String, source: String, sourcePort: String, target: String, targetPort: String) {
        guard let edge = projection.edgeIDByDisplayID[edgeID],
              let (from, to) = endpoints(source: source, sourcePort: sourcePort, target: target, targetPort: targetPort) else { return }
        if let internalEdge=WorkflowInlineContainerProjection.internalEdge(in:owner.document,displayID:edge) {
            _=apply(.inlineContainer(root:internalEdge.root,command:.batch([.disconnect(edgeID:internalEdge.edgeID),.connectWorkflow(from:from,to:to)])),lease:owner.lease())
        } else {_ = apply(.reconnect(edgeID: edge, from: from, to: to), lease: owner.lease())}
    }
    private func node(_ displayID: String) -> WorkflowNode? {
        guard let address = projection.addressByDisplayID[displayID] else { return nil }
        return owner.authoringNode(at:address)
    }
    func canvasIconType(nodeID: String) -> String {
        if iconProjectionRevision != projectionRevision {
            projectedIconTypes.removeAll(keepingCapacity:true)
            iconProjectionRevision=projectionRevision
        }
        if let icon=projectedIconTypes[nodeID] {return icon}
        let source=node(nodeID)
        let icon:String
        if let source, case .modelBlock(let reference, _) = source.payload,
           let definition = owner.document.definitions.first(where: { $0.reference == reference }), definition.composition != nil {
            icon=definition.resolvedAppearance?.rawValue ?? ModuleCompositionPresentation.iconType(definition.composition?.kind)
        } else {icon=source.map {Self.icon($0.payload)} ?? "unknown"}
        // Retain only current projected components; arbitrary misses cannot grow it.
        if source != nil {projectedIconTypes[nodeID]=icon}
        return icon
    }
    func canvasSummary(_ node: GraphNode) -> String {
        guard let source = self.node(node.id) else { return "Unavailable source" }
        switch source.payload {
        case .modelPrimitive(_, let version, _): return "Model template · configuration v\(version)"
        case .dataOperator: return "Data operation"
        case .modelBlock: return "Pinned Block definition"
        case .modelDefinition: return "Model definition reference"
        case .trainingData: return "Explicit feature and target mapping"
        case .optimizerRecipe: return "Optimizer recipe"
        case .schedulerRecipe: return "Scheduler recipe"
        case .lossRecipe: return "Objective recipe"
        case .initializationRecipe: return "Starting settings recipe"
        case .gradientPolicyRecipe: return "Gradient and update recipe"
        case .datasetRecipe: return "Dataset view recipe"
        case .batchRecipe: return "Batch reader recipe"
        case .trainabilityRecipe: return "Trainable settings recipe"
        case .runSetup: return "Run configuration"
        case .estimator(let values):
            let document = DocumentClassificationConfiguration.isDocument(values)
            switch values["operation"]?.stringValue ?? "fit" {
            case "predict": return document ? "Classify text with a frozen model" : "Predict with a frozen model"
            case "evaluate": return document ? "Evaluate labeled documents with a frozen model" : "Evaluate with a frozen model"
            default: return document ? "Fit Training-only document classification" : "Fit a model on prepared data"
            }
        case .train: return "Train and expose the verified trained model"
        case .frozenModel: return "Verified reusable model reference"
        case .gameSession:return "Run bounded frozen-policy episodes over authored rules"
        case .generateText: return "Generate bounded text from captured weights and tokenizer"
        case .saveModel: return "Save a verified model package or checkpoint"
        case .evaluateTest: return "Frozen Validation or Test evaluation terminal"
        }
    }
    func canvasPortDescriptor(nodeID: String, port: String, direction: PortDirection) -> CanvasPortDescriptor {
        if let address=projection.addressByDisplayID[nodeID],let member=mixtureInlinePorts[inlinePortKey(address,port,direction)]{return member}
        guard let address = projection.addressByDisplayID[nodeID],
              let descriptor = projection.portDescriptor(address: address, port: port, direction: direction),
              let contract = descriptor.contract else {
            let reason = projection.addressByDisplayID[nodeID].flatMap { projection.portDescriptor(address: $0, port: port, direction: direction)?.unresolvedReason }
            return .unknown(direction: direction, reason: reason ?? "Port contract unavailable")
        }
        let kind = CanvasPortKind(rawValue: contract.kind.rawValue) ?? .unknown
        if kind == .tensor,address.domain == .model {
            let candidates=tensorProjections.map {$0.port(instancePath:[],nodeID:address.localID,port:port,direction:direction)}.filter {$0.kind == .tensor}
            if var first=candidates.first,candidates.allSatisfy({$0 == first}) {
                if direction == .input,modelInputEndpoints.contains(.init(address:address,port:port)) {first.role="Model input"}
                return first
            }
        }
        var declared = ["Declared contract; not measured execution data"]
        if direction == .input,modelInputEndpoints.contains(.init(address:address,port:port)) {declared.append("Model input · configure in Model definition")}
        if contract.required { declared.append("Required") }
        if let maximum = contract.maximumConnections { declared.append("At most \(maximum) connection(s)") }
        return .init(kind: kind, direction: direction, declared: declared)
    }
    func canvasPortDescription(nodeID: String, port: String) -> String {
        guard let address = projection.addressByDisplayID[nodeID] else { return "Port contract unavailable" }
        return projection.portDescriptor(address: address, port: port, direction: .output)?.accessibilityDescription
            ?? projection.portDescriptor(address: address, port: port, direction: .input)?.accessibilityDescription
            ?? "Port contract unavailable"
    }
    func canOpenBlock(_ id: String) -> Bool {
        if active,owner.isCurrent(owner.lease()),!owner.isBusy,projectionError==nil,owner.document.drafts.isEmpty,
           inlineContainers.contains(where:{$0.id==id}){return true}
        guard active, owner.isCurrent(owner.lease()), !owner.isBusy, projectionError == nil,
              onOpenBlock != nil, let node = node(id), case .modelBlock(let reference, _) = node.payload,
              cachedDocument.definitions.contains(where: { $0.reference == reference }) else { return false }
        // Independent files remain reachable while the owning private draft is unfinished.
        return owner.flowRootURL != nil || owner.document.drafts.isEmpty || owner.document.drafts.first?.instancePath.first == node.address.localID
    }
    func openBlock(_ id: String) {
        if owner.document.drafts.isEmpty, inlineContainers.contains(where:{$0.id==id}) {toggleInlineContainer(id);return}
        guard canOpenBlock(id), let address = projection.addressByDisplayID[id] else { return }
        onOpenBlock?(address, owner.lease())
    }
    func selectSource(_ id: String) {
        guard !isBusy, let node = node(id), case .dataOperator(let operatorID, _) = node.payload, DataFlowSourceOperators.contains(operatorID) else { return }
        onSelectSource?(node.address, owner.lease())
    }
    nonisolated static func icon(_ payload: WorkflowNodePayload) -> String {
        switch payload {
        case .dataOperator(let id, _): return id
        case .modelPrimitive(let type, _, _): return type
        case .modelBlock(let reference, _):
            if reference == SeparableConv2dBlock.reference { return "separable_conv2d" }
            let v = reference.version
            return "composite:\(reference.id)@\(v.major).\(v.minor).\(v.patch)"
        case .modelDefinition: return "workflow.modelDefinition"
        case .trainingData: return "workflow.trainingData"
        case .optimizerRecipe(let type,_,_), .schedulerRecipe(let type,_,_), .lossRecipe(let type,_,_), .initializationRecipe(let type,_,_), .gradientPolicyRecipe(let type,_,_), .trainabilityRecipe(let type,_,_), .datasetRecipe(let type,_,_), .batchRecipe(let type,_,_): return type
        case .runSetup: return "workflow.runSetup"
        case .estimator(let config):
            switch config["backendID"]?.stringValue {
            case "tensorflow": return "workflow.kerasClassifier"
            case "sklearn", nil: return "workflow.logisticRegression"
            default: return "workflow.train"
            }
        case .train: return "workflow.train"
        case .frozenModel: return "workflow.frozenModel"
        case .gameSession:return "workflow.gameSession"
        case .generateText: return "workflow.generateText"
        case .saveModel: return "workflow.saveModel"
        case .evaluateTest: return "workflow.evaluateTest"
        }
    }
    static func catalog(modelRegistry: ComponentRegistry, dataRegistry: DataFlowOperatorRegistry) throws -> [WorkflowCreationEntry] {
        guard Set(modelRegistry.components.map(\.id)).count == modelRegistry.components.count,
              Set(dataRegistry.operators.map(\.id)).count == dataRegistry.operators.count,
              modelRegistry.components.allSatisfy({ Set($0.fields.map(\.key)).count == $0.fields.count }) else {
            throw DataFlowContractError("Workflow creation registries contain duplicate component or field identities.")
        }
        var result: [WorkflowCreationEntry] = []
        func append(kind: String, key: String, title: String, category: String, fields: [ComponentField] = [], payload: WorkflowNodePayload, definition: CompositeDefinition? = nil, iconOverride:String?=nil) throws {
            // Deterministic opaque creation identity. It is only looked up in a dictionary,
            // never parsed as a node-local projected type or a worker operator.
            let id = "wf-create-" + (try JSONEncoder().encode([kind, key])).base64EncodedString()
            let node = WorkflowNode(address: .init(domain: payload.domain, localID: "catalog-preview"), payload: payload)
            var document = WorkflowDocument(projectID: "00000000-0000-0000-0000-000000000000", nodes: [node], definitions: definition.map { [$0] } ?? [])
            // Train owns data access and batching inline in current authoring.
            switch payload {case .gameSession:document.schemaVersion=22;case .generateText:document.schemaVersion=19;case .train:document.schemaVersion=18;case .saveModel:document.schemaVersion=17;default:break}
            let ports = try WorkflowPortContracts.resolveForAuthoring(node, document: document, modelRegistry: modelRegistry, dataRegistry: dataRegistry)
            result.append(.init(id: id, component: .init(id: id, title: title, category: category,
                inputs: ports.inputs.filter { if case .train = payload { return !["optimizer", "scheduler", "objective", "initialization", "gradientPolicy", "trainability", "dataset", "loader"].contains($0.id) }; return true }.map(\.id), outputs: ports.outputs.map(\.id), fields: fields), iconType: iconOverride ?? definition?.resolvedAppearance?.rawValue ?? definition?.composition.map { ModuleCompositionPresentation.iconType($0.kind) } ?? icon(payload), payload: payload, definition: definition))
        }
        let recurrentCellAnchor=modelRegistry.components.first {RecurrentCellConfiguration.componentIDs.contains($0.id)}?.id
        let recurrentAnchor=modelRegistry.components.first {RecurrentSequenceConfiguration.componentIDs.contains($0.id)}?.id
        let tensorAnchors=TensorOperationPalette.consolidatedFamilies.compactMap {family in modelRegistry.components.first(where:{family.types.contains($0.id)}).map {($0.id,family)}}
        let reshapeAnchor=modelRegistry.components.first {ReshapeAuthoringConfiguration.componentIDs.contains($0.id)}?.id
        let linearAnchor=modelRegistry.components.first {LinearConfiguration.componentIDs.contains($0.id)}?.id
        let normalizationAnchor=modelRegistry.components.first {NormalizationConfiguration.componentIDs.contains($0.id)}?.id
        let convolutionAnchor=modelRegistry.components.first {ConvolutionConfiguration.componentIDs.contains($0.id)}?.id
        for component in modelRegistry.components where component.id != "transformer" && (!ComponentRibbonCatalog.replacedTypes.contains(component.id) || component.id == "max_pool2d" || component.id == convolutionAnchor || component.id == normalizationAnchor || tensorAnchors.contains(where:{$0.0==component.id}) || component.id == reshapeAnchor || component.id == linearAnchor || component.id == recurrentAnchor || component.id == recurrentCellAnchor) && !RetiredTrainingRecipeMigration.componentIDs.contains(component.id) && component.id != "index_input" {
            if component.id == recurrentCellAnchor {
                let defaults=try RecurrentCellConfiguration.defaults(for:.rnn,registry:modelRegistry)
                try append(kind:"model",key:RecurrentCellPalette.type,title:"Recurrent Cell",category:"Layers",payload:.modelPrimitive(type:"rnn_cell",configurationVersion:2,config:defaults),iconOverride:RecurrentCellPalette.type)
                continue
            }
            if component.id == recurrentAnchor {
                let defaults=try RecurrentSequenceConfiguration.defaults(for:.rnn,registry:modelRegistry)
                try append(kind:"model",key:RecurrentPalette.type,title:"Recurrent",category:"Layers",payload:.modelPrimitive(type:"rnn",configurationVersion:2,config:defaults),iconOverride:RecurrentPalette.type)
                continue
            }
            if let family=tensorAnchors.first(where:{$0.0==component.id})?.1,let type=family.types.first(where:{modelRegistry.component(type:$0) != nil}) {
                let defaults=try TensorOperationAuthoring.defaults(type,registry:modelRegistry)
                try append(kind:"model",key:TensorOperationPalette.type(family),title:family.title,category:component.category,payload:.modelPrimitive(type:type,configurationVersion:2,config:defaults),iconOverride:TensorOperationPalette.type(family))
                continue
            }
            if component.id == reshapeAnchor,let variant=ReshapePalette.defaultVariant(allowedTypes:Set(modelRegistry.components.map(\.id))) {
                let defaults=try ReshapeAuthoringConfiguration.defaults(for:variant,registry:modelRegistry)
                try append(kind:"model",key:ReshapePalette.type,title:"Reshape",category:"Tensor layout",payload:.modelPrimitive(type:variant.rawValue,configurationVersion:2,config:defaults),iconOverride:ReshapePalette.type)
                continue
            }
            if component.id == linearAnchor {
                let defaults=try LinearConfiguration.defaults(for:.explicit,registry:modelRegistry)
                try append(kind:"model",key:LinearPalette.type,title:"Linear",category:"Layers",payload:.modelPrimitive(type:"linear",configurationVersion:2,config:defaults),iconOverride:LinearPalette.type)
                continue
            } else if component.id == normalizationAnchor {
                let defaults=try NormalizationConfiguration.defaults(for:.layerNorm,registry:modelRegistry)
                try append(kind:"model",key:NormalizationPalette.type,title:"Normalization",category:"Conditioning",payload:.modelPrimitive(type:"layer_norm",configurationVersion:2,config:defaults),iconOverride:NormalizationPalette.type)
                continue
            }
            if component.id == convolutionAnchor {
                let defaults=try ConvolutionConfiguration.defaults(for:.conv2d,registry:modelRegistry)
                try append(kind:"model",key:ConvolutionPalette.type,title:"Convolution",category:"Layers",payload:.modelPrimitive(type:"conv2d",configurationVersion:2,config:defaults),iconOverride:ConvolutionPalette.type)
                continue
            }
            if component.id == "max_pool2d" {
                let defaults=try PoolingConfiguration.defaults(for:.maxPool2d,registry:modelRegistry)
                try append(kind:"model",key:PoolingPalette.type,title:"Pooling",category:"Pooling",payload:.modelPrimitive(type:"max_pool2d",configurationVersion:2,config:defaults),iconOverride:PoolingPalette.type)
                continue
            }
            let config = Dictionary(uniqueKeysWithValues: component.fields.map { ($0.key, $0.defaultValue) })
            let isGradientControl = component.id == "gradient_policy" || component.id == "trainability"
            try append(kind: component.id == "initialization" || isGradientControl ? "control":"model", key: component.id, title: component.id=="input" ? "Model input":component.id=="output" ? "Model definition":component.title, category: component.id=="input" ? "Boundaries":component.id == "dropout" ? "Conditioning":component.category,
                       fields: component.fields,
                       payload: .modelPrimitive(type: component.id, configurationVersion: 2, config: config))
        }
        if modelRegistry.component(type: "depthwise_conv2d") != nil, modelRegistry.component(type: "conv2d") != nil {
            let definition = try SeparableConv2dBlock.definition()
            try CompositeDefinitionValidation.validate(definition, definitions: [definition], registry: modelRegistry)
            try append(kind: "block", key: definition.reference.id, title: definition.title, category: "Layers",
                       fields: definition.parameters.map(\.field), payload: .modelBlock(reference: definition.reference, overrides: [:]), definition: definition)
        }
        if modelRegistry.component(type:"transformer_encoder") != nil,modelRegistry.component(type:"transformer_decoder") != nil {
            let definition=try TransformerBlock.definition()
            try CompositeDefinitionValidation.validate(definition,definitions:[definition],registry:modelRegistry)
            try append(kind:"block",key:definition.reference.id,title:"Transformer",category:"Layers",payload:.modelBlock(reference:definition.reference,overrides:[:]),definition:definition)
        }
        let activationDefaults=try ActivationConfiguration.defaults(for:.relu,registry:modelRegistry)
        try append(kind:"model",key:ActivationPalette.type,title:"Activation",category:"Activations",payload:.modelPrimitive(type:"relu",configurationVersion:2,config:activationDefaults),iconOverride:ActivationPalette.type)
        var parallel=CompositeExamples.parallelClassifier.definitions[0]
        parallel.reference = .init("kwip.parallel-features.container")
        parallel.composition = .init(kind:.graph,children:parallel.body.nodes.map {.init(localID:$0.id)})
        try append(kind:"block",key:parallel.reference.id,title:"Parallel features",category:"Merge",fields:parallel.parameters.map(\.field),payload:.modelBlock(reference:parallel.reference,overrides:[:]),definition:parallel)
        for operation in dataRegistry.operators where operation.id != DataFlowSourceOperators.capturedTable && !ComponentRibbonCatalog.replacedTypes.contains(operation.id) {
            try append(kind: "data", key: operation.id, title: operation.title, category: "Data",
                       payload: .dataOperator(operatorID: operation.id, config: operation.defaultConfig))
        }
        try append(kind: "control", key: "estimator", title: "Prebuilt models", category: "Training", payload: .estimator(config: PrebuiltModelConfiguration.defaults()))
        try append(kind: "control", key: "train", title: "Train", category: "Training", payload: .train(config: ["runSetup":.object([:]),"dataMapping":.object([:])]))
        try append(kind: "control", key: "frozenModel", title: "Load model", category: "Boundaries", payload: .frozenModel(config: [:]))
        try append(kind: "control", key: "evaluateTest", title: "Evaluate", category: "Evaluation", payload: .evaluateTest(config: [:]))
        try append(kind:"control",key:"gameSession",title:"Game session",category:"Evaluation",payload:.gameSession(config:[:]))
        try append(kind:"control",key:"generateText",title:"Generate text",category:"Evaluation",payload:.generateText(config:[:]))
        try append(kind: "control", key: "saveModel", title: "Save model", category: "Training", payload: .saveModel(config: [:]))
        return result
    }
}
