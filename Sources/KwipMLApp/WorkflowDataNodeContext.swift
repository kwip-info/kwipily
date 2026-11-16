import Foundation
import Combine
import KwipMLCore

/// One selected data node, projected directly from the authoritative mixed workflow.
/// Stable view identity does not grant a stale callback permission to edit a new state.
@MainActor final class WorkflowDataNodeContext: ObservableObject, DataNodeEditing {
    let owner: WorkflowAuthoringStore
    let address: WorkflowNodeAddress
    private let sources: any WorkflowDataSources
    private let outputProvider:(any WorkflowDataOutputs)?
    let authority:WorkflowNodeEditingAuthority
    var contextIdentity:String {authority.contextIdentity}
    private var observations:[AnyCancellable]=[]
    private var token:WorkflowEditorLease {authority.token}
    private var message:String? {get {authority.message} set {authority.message=newValue}}

    init(owner: WorkflowAuthoringStore, address: WorkflowNodeAddress, sources: any WorkflowDataSources,outputs:(any WorkflowDataOutputs)?=nil) throws {
        self.owner=owner;self.address=address;self.sources=sources;self.outputProvider=outputs
        authority=try WorkflowNodeEditingAuthority(owner:owner,address:address)
        guard sources.projectID == owner.projectID, sources.sessionID == owner.sessionID,
              outputs == nil || (outputs?.projectID == owner.projectID && outputs?.sessionID == owner.sessionID),
              owner.isCurrent(token), address.domain == .data,
              owner.selection.nodeAddresses == [address], owner.selection.edgeIDs.isEmpty,
              owner.document.nodes.contains(where: { node in
                  guard node.address == address else {return false}
                  if case .dataOperator = node.payload {return true};return false
              }) else {throw DataFlowContractError("Select a data node in the active workflow and its matching source session.")}
        observations.append(authority.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()})
        observations.append(sources.changes.sink {[weak self] _ in self?.objectWillChange.send()})
        if let outputs {observations.append(outputs.changes.sink {[weak self] _ in self?.objectWillChange.send()})}
    }
    var decisionAppearance:AppAppearanceState {(outputProvider as? WorkflowWorkspaceSession)?.configurationAppearance ?? AppAppearanceState()}
    var hasPendingDecisionDraft:Bool {owner.hasPendingInspectorEdits}
    var connectedDecisionQuestions:[String:JSONValue]? {
        guard let host=outputProvider as? WorkflowWorkspaceSession,let edge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="model"}),let value=host.modelLoader.reference(edge.from.address)?.metadata["questionSet"],case .object(let q)=value else{return nil};return q
    }
    var connectedLearnedPreparation:[String:JSONValue]? {
        guard isCurrent,let host=outputProvider as? WorkflowWorkspaceSession,let edge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="model"}),edge.kind == .frozenModel,let ref=host.modelLoader.reference(edge.from.address),ref.hasValidPins,case .object(let summary)=ref.metadata["learnedPreparation"] else{return nil}
        return summary
    }
    var decisionQuestionResources:[(String,[String:JSONValue])] {DecisionQuestionResource.choices(owner.document)}
    func makeGameRuleValidationSession()->GameRuleValidationSession? {(outputProvider as? WorkflowWorkspaceSession)?.makeGameRuleValidationSession()}
    var isCurrent: Bool {
        authority.isCurrent && sources.projectID == owner.projectID && sources.sessionID == owner.sessionID
            && owner.selection.nodeAddresses == [address] && owner.selection.edgeIDs.isEmpty
    }
    var isBusy: Bool { !isCurrent || owner.isBusy || sources.isBusy || !owner.document.drafts.isEmpty }
    var node: DataFlowNode? {
        guard isCurrent, let item=owner.document.nodes.first(where:{$0.address == address}),
              case .dataOperator(let id,let config)=item.payload else {return nil}
        return .init(id:address.localID,operatorID:id,config:config)
    }
    func inputSourceName(forInput port:String)->String? {
        guard isCurrent, let edge=owner.document.edges.first(where:{$0.to.address == address && $0.to.port == port}), let node=owner.document.nodes.first(where:{$0.address == edge.from.address}) else {return nil}
        if !node.name.isEmpty {return node.name}
        if case .dataOperator(let id,_)=node.payload {return owner.dataRegistry.operatorDefinition(id:id)?.title ?? id}
        return "Connected component"
    }
    var authoredName: String { isCurrent ? owner.document.nodes.first(where:{$0.address == address})?.name ?? "" : "" }
    var operatorTitle: String { node.map {owner.dataRegistry.operatorDefinition(id:$0.operatorID)?.title ?? $0.operatorID} ?? "Unavailable node" }
    var operatorDefinition: DataFlowOperatorDefinition? { node.flatMap { owner.dataRegistry.operatorDefinition(id: $0.operatorID) } }
    var statusMessage: String { message ?? sources.statusMessage }
    var upstreamSchema: [DataColumnSchema] { isCurrent ? sources.recordedSchema(for:address,in:owner.document) : [] }
    func upstreamSchema(forInput port:String)->[DataColumnSchema] { isCurrent ? sources.recordedSchema(for:address,in:owner.document,inputPort:port):[] }
    var schemaDisclosure:String {
        guard isCurrent else {return "Select a component in the current canvas."}
        let inputs=owner.document.edges.filter {$0.to.address == address && $0.from.address.domain == .data}
        let details=inputs.compactMap {edge -> String? in
            guard let capture=outputProvider?.snapshot(for:edge.from.address),let value=capture.ports[edge.from.port],DataObservedSchemaProjection.columns(value,isCurrent:true) != nil else {return nil}
            let name=owner.document.nodes.first {$0.address == edge.from.address}?.name ?? edge.from.address.localID
            let current=capture.matchesCurrentConstruction && capture.matchesCurrentBindings
            let status=capture.phase == .completed ? "completed":"partial · \(capture.phase.rawValue)"
            let host=outputProvider as? WorkflowWorkspaceSession
            let ledger=host?.currentDataPreparation?.observations ?? host?.execution?.preparationObservations
            let observed=ledger.flatMap {$0.observation(at:.init(node:edge.from.address.localID,port:edge.from.port),current:$0.identity)}
            let origins=observed?.columns.prefix(6).map {"\($0.schema.name) (\($0.schema.dtype)) ← \($0.origin.nodeID) · \($0.origin.port) · \($0.origin.columnName)"}.joined(separator:"; ") ?? ""

            return "\(name) · \(edge.from.port) → \(edge.to.port) · observed \(status) · Run \(capture.runID.uuidString.prefix(8)) · revision \(capture.capturedRevision) · \(current ? "current recorded snapshot":"stale; choices use current declarations")"+(origins.isEmpty ? "":"\n"+origins)
        }
        return (details.isEmpty ? "Projected column guidance from configuration and recorded source headers; no observed run at this input.":details.joined(separator:"\n"))+"\nExternal files are not reread while editing. Refresh a source to recheck its contents."
    }
    var usesFixtures: Bool {false}
    var fixtureColumns: [DataColumnSchema] {[]}
    var sourceDatabaseProjectID: String? { isCurrent ? owner.projectID : nil }
    @discardableResult func connectDatabaseSource() -> Bool {
        guard isCurrent, !isBusy else { return false }
        return sources.connectDatabaseSource(for: address, lease: token)
    }
    var sourceDisplayName: String? { isCurrent ? sources.bindingDisplayName(for:address) : nil }
    var sourceIsLinked: Bool {sourceDisplayName != nil}
    var apiFetchReceipt:[String:JSONValue]? {sources.apiFetchReceipt(for:address)}
    var sourceInspectionPreview: SourceInspectionResult? { isCurrent ? sources.sourceInspectionPreview(for: address) : nil }
    var sourceInspectionStatus: SourceInspectionStatus? { isCurrent ? sources.sourceInspectionStatus(for: address) : nil }
    var sourcePermissionSummary: String {
        guard sourceIsLinked else { return "No local source permission selected." }
        return sourceInspectionPreview == nil
            ? "Local source linked. Refresh to verify access and inspect its contents."
            : "Read access succeeded for this recorded snapshot. External changes or revoked access require Refresh; no source reads occur while editing."
    }
    var databaseOutputs:DatabaseOutputStore? {sources.databaseOutputs}
    var outputDestinationSummary: String {
        guard isCurrent, node?.operatorID == "kwip.data.outputDelimited" else { return "Select a table output component to choose its destination." }
        return sources.outputDestinationSummary(for: address)
    }
    @discardableResult func chooseOutputDestination() -> Bool {
        guard !isBusy, node?.operatorID == "kwip.data.outputDelimited" else { return false }
        message = nil
        return sources.chooseOutputDestination(for: address, lease: token)
    }
    var sourceSeparatorValues: [String] { [",", "\t", ";", "|"] }
    private var capturedOutput:WorkflowCapturedDataOutput? {
        guard isCurrent,let outputProvider,outputProvider.projectID == owner.projectID,
              outputProvider.sessionID == owner.sessionID,
              let snapshot=outputProvider.snapshot(for:address),snapshot.projectID == owner.projectID,
              snapshot.sessionID == owner.sessionID,snapshot.address == address,address.domain == .data else {return nil}
        return snapshot
    }
    var outputIsCurrent: Bool { capturedOutput.map { $0.matchesCurrentConstruction && $0.matchesCurrentBindings } ?? false }
    var outputs:[String:Any]? {capturedOutput?.renderedPorts}
    var imageArtifactResolver: FileImageArtifactResolver? {
        guard isCurrent, capturedOutput != nil else { return nil }
        return outputProvider?.imageArtifactResolver(for: address)
    }
    var fullTableViewSession: TableViewQuerySession? {
        guard isCurrent else { return nil }
        return outputProvider?.fullTableViewSession(for: address)
    }
    func capturedTableViewSession(port:String) -> TableViewQuerySession? {
        guard isCurrent else {return nil}
        return outputProvider?.capturedTableViewSession(for:.init(address:address,port:port))
    }
    var outputPopulationLabel: String? { capturedOutput?.dataCompleteness }
    var outputCapturedAt: Date? { capturedOutput?.capturedAt }
    var sourcePreviewCompletedAt: Date? { isCurrent ? sources.sourceInspectionCompletedAt(for: address) : nil }
    var outputStatus:String {
        guard let capture=capturedOutput else {return "Not evaluated"}
        var status:String
        switch capture.phase {
        case .completed:status=capture.dataCompleteness ?? "Captured preparation · workflow completed"
        case .failed:status="Captured partial observations · workflow failed"
        case .cancelled:status="Captured partial observations · workflow cancelled"
        default:status="Captured partial observations · \(capture.phase.rawValue)"
        }
        if !capture.matchesCurrentConstruction || !capture.matchesCurrentBindings {
            var reasons:[String]=[]
            if !capture.matchesCurrentConstruction {reasons.append("workflow changed")}
            if !capture.matchesCurrentBindings {reasons.append("source bindings changed")}
            status += " · Stale: "+reasons.joined(separator:", ")
        }
        return status+" · sources not revalidated"
    }
    var outputDisclosure:String? {
        guard let capture=capturedOutput else {return nil}
        return "Sampled preview from run \(capture.runID.uuidString). Request \(capture.requestIdentity). Captured source revision \(capture.capturedRevision), private binding revision \(capture.bindingRevision). Displayed values are a bounded sample, not the full dataset. Layout and label edits preserve this capture; configuration or connection changes mark it stale."
    }
    var emptyOutputMessage:String {
        let hasExecutionTerminal=owner.document.nodes.contains {node in
            if case .train=node.payload {return true}
            if case .estimator=node.payload {return true}
            return false
        }
        if !hasExecutionTerminal {
            return "Choose Run Flow to capture outputs from the connected data components. No values are inferred from the current diagram."
        }
        return "Run the connected training or estimator flow to capture preparation outputs. No values are inferred from the current diagram."
    }
    var retainedRows:String? {nil}

    private func accepts(_ expected:[String:JSONValue])->Bool {!isBusy && authority.accepts(expected)}
    private func apply(_ command:WorkflowCommand)->Bool {!isBusy && authority.apply(command)}
    @discardableResult func configure(_ config:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard accepts(expected),let node else {return false}
        guard !RetiredDataOperatorMigration.retired.contains(node.operatorID) else {message="Retired tool settings require exact project upgrade or explicit repair.";return false}
        return apply(.configure(address:address,config:config))
    }
    @discardableResult func rename(_ name:String)->Bool {apply(.rename(address:address,name:String(name.prefix(80))))}
    @discardableResult func applyTableConfiguration(_ config:[String:JSONValue],expected:[String:JSONValue])->Bool {
        guard accepts(expected),let node else {return false}
        guard !RetiredDataOperatorMigration.retired.contains(node.operatorID) else {message="Retired tool settings require exact project upgrade or explicit repair.";return false}
        do {
            if node.operatorID==SequenceWindowsConfiguration.operatorID {
                try SequenceWindowsConfiguration.validateConnectedPorts(config,inputs:Set(owner.document.edges.filter {$0.to.address==address}.map { $0.to.port }),outputs:Set(owner.document.edges.filter {$0.from.address==address}.map { $0.from.port }))
            }
            let checked: [String: JSONValue]
            if node.operatorID == "kwip.data.union" {
                let ports=try UnionConfiguration(config).inputs
                let connected=Set(owner.document.edges.filter {$0.to.address == address}.map { $0.to.port })
                checked=try UnionTableConfiguration.validate(config,schemas:Dictionary(uniqueKeysWithValues:ports.map {($0,upstreamSchema(forInput:$0))}),connected:connected)
            } else {
                checked = try DataTableConfiguration.validate(config,operatorID:node.operatorID,schema:upstreamSchema,leftSchema:upstreamSchema(forInput:"left"),rightSchema:upstreamSchema(forInput:"right"),definition:operatorDefinition)
            }
            return apply(.configure(address:address,config:checked))
        } catch {message=error.localizedDescription;return false}
    }
    func numericOrigin(key:String)->String {authority.numericOrigin(key:key)}
    func currentNumericOrigin(key:String)->String? {!isBusy ? numericOrigin(key:key) : nil}
    func setNumericPending(origin:String,key:String,pending:Bool) {authority.setNumericPending(origin:origin,key:key,pending:pending)}
    func formulaSampleSession(configuration:[String:JSONValue])->TableViewQuerySession? {
        guard isCurrent,!isBusy else {return nil}
        return outputProvider?.formulaSampleSession(for:address,configuration:configuration)
    }
    func retire() {authority.retire();observations.removeAll()}
    private var permitsSourceOperation:Bool {!isBusy && DataFlowSourceOperators.contains(node?.operatorID ?? "")}
    @discardableResult func chooseSource()->Bool {
        guard permitsSourceOperation else {return false}
        message=nil
        return sources.chooseSource(for:address,lease:token)
    }
    @discardableResult func refreshSource()->Bool {
        guard permitsSourceOperation else {return false}
        message=nil
        return sources.inspectSource(for:address,lease:token)
    }
    @discardableResult func setSeparator(_ value:String,expected:[String:JSONValue])->Bool {
        guard permitsSourceOperation,sourceSeparatorValues.contains(value),accepts(expected) else {return false}
        var next=expected;next["separator"] = .string(value)
        guard configure(next,expected:expected) else {return false}
        // Source services receive the renewed lease only after canonical persistence.
        message=nil
        if !sources.inspectSource(for:address,lease:token) {message="Separator saved. " + sources.statusMessage}
        return true
    }
}
