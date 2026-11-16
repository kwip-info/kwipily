import Foundation

/// An explicit data target executes only its incoming dependency closure. Model
/// resources may be consumed, but their training graphs are never executed.
public struct WorkflowScopedDataPreparation {
    public let document: WorkflowDocument
    public let target: WorkflowNodeAddress
    public let apiSources: Set<WorkflowNodeAddress>
    public let capturedSources:[WorkflowNodeAddress:WorkflowEndpoint]
    public let permissionDocument:WorkflowDocument

    public init(document: WorkflowDocument, target: WorkflowNodeAddress, registry: DataFlowOperatorRegistry,capturedPin:(WorkflowEndpoint)throws->String = {_ in throw WorkflowCapturedTableRequirement()}) throws {
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty else { throw DataFlowContractError("Apply or discard pending connections before preparing data.") }
        guard let terminal=document.nodes.first(where:{$0.address==target}),case .dataOperator=terminal.payload else {throw DataFlowContractError("Choose a data component.")}
        var included=Set<WorkflowNodeAddress>(),pending=[target],captured=Set<WorkflowEndpoint>()
        while let address=pending.popLast() {
            guard included.insert(address).inserted else {continue}
            guard let node=document.nodes.first(where:{$0.address==address}),case .dataOperator(let id,_)=node.payload else {throw DataFlowContractError("Data preparation cannot execute model or training components.")}
            guard address==target || id != "kwip.data.outputDelimited" else {throw DataFlowContractError("A data inspection cannot execute an upstream Table Output. Choose the output explicitly.")}
            for edge in document.edges where edge.to.address==address {
                if edge.from.address.domain == .data {pending.append(edge.from.address)}
                else {
                    if edge.kind == .table,edge.from.address.domain == .control,
                       ["predictions","profiles","confusion","routing","routingSample","table","summary","episodes","moves","iterations","evaluation","campaign","opponents","performance"].contains(edge.from.port),
                       let provider=document.nodes.first(where:{$0.address==edge.from.address}) {
                        switch provider.payload {case .gameSession,.train,.evaluateTest,.generateText,.estimator:captured.insert(edge.from);continue;default:break}
                    }
                    guard edge.kind == .frozenModel,edge.to.port=="model",let provider=document.nodes.first(where:{$0.address==edge.from.address}) else {throw DataFlowContractError("Data preparation requires data inputs or verified model resources.")}
                    switch provider.payload {case .frozenModel:break;case .train where id==TypedDecisionConfiguration.operatorID:break;default:throw DataFlowContractError("Data preparation cannot run an upstream model provider.")}
                    included.insert(edge.from.address)
                }
            }
        }
        var scoped=document
        scoped.nodes=document.nodes.filter{included.contains($0.address)}
        scoped.edges=document.edges.filter{(included.contains($0.from.address) || captured.contains($0.from)) && included.contains($0.to.address) && $0.to.address.domain == .data}
        scoped.modelInputs=nil;scoped.namedModelInputs=nil;scoped.modelSourceVersions=nil;scoped.guidedBuild=nil;scoped.definitions=[]
        var permissions=scoped
        // Preserve producer identities for policy checks, never their input graphs.
        let providerAddresses=Set(captured.map(\.address))
        permissions.nodes=document.nodes.filter{included.contains($0.address) || providerAddresses.contains($0.address)}
        self.permissionDocument=permissions
        let pins=try Dictionary(uniqueKeysWithValues:captured.map{($0,try capturedPin($0))})
        let replacement=try WorkflowPredictionPreparation.replacingCapturedInputs(in:scoped,endpoints:captured,pins:pins)
        scoped=replacement.document;capturedSources=replacement.sources
        _=try WorkflowDataPreparationPlan.compile(scoped,registry:registry,allowTrainingTerminals:true)
        self.document=scoped;self.target=target
        apiSources=Set(scoped.nodes.compactMap {node in if case .dataOperator(let id,let config)=node.payload,id==DataFlowSourceOperators.tableInput,config["format"] == .string("api"){return node.address};return nil})
    }
}

/// Missing, stale or omitted capture is a prerequisite failure, not permission to
/// rerun a producer or substitute its visible preview.
public struct WorkflowCapturedTableRequirement:Error,LocalizedError {
    public init() {}
    public var errorDescription:String? {"The selected data branch requires a current, complete captured table from its connected producer. No producer was rerun and no output was written."}
}
