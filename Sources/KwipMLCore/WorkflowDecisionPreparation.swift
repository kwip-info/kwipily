import Foundation

/// Executes a bounded Decide branch using a pinned model resource. Model-provider
/// inputs are deliberately excluded: preparing decisions never repeats training.
public struct WorkflowDecisionPreparation {
    public let document:WorkflowDocument
    public static func consumer(in document:WorkflowDocument,for selected:WorkflowNodeAddress)->WorkflowNodeAddress? {
        var pending=[selected],seen=Set<WorkflowNodeAddress>(),found=Set<WorkflowNodeAddress>()
        while let address=pending.popLast() {
            guard address.domain == .data,seen.insert(address).inserted else {continue}
            if let node=document.nodes.first(where:{$0.address==address}),case .dataOperator(let id,_)=node.payload,id==TypedDecisionConfiguration.operatorID {found.insert(address)}
            pending += document.edges.filter {$0.to.address==address && $0.kind != .frozenModel}.map(\.from.address)
        }
        return found.count==1 ? found.first:nil
    }
    public init(document:WorkflowDocument,consumer:WorkflowNodeAddress)throws {
        guard let node=document.nodes.first(where:{$0.address==consumer}),case .dataOperator(let id,_)=node.payload,id==TypedDecisionConfiguration.operatorID else {throw DataFlowContractError("Choose a Decide component.")}
        _ = try WorkflowTrainedModelSource.resolve(in:document,consumer:consumer)
        var downstream=Set<WorkflowNodeAddress>(),pending=[consumer]
        while let address=pending.popLast() {
            guard address.domain == .data,downstream.insert(address).inserted else {continue}
            pending += document.edges.filter {$0.from.address==address && $0.to.address.domain == .data}.map(\.to.address)
        }
        var included=Set<WorkflowNodeAddress>();pending=Array(downstream)
        while let address=pending.popLast() {
            guard included.insert(address).inserted else {continue}
            guard address.domain == .data else {continue}
            for edge in document.edges where edge.to.address==address {
                if edge.from.address.domain == .data {pending.append(edge.from.address)}
                else {
                    guard edge.kind == .frozenModel,edge.to.port=="model",let target=document.nodes.first(where:{$0.address==address}),case .dataOperator(let id,_)=target.payload,id==TypedDecisionConfiguration.operatorID else {throw DataFlowContractError("Decision preparation requires data inputs and verified model resources only.")}
                    _ = try WorkflowTrainedModelSource.resolve(in:document,consumer:address)
                    included.insert(edge.from.address)
                }
            }
        }
        var result=document
        result.nodes=document.nodes.filter {included.contains($0.address)}
        result.edges=document.edges.filter {included.contains($0.to.address) && $0.to.address.domain == .data && included.contains($0.from.address)}
        result.modelInputs=nil;result.namedModelInputs=nil;result.modelSourceVersions=nil;result.guidedBuild=nil;result.definitions=[];result.drafts=[]
        try WorkflowCodec.validate(result)
        self.document=result
    }
}
