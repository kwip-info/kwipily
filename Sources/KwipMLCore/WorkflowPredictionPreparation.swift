import Foundation

/// A derived preparation captures immutable predictions as private, hash-pinned sources.
/// It never substitutes a prediction preview or follows a training edge to retrain a model.
public struct WorkflowPredictionPreparation {
    public let document:WorkflowDocument
    public let sources:[WorkflowNodeAddress:WorkflowEndpoint]
    public static func dependentData(in document:WorkflowDocument)->Set<WorkflowNodeAddress> {
        var found=Set<WorkflowNodeAddress>()
        var pending=document.edges.filter {$0.from.address.domain == .control && ["predictions","profiles","confusion","routing","routingSample","table","summary","episodes","moves","iterations","evaluation","campaign","opponents","performance"].contains($0.from.port) && $0.kind == .table}.map(\.to.address)
        while let address=pending.popLast() {
            guard address.domain == .data,found.insert(address).inserted else {continue}
            pending+=document.edges.filter {$0.from.address==address}.map(\.to.address)
        }
        return found
    }
    public init(document:WorkflowDocument,pins:[WorkflowEndpoint:String])throws {
        let downstream=Self.dependentData(in:document)
        guard !downstream.isEmpty else {throw DataFlowContractError("Connect a prediction table to a data tool first.")}
        let incoming=Dictionary(grouping:document.edges,by:{$0.to.address})
        var included=Set<WorkflowNodeAddress>(),pending=Array(downstream),endpoints=Set<WorkflowEndpoint>()
        while let address=pending.popLast() {
            guard address.domain == .data,included.insert(address).inserted else {continue}
            for edge in incoming[address] ?? [] {
                if edge.from.address.domain == .data {pending.append(edge.from.address)}
                else {
                    guard edge.kind == .table,["predictions","profiles","confusion","routing","routingSample","table","summary","episodes","moves","iterations","evaluation","campaign","opponents","performance"].contains(edge.from.port),let provider=document.nodes.first(where:{$0.address==edge.from.address}) else {throw DataFlowContractError("Only verified prediction tables may enter this preparation boundary.")}
                    switch provider.payload {case .gameSession,.train,.evaluateTest,.generateText,.estimator:break;default:throw DataFlowContractError("Prediction source must be Train or Evaluate.")}
                    guard let pin=pins[edge.from],pin.count==64,pin.allSatisfy({$0.isHexDigit && !$0.isUppercase}) else {throw DataFlowContractError("Run the connected Train/Evaluate components before preparing their predictions.")}
                    endpoints.insert(edge.from)
                }
            }
        }
        var result=document;result.nodes=document.nodes.filter {included.contains($0.address)}
        result.edges=document.edges.filter {included.contains($0.to.address)}
        result.modelInputs=nil;result.namedModelInputs=nil;result.modelSourceVersions=nil;result.guidedBuild=nil;result.definitions=[];result.drafts=[]
        let replacement=try Self.replacingCapturedInputs(in:result,endpoints:endpoints,pins:pins)
        result=replacement.document
        let sources=replacement.sources

        try WorkflowCodec.validate(result)
        self.document=result;self.sources=sources
    }
    /// Shared hash-pinned substitution used by whole-flow and selected-target preparation.
    public static func replacingCapturedInputs(in document:WorkflowDocument,endpoints:Set<WorkflowEndpoint>,pins:[WorkflowEndpoint:String])throws->(document:WorkflowDocument,sources:[WorkflowNodeAddress:WorkflowEndpoint]) {
        var result=document
        var sources:[WorkflowNodeAddress:WorkflowEndpoint]=[:]
        for endpoint in endpoints.sorted(by:{$0.address.localID==$1.address.localID ? $0.port<$1.port:$0.address.localID<$1.address.localID}) {
            guard let pin=pins[endpoint],pin.count==64,pin.allSatisfy({$0.isHexDigit && !$0.isUppercase}) else {throw WorkflowCapturedTableRequirement()}
            let id=["table","predictions"].contains(endpoint.port) ? "predictions-"+endpoint.address.localID:"captured-"+endpoint.address.localID+"-"+endpoint.port
            let address=WorkflowNodeAddress(domain:.data,localID:id)
            guard id.count<=128,!result.nodes.contains(where:{$0.address==address}) else {throw DataFlowContractError("Prediction capture source identity collides with a data component. Rename its source identifier before continuing.")}
            result.nodes.append(.init(address:address,name:"Captured predictions",payload:.dataOperator(operatorID:DataFlowSourceOperators.capturedTable,config:["sha256":.string(pin)])))
            for i in result.edges.indices where result.edges[i].from==endpoint {result.edges[i].from = .init(address:address,port:"table")}
            sources[address]=endpoint
        }
        return (result,sources)
    }
}
