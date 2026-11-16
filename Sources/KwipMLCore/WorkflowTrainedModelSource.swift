import Foundation

/// An artifact provider, not permission to execute or reconstruct editable training.
public enum WorkflowTrainedModelSource:Equatable {
    case savedCheckpoint(WorkflowEndpoint)
    case train(WorkflowEndpoint)
    case estimator(WorkflowEndpoint)
    public var isEstimator:Bool {if case .estimator=self {return true};return false}
    public var endpoint:WorkflowEndpoint {switch self {case .savedCheckpoint(let endpoint),.train(let endpoint),.estimator(let endpoint):return endpoint}}
    public var address:WorkflowNodeAddress {endpoint.address}
    public var isTrain:Bool {if case .train=self {return true};return false}
    public static func resolve(in document:WorkflowDocument,consumer:WorkflowNodeAddress)throws->Self {
        guard let target=document.nodes.first(where:{$0.address==consumer}) else {throw workflowFailure(.invalidSelection,"Choose an existing artifact consumer.",consumer)}
        switch target.payload {case .evaluateTest,.saveModel,.generateText:break;case .dataOperator(let id,_) where id==TypedDecisionConfiguration.operatorID:break;default:throw workflowFailure(.invalidSelection,"Choose Evaluate or Save model.",consumer)}
        let links=document.edges.filter {$0.to.address==consumer && $0.to.port=="model"}
        guard links.count==1,let edge=links.first,edge.kind == .frozenModel,edge.from.port=="model",
              let provider=document.nodes.first(where:{$0.address==edge.from.address}) else {throw workflowFailure(.requiredInput,"Connect exactly one Trained model provider.",consumer,"model")}
        switch provider.payload {
        case .frozenModel(let config):
            _ = try LoadModelConfiguration.decode(config)
            return .savedCheckpoint(edge.from)
        case .estimator(let config) where PrebuiltModelConfiguration.hasModel(config):return .estimator(edge.from)
        case .train where document.schemaVersion>=17:return .train(edge.from)
        default:throw workflowFailure(.invalidSelection,"Use Train's completed model or a Saved checkpoint bound to verified History.",consumer,"model")
        }
    }
    public static func promoteAuthoring(_ document:WorkflowDocument)throws->WorkflowDocument {
        let upgraded=try RetiredToolAuthoringUpgrade.plan(document)
        var result=try WorkflowTrainRunSetup.migrateAuthoringSchema(upgraded)
        result.schemaVersion=max(17,result.schemaVersion)
        try WorkflowCodec.validate(result)
        return result
    }
}
