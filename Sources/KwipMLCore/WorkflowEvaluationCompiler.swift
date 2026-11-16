import Foundation

/// Declared graph lineage, not a measured archive or permission to execute data sources.
public struct WorkflowEvaluationLineage:Equatable {
    public let source:WorkflowNodeAddress
    public let split:WorkflowNodeAddress
    public let role:String
    public let states:[WorkflowNodeAddress]
}
public enum WorkflowEvaluationMembershipValidation:String,Codable {
    /// The selected archived port must contain exactly the originating protected
    /// split's observation IDs. A filter/sample declaration does not prove this.
    case exactArchivedProtectedPopulation
}
/// Resolves private frozen-model binding and archived table identifiers only.
/// It deliberately contains no executable preparation or editable model plan.
public struct WorkflowEvaluationPlan:Equatable {
    public let projectID:String
    public let revision:UInt64
    public let evaluate:WorkflowNodeAddress
    public let frozenModel:WorkflowNodeAddress
    public let testEndpoint:DataFlowEndpoint
    public let canonicalTestEndpoint:DataFlowEndpoint
    public let testLineage:WorkflowEvaluationLineage
    public let membershipValidation:WorkflowEvaluationMembershipValidation
    public let settings:WorkflowEvaluationSettings
    public var capturedModelSource:WorkflowTrainedModelSource? = nil
    public var modelSource:WorkflowTrainedModelSource {capturedModelSource ?? .savedCheckpoint(.init(address:frozenModel,port:"model"))}
    public var isTrain:Bool {modelSource.isTrain}
}
public enum WorkflowEvaluationCompiler {
    public static func compile(_ document:WorkflowDocument,evaluate:WorkflowNodeAddress,
                               modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)throws->WorkflowEvaluationPlan {
        do {try WorkflowCodec.validate(document)} catch {throw workflowFailure(.invalidDocument,error.localizedDescription)}
        guard document.schemaVersion>=5 else {throw workflowFailure(.configuration,"Final test evaluation requires workflow schema 5.",evaluate)}
        guard document.drafts.isEmpty else {throw workflowFailure(.unresolvedDraft,"Apply or discard unresolved Block drafts before compiling.")}
        guard let terminal=document.nodes.first(where:{$0.address==evaluate}),case .evaluateTest(let config)=terminal.payload else {
            throw workflowFailure(.invalidSelection,"Select an Evaluate terminal.",evaluate)
        }
        let settings=try WorkflowEvaluationSettings.decode(config)
        let modelSource=try WorkflowTrainedModelSource.resolve(in:document,consumer:evaluate)
        let region=try WorkflowCompiler.dependencies(document,terminal:evaluate,modelRegistry:modelRegistry,dataRegistry:dataRegistry,artifactSources:modelSource.isTrain ? [modelSource.address]:[])
        func input(_ port:String)throws->WorkflowEndpoint {
            guard let edge=region.incoming[evaluate]?.first(where:{$0.to.port==port}) else {throw workflowFailure(.requiredInput,"Connect required evaluation input.",evaluate,port)}
            return edge.from
        }
        let frozen=try input("model"),test=try input("test")
        let evidence=try WorkflowDataLineage.resolve(region,dataRegistry:dataRegistry)
        guard case .rows(let rows)=evidence[test],rows.role==settings.partitionRole,let split=rows.split,
              let partition=region.nodes[split],case .dataOperator(let operatorID,_)=partition.payload,
              ["kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.partition"].contains(operatorID) else {
            throw workflowFailure(.lineage,"Connect the selected Validation or Test population from its declared Partition.",evaluate,"test")
        }
        return .init(projectID:document.projectID,revision:document.revision,evaluate:evaluate,frozenModel:frozen.address,
                     testEndpoint:.init(node:test.address.localID,port:test.port),canonicalTestEndpoint:.init(node:split.localID,port:settings.partitionRole),
                     testLineage:.init(source:rows.source,split:split,role:rows.role,states:rows.states),
                     membershipValidation:.exactArchivedProtectedPopulation,settings:settings,capturedModelSource:modelSource)
    }
}
