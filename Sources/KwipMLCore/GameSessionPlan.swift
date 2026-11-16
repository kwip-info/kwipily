import Foundation

/// Pure binding compilation. Captured model files remain the native host's authority.
public struct GameSessionPlan {
    public let address:WorkflowNodeAddress
    public let configuration:GameSessionConfiguration
    public let definition:[String:JSONValue]
    public let canvasPolicies:[String:ModelExecutionSnapshot]
    public let initializations:[String:JSONValue]
    public let savedPolicies:[String:WorkflowNodeAddress]
    public static func compile(_ document:WorkflowDocument,address:WorkflowNodeAddress,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)throws->Self {
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty,let node=document.nodes.first(where:{$0.address==address}),case .gameSession(let raw)=node.payload else {throw DataFlowContractError("Select a Game session with no unresolved drafts.")}
        let config=try GameSessionConfiguration(raw)
        _=try WorkflowCompiler.dependencies(document,terminal:address,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        func source(_ port:String)throws->WorkflowNode {
            let links=document.edges.filter{$0.to.address==address && $0.to.port==port}
            guard links.count==1,let link=links.first,let node=document.nodes.first(where:{$0.address==link.from.address}) else {throw DataFlowContractError("Connect exactly one \(port) input.")}
            return node
        }
        let rules=try source("definition")
        guard case .dataOperator(GameRuleConfiguration.operatorID,let fields)=rules.payload,case .object(let definition)=fields["definition"],document.edges.contains(where:{$0.from.address==rules.address && $0.from.port=="definition" && $0.to.address==address && $0.to.port=="definition"}) else {throw DataFlowContractError("Definition requires the typed Game definition resource, not an arbitrary table.")}
        try GameRuleConfiguration.validate(fields)
        var canvas:[String:ModelExecutionSnapshot]=[:],starts:[String:JSONValue]=[:],saved:[String:WorkflowNodeAddress]=[:]
        for role in ["A","B"] {
            switch config.source(role) {
            case "canvas":
                let source=try source("policy"+role)
                let model=try WorkflowCompiler.modelSnapshot(document,definition:source.address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,context:.frozenPolicy)
                canvas[role]=model
                let recipes=try LayerParameterSettings.recipes(document:document,model:model,registry:modelRegistry,seed:Int(config.policy(role)["initializationSeed"]?.integerValue ?? 17),initialization:nil,trainability:nil,train:address)
                starts[role]=recipes.0?.configuration.recipeValue
            case "saved":
                let source=try source("policy"+role)
                switch source.payload {case .train:saved[role]=source.address;case .frozenModel(let c):guard c.isEmpty else{throw DataFlowContractError("Imported task models require a separate game adapter.")};saved[role]=source.address;default:throw DataFlowContractError("Saved policy requires a verified canvas training checkpoint; imported task models need a separate adapter.")}
            default:break
            }
        }
        return .init(address:address,configuration:config,definition:definition,canvasPolicies:canvas,initializations:starts,savedPolicies:saved)
    }
}
