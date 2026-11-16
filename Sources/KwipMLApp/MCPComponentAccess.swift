import KwipMLCore

/// Shared identity projection for current and captured documents. Never includes configuration values.
enum MCPComponentAccess {
    static func identifiers(_ node:WorkflowNode)->Set<String> {
        var values:Set<String>=[WorkflowCanvasAdapter.icon(node.payload)]
        if case .estimator=node.payload {values.insert("workflow.logisticRegression")}
        if case .modelPrimitive(let type,_,_)=node.payload {values.formUnion(Self.families(type))}

        return values
    }
    static func families(_ type:String)->Set<String> {
        var result:Set<String>=[]
        if ActivationConfiguration.componentIDs.contains(type){result.insert(ActivationPalette.type)}
        if PoolingConfiguration.componentIDs.contains(type){result.insert(PoolingPalette.type)}
        if ConvolutionConfiguration.componentIDs.contains(type){result.insert(ConvolutionPalette.type)}
        if NormalizationConfiguration.componentIDs.contains(type){result.insert(NormalizationPalette.type)}
        if LinearConfiguration.componentIDs.contains(type){result.insert(LinearPalette.type)}
        if ReshapeAuthoringConfiguration.componentIDs.contains(type){result.insert(ReshapePalette.type)}
        if RecurrentSequenceConfiguration.componentIDs.contains(type){result.insert(RecurrentPalette.type)}
        if RecurrentCellConfiguration.componentIDs.contains(type){result.insert(RecurrentCellPalette.type)}
        for family in TensorOperationPalette.consolidatedFamilies where family.types.contains(type){result.insert(TensorOperationPalette.type(family))}
        return result
    }
    static func identifiers(_ document:WorkflowDocument)->Set<String> {
        var result=document.nodes.reduce(into:Set<String>()){$0.formUnion(identifiers($1))}
        for root in document.nodes where WorkflowMixtureContainer.contains(document,root:root.address) {
            for chain in (try? WorkflowMixtureContainer.chains(in:document,root:root.address)) ?? [] {
                for node in chain.graph.nodes {
                    result.formUnion(identifiers(.init(address:.init(domain:.model,localID:node.id),payload:.modelPrimitive(type:node.type,configurationVersion:2,config:node.config))))
                }
            }
        }
        for definition in document.definitions {
            if let appearance=definition.resolvedAppearance{result.insert(appearance.rawValue)}
            else if let composition=definition.composition{result.insert(ModuleCompositionPresentation.iconType(composition.kind))}
            for node in definition.body.nodes{result.formUnion(identifiers(.init(address:.init(domain:.model,localID:node.id),payload:.modelPrimitive(type:node.type,configurationVersion:2,config:node.config))))}
        }
        return result
    }
}
