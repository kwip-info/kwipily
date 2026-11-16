import Foundation

extension AttentionConfiguration {
    public static func applying(_ values:[String:JSONValue],to node:WorkflowNode,connectedInputPorts:Set<String>=[],connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,let version,_)=node.payload,componentIDs.contains(type) else {throw GraphConfigurationError("Choose an existing Attention or Transformer component.")}
        let checked=try Self(componentID:type,values:values)
        let canonical=try GraphMigration.effectiveConfiguration(of:.init(type:type,config:checked.values),schemaVersion:2,registry:registry)
        guard connectedInputPorts.isSubset(of:Set(checked.configuredInputs)),connectedOutputPorts.isSubset(of:Set(checked.outputPorts)) else {
            throw GraphConfigurationError("Disconnect connected or exposed masks and attention weights before removing their ports.")
        }
        if let settings=node.parameterSettings {
            try LayerParameterSettings.validate(settings)
            for (name,value) in settings {
                guard let parameter=checked.parameterShapes.first(where:{$0.name==name}) else {throw GraphConfigurationError("This change removes an authored weight or bias policy. Remove that policy explicitly before changing the layer.")}
                if case .object(let row)=value,case .object(let start)=row["initialization"],let algorithm=start["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),parameter.shape.count<2 {throw GraphConfigurationError("Xavier and Kaiming require multidimensional weights.")}
            }
        }
        var next=node;next.payload = .modelPrimitive(type:type,configurationVersion:version,config:canonical);return next
    }
    public static func applying(_ values:[String:JSONValue],to node:GraphNode,connectedInputPorts:Set<String>=[],connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:2,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try applying(values,to:wrapped,connectedInputPorts:connectedInputPorts,connectedOutputPorts:connectedOutputPorts,registry:registry)
        guard case .modelPrimitive(_,_,let config)=changed.payload else {throw GraphConfigurationError("Attention settings did not resolve.")}
        var next=node;next.config=config;return next
    }
}
