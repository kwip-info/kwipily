import Foundation

/// Shared authoring surface; underlying transform identities and batch-safe math remain unchanged.
public enum ReshapeAuthoringConfiguration {
    public enum Variant:String,CaseIterable,Codable {
        case dimensions="reshape",flatten="flatten"
        public var title:String {self == .dimensions ? "Set dimensions":"Flatten axes"}
    }
    public static let componentIDs:Set<String>=["reshape","flatten"]
    public static func defaults(for variant:Variant,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:variant.rawValue) else {throw GraphConfigurationError("The selected transform is unavailable.")}
        return try TensorTransformConfiguration(operation:variant.rawValue,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)})).values
    }
    public static func outputLayout(variant:Variant,configuration:[String:JSONValue],input:BatchedTensorLayout)throws->BatchedTensorLayout {
        try TensorTransformConfiguration(operation:variant.rawValue,values:configuration).outputLayout(for:input)
    }
    public static func inputLayout(nodeID:String,graph:GraphDocument,validation:ValidationResult)->BatchedTensorLayout? {
        let edges=graph.edges.filter {$0.target==nodeID && $0.targetPort=="in"}
        guard edges.count==1,let edge=edges.first else {return nil}
        return validation.outputTensorContracts[.init(edge.source,edge.sourcePort)]?.layout
    }
    public static func switching(_ node:WorkflowNode,to variant:Variant,configuration:[String:JSONValue]?=nil,registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,let version,let current)=node.payload,componentIDs.contains(type) else {throw GraphConfigurationError("Choose an existing Reshape or Flatten component.")}
        guard registry.component(type:variant.rawValue) != nil else {throw GraphConfigurationError("The selected transform is unavailable in this editor.")}
        let config=try configuration ?? (type==variant.rawValue ? GraphMigration.effectiveConfiguration(of:.init(type:type,config:current),schemaVersion:version,registry:registry):defaults(for:variant,registry:registry))
        let validated=try TensorTransformConfiguration(operation:variant.rawValue,values:config).values
        guard node.parameterSettings?.isEmpty != false else {throw GraphConfigurationError("Tensor transforms have no learned parameters. Repair the unexpected parameter policy before changing method.")}
        if type==variant.rawValue,configuration==nil {return node}
        var next=node;next.payload = .modelPrimitive(type:variant.rawValue,configurationVersion:type==variant.rawValue ? version:2,config:validated);return next
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to variant:Variant,configuration:[String:JSONValue]?=nil,registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:configurationVersion,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try switching(wrapped,to:variant,configuration:configuration,registry:registry)
        guard case .modelPrimitive(let type,_,let config)=changed.payload else {throw GraphConfigurationError("Transform settings did not resolve.")}
        var next=node;next.type=type;next.config=config;return next
    }
}
