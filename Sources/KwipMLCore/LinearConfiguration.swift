import Foundation

/// One authoring surface over explicit and inferred affine layers; saved primitive contracts remain unchanged.
public struct LinearConfiguration:Equatable {
    public enum Variant:String,Codable,CaseIterable {
        case explicit="linear",inferred="lazy_linear"
        public var title:String {self == .explicit ? "Set explicitly":"Infer from input"}
    }
    public static let defaultVariant:Variant = .explicit
    public static var componentIDs:Set<String> {Set(Variant.allCases.map(\.rawValue))}
    public let variant:Variant
    public let values:[String:JSONValue]
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let variant=Variant(rawValue:componentID) else {throw GraphConfigurationError("Choose an existing supported linear variant; arbitrary primitive replacement is not allowed.")}
        self.variant=variant
        var source=GraphNode(id:"linear",type:componentID,config:values)
        // GraphNode's convenience initializer supplies v2 bias; decoded v1 source
        // must instead reach the existing version-aware migration exactly as saved.
        source.config=values
        self.values=try GraphMigration.effectiveConfiguration(of:source,schemaVersion:configurationVersion,registry:registry)
    }
    public static func defaults(for variant:Variant,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:variant.rawValue) else {throw GraphConfigurationError("The selected linear variant is unavailable.")}
        return try Self(componentID:variant.rawValue,configurationVersion:2,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}),registry:registry).values
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual linear primitive.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to variant:Variant,configuration:[String:JSONValue]?=nil,connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,let previous=Variant(rawValue:type) else {throw GraphConfigurationError("Only an actual linear primitive may change linear variants.")}
        let validated:Self
        if previous==variant,configuration==nil {validated=try Self.configuration(of:node,registry:registry)}
        else {
            var values=try configuration ?? defaults(for:variant,registry:registry)
            if configuration==nil {
                let current=try Self.configuration(of:node,registry:registry)
                for key in ["out_features","bias"] {values[key]=current.values[key]}
            }
            validated=try .init(componentID:variant.rawValue,configurationVersion:2,values:values,registry:registry)
        }
        guard let component=registry.component(type:variant.rawValue),connectedOutputPorts.isSubset(of:Set(component.configuredOutputs(validated.values))) else {throw GraphConfigurationError("The selected linear variant removes an output that still has a connection. Repair the connection explicitly first.")}
        if let settings=node.parameterSettings {
            try LayerParameterSettings.validate(settings)
            let choices=try LayerParameterInventory.choices(type:variant.rawValue,config:validated.values,version:2,registry:registry)
            for (parameter,value) in settings {
                guard let choice=choices.first(where:{$0.id==parameter}) else {throw GraphConfigurationError("This change removes the authored \(parameter) initialization or trainability policy. Remove that policy explicitly before changing the linear.")}
                if case .object(let row)=value,case .object(let initialization)=row["initialization"],
                   let algorithm=initialization["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {
                    throw GraphConfigurationError("Xavier and Kaiming initialization require a multidimensional parameter. Choose a supported initializer for \(parameter) before changing linear.")
                }
            }
        }
        if previous==variant,configuration==nil {return node}
        var result=node
        result.payload = .modelPrimitive(type:variant.rawValue,configurationVersion:2,config:validated.values)
        return result
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to variant:Variant,configuration:[String:JSONValue]?=nil,connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:configurationVersion,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try switching(wrapped,to:variant,configuration:configuration,connectedOutputPorts:connectedOutputPorts,registry:registry)
        guard case .modelPrimitive(let type,_,let config)=changed.payload else {throw GraphConfigurationError("Linear replacement did not resolve an existing primitive.")}
        var result=node;result.type=type;result.config=config;result.parameterSettings=changed.parameterSettings
        return result
    }
}
