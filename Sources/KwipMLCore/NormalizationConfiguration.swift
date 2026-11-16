import Foundation

/// One authoring family over the supported normalization primitives; running statistics remain checkpoint-owned.
public struct NormalizationConfiguration:Equatable {
    public enum Method:String,Codable,CaseIterable {case batch,instance,layer,group,rms}
    public enum Variant:String,Codable,CaseIterable {
        case rmsNorm="rms_norm",layerNorm="layer_norm",batchNorm1d="batch_norm1d",batchNorm2d="batch_norm2d"
        case instanceNorm1d="instance_norm1d",instanceNorm2d="instance_norm2d",groupNorm="group_norm"
        public var method:Method {switch self {case .rmsNorm:return .rms;case .batchNorm1d,.batchNorm2d:return .batch;case .instanceNorm1d,.instanceNorm2d:return .instance;case .layerNorm:return .layer;case .groupNorm:return .group}}
        public var dimensions:Int? {switch self {case .batchNorm1d,.instanceNorm1d:return 1;case .batchNorm2d,.instanceNorm2d:return 2;default:return nil}}
        public var title:String {switch self {case .rmsNorm:return "RMS Normalization";case .layerNorm:return "Layer Normalization";case .groupNorm:return "Group Normalization";default:return "\(method == .batch ? "Batch":"Instance") Normalization \(dimensions!)D"}}
    }
    public static let defaultVariant:Variant = .layerNorm
    public static var componentIDs:Set<String> {Set(Variant.allCases.map(\.rawValue))}
    public let variant:Variant
    public let values:[String:JSONValue]
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let variant=Variant(rawValue:componentID) else {throw GraphConfigurationError("Choose an existing supported normalization variant; arbitrary primitive replacement is not allowed.")}
        self.variant=variant
        self.values=try GraphMigration.effectiveConfiguration(of:.init(id:"normalization",type:componentID,config:values),schemaVersion:configurationVersion,registry:registry)
    }
    public static func defaults(for variant:Variant,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:variant.rawValue) else {throw GraphConfigurationError("The selected normalization variant is unavailable.")}
        return try Self(componentID:variant.rawValue,configurationVersion:2,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}),registry:registry).values
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual normalization primitive.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to variant:Variant,configuration:[String:JSONValue]?=nil,connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,let previous=Variant(rawValue:type) else {throw GraphConfigurationError("Only an actual normalization primitive may change normalization variants.")}
        let validated:Self
        if previous==variant,configuration==nil {validated=try Self.configuration(of:node,registry:registry)}
        else {validated=try .init(componentID:variant.rawValue,configurationVersion:2,values:configuration ?? defaults(for:variant,registry:registry),registry:registry)}
        guard let component=registry.component(type:variant.rawValue),connectedOutputPorts.isSubset(of:Set(component.configuredOutputs(validated.values))) else {throw GraphConfigurationError("The selected normalization variant removes an output that still has a connection. Repair the connection explicitly first.")}
        if let settings=node.parameterSettings {
            try LayerParameterSettings.validate(settings)
            let choices=try LayerParameterInventory.choices(type:variant.rawValue,config:validated.values,version:2,registry:registry)
            for (parameter,value) in settings {
                guard let choice=choices.first(where:{$0.id==parameter}) else {throw GraphConfigurationError("This change removes the authored \(parameter) initialization or trainability policy. Remove that policy explicitly before changing the normalization.")}
                if case .object(let row)=value,case .object(let initialization)=row["initialization"],
                   let algorithm=initialization["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {
                    throw GraphConfigurationError("Xavier and Kaiming initialization require a multidimensional parameter. Choose a supported initializer for \(parameter) before changing normalization.")
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
        guard case .modelPrimitive(let type,_,let config)=changed.payload else {throw GraphConfigurationError("Normalization replacement did not resolve an existing primitive.")}
        var result=node;result.type=type;result.config=config;result.parameterSettings=changed.parameterSettings
        return result
    }
}
