import Foundation

/// One authoring family over the seven existing convolution primitives without changing numerical runtime identities.
public struct ConvolutionConfiguration:Equatable {
    public enum Operation:String,Codable,CaseIterable {case standard,transposed,depthwise}
    public enum Variant:String,Codable,CaseIterable {
        case conv1d,conv2d,conv3d
        case convTranspose1d="conv_transpose1d",convTranspose2d="conv_transpose2d",convTranspose3d="conv_transpose3d"
        case depthwiseConv2d="depthwise_conv2d"
        public var operation:Operation {switch self {case .convTranspose1d,.convTranspose2d,.convTranspose3d:return .transposed;case .depthwiseConv2d:return .depthwise;default:return .standard}}
        public var dimensions:Int {switch self {case .conv1d,.convTranspose1d:return 1;case .conv3d,.convTranspose3d:return 3;default:return 2}}
        public var title:String {"\(operation == .transposed ? "Transposed ":operation == .depthwise ? "Depthwise ":"")Convolution \(dimensions)D"}
    }
    public static let defaultVariant:Variant = .conv2d
    public static var componentIDs:Set<String> {Set(Variant.allCases.map(\.rawValue))}
    public let variant:Variant
    public let values:[String:JSONValue]
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let variant=Variant(rawValue:componentID) else {throw GraphConfigurationError("Choose an existing supported convolution variant; arbitrary primitive replacement is not allowed.")}
        self.variant=variant
        self.values=try GraphMigration.effectiveConfiguration(of:.init(id:"convolution",type:componentID,config:values),schemaVersion:configurationVersion,registry:registry)
    }
    public static func defaults(for variant:Variant,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:variant.rawValue) else {throw GraphConfigurationError("The selected convolution variant is unavailable.")}
        return try Self(componentID:variant.rawValue,configurationVersion:2,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}),registry:registry).values
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual convolution primitive.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to variant:Variant,configuration:[String:JSONValue]?=nil,connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,let previous=Variant(rawValue:type) else {throw GraphConfigurationError("Only an actual convolution primitive may change convolution variants.")}
        let validated:Self
        if previous==variant,configuration==nil {validated=try Self.configuration(of:node,registry:registry)}
        else {validated=try .init(componentID:variant.rawValue,configurationVersion:2,values:configuration ?? defaults(for:variant,registry:registry),registry:registry)}
        guard let component=registry.component(type:variant.rawValue),connectedOutputPorts.isSubset(of:Set(component.configuredOutputs(validated.values))) else {throw GraphConfigurationError("The selected convolution variant removes an output that still has a connection. Repair the connection explicitly first.")}
        if let settings=node.parameterSettings {
            try LayerParameterSettings.validate(settings)
            let choices=try LayerParameterInventory.choices(type:variant.rawValue,config:validated.values,version:2,registry:registry)
            for (parameter,value) in settings {
                guard let choice=choices.first(where:{$0.id==parameter}) else {throw GraphConfigurationError("This change removes the authored \(parameter) initialization or trainability policy. Remove that policy explicitly before changing the convolution.")}
                if case .object(let row)=value,case .object(let initialization)=row["initialization"],
                   let algorithm=initialization["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {
                    throw GraphConfigurationError("Xavier and Kaiming initialization require a multidimensional weight. Use a supported bias initializer before this change.")
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
        guard case .modelPrimitive(let type,_,let config)=changed.payload else {throw GraphConfigurationError("Convolution replacement did not resolve an existing primitive.")}
        var result=node;result.type=type;result.config=config;result.parameterSettings=changed.parameterSettings
        return result
    }
}
