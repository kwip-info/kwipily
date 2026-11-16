import Foundation

/// One authoring family over the eight existing, independently validated pooling primitives.
public struct PoolingConfiguration:Equatable {
    public enum Method:String,Codable,CaseIterable {case max,average}
    public enum Sizing:String,Codable,CaseIterable {case fixed,adaptive}
    public enum Variant:String,Codable,CaseIterable {
        case maxPool1d="max_pool1d",maxPool2d="max_pool2d",maxPool3d="max_pool3d"
        case avgPool1d="avg_pool1d",avgPool2d="avg_pool2d"
        case adaptiveAvgPool1d="adaptive_avg_pool1d",adaptiveAvgPool2d="adaptive_avg_pool2d",adaptiveMaxPool2d="adaptive_max_pool2d"
        public var method:Method {switch self {case .maxPool1d,.maxPool2d,.maxPool3d,.adaptiveMaxPool2d:return .max;default:return .average}}
        public var sizing:Sizing {switch self {case .adaptiveAvgPool1d,.adaptiveAvgPool2d,.adaptiveMaxPool2d:return .adaptive;default:return .fixed}}
        public var dimensions:Int {switch self {case .maxPool1d,.avgPool1d,.adaptiveAvgPool1d:return 1;case .maxPool3d:return 3;default:return 2}}
        public var title:String {"\(sizing == .adaptive ? "Adaptive ":"")\(method == .max ? "Max":"Average") Pool \(dimensions)D"}
        public var supportsIndices:Bool {method == .max}
    }
    public static let defaultVariant:Variant = .maxPool2d
    public static var componentIDs:Set<String> {Set(Variant.allCases.map(\.rawValue))}
    public let variant:Variant
    public let values:[String:JSONValue]
    public var returnsIndices:Bool {variant.supportsIndices && values["return_indices"] == .bool(true)}
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let variant=Variant(rawValue:componentID) else {throw GraphConfigurationError("Choose an existing supported pooling variant; arbitrary primitive replacement is not allowed.")}
        self.variant=variant
        self.values=try GraphMigration.effectiveConfiguration(of:.init(id:"pooling",type:componentID,config:values),schemaVersion:configurationVersion,registry:registry)
    }
    public static func defaults(for variant:Variant,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:variant.rawValue) else {throw GraphConfigurationError("The selected pooling variant is unavailable.")}
        return try Self(componentID:variant.rawValue,configurationVersion:2,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}),registry:registry).values
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual pooling primitive.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to variant:Variant,configuration:[String:JSONValue]?=nil,connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,let previous=Variant(rawValue:type) else {throw GraphConfigurationError("Only an actual pooling primitive may change pooling variants.")}
        let validated:Self
        if previous==variant,configuration==nil {validated=try Self.configuration(of:node,registry:registry)}
        else {validated=try .init(componentID:variant.rawValue,configurationVersion:2,values:configuration ?? defaults(for:variant,registry:registry),registry:registry)}
        guard let component=registry.component(type:variant.rawValue),connectedOutputPorts.isSubset(of:Set(component.configuredOutputs(validated.values))) else {throw GraphConfigurationError("The selected pooling variant removes an output that still has a connection. Repair the connection explicitly first.")}
        if connectedOutputPorts.contains("indices") {
            guard validated.returnsIndices else {throw GraphConfigurationError("Winning indices are connected. Keep Return indices enabled on a Max pooling variant or repair those connections first.")}
            guard previous.dimensions==variant.dimensions else {throw GraphConfigurationError("Connected winning indices retain their flattened spatial-axis meaning. Repair those connections before changing pooling dimensions.")}
        }
        if previous==variant,configuration==nil {return node}
        var result=node
        result.payload = .modelPrimitive(type:variant.rawValue,configurationVersion:2,config:validated.values)
        return result
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to variant:Variant,configuration:[String:JSONValue]?=nil,connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:configurationVersion,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try switching(wrapped,to:variant,configuration:configuration,connectedOutputPorts:connectedOutputPorts,registry:registry)
        guard case .modelPrimitive(let type,_,let config)=changed.payload else {throw GraphConfigurationError("Pooling replacement did not resolve an existing primitive.")}
        var result=node;result.type=type;result.config=config;result.parameterSettings=changed.parameterSettings
        return result
    }
}
