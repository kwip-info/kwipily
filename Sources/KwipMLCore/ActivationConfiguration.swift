import Foundation

/// One authoring family over existing primitive identities and reviewed runtime contracts.
public struct ActivationConfiguration:Equatable {
    public enum Function:String,CaseIterable,Codable {
        case relu,leakyReLU="leaky_relu",prelu,elu,gelu,silu,sigmoid,tanh,softmax,logSoftmax="log_softmax"
        public var title:String {
            switch self {case .relu:return "ReLU";case .leakyReLU:return "Leaky ReLU";case .prelu:return "PReLU";case .elu:return "ELU";case .gelu:return "GELU";case .silu:return "SiLU";case .sigmoid:return "Sigmoid";case .tanh:return "Tanh";case .softmax:return "Softmax";case .logSoftmax:return "Log Softmax"}
        }
    }
    public static let defaultFunction:Function = .relu
    public static var componentIDs:Set<String> {Set(Function.allCases.map(\.rawValue))}
    public let function:Function
    public let values:[String:JSONValue]
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let function=Function(rawValue:componentID) else {throw GraphConfigurationError("Choose an existing activation function; arbitrary primitive changes are not allowed.")}
        self.function=function
        self.values=try GraphMigration.effectiveConfiguration(of:.init(id:"activation",type:componentID,config:values),schemaVersion:configurationVersion,registry:registry)
    }
    public static func defaults(for function:Function,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:function.rawValue) else {throw GraphConfigurationError("The selected activation is unavailable in this registry.")}
        return try Self(componentID:function.rawValue,configurationVersion:2,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}),registry:registry).values
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual activation primitive.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to function:Function,configuration:[String:JSONValue]?=nil,registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,let previous=Function(rawValue:type) else {throw GraphConfigurationError("Choose an actual activation primitive; arbitrary type replacement is not allowed.")}
        if previous==function,configuration==nil {
            _ = try Self.configuration(of:node,registry:registry)
            return node
        }
        // An explicit valid replacement repairs malformed saved settings; it does
        // not require the old settings to execute or silently reuse invalid values.
        let requested=try configuration ?? defaults(for:function,registry:registry)
        let validated=try Self(componentID:function.rawValue,configurationVersion:2,values:requested,registry:registry)
        var result=node
        result.payload = .modelPrimitive(type:function.rawValue,configurationVersion:2,config:validated.values)
        // The sole learned activation is PReLU. Explicit function replacement cannot
        // carry its slope initialization/trainability directives onto another module.
        if previous != function {result.parameterSettings=nil}
        return result
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to function:Function,configuration:[String:JSONValue]?=nil,registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:configurationVersion,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try switching(wrapped,to:function,configuration:configuration,registry:registry)
        guard case .modelPrimitive(let type,_,let values)=changed.payload else {throw GraphConfigurationError("Activation replacement did not resolve a primitive.")}
        var result=node;result.type=type;result.config=values;result.parameterSettings=changed.parameterSettings
        return result
    }
}
