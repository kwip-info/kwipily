import Foundation

/// Shared sequence/cell authoring; every edit stays within the source primitive family.
public struct RecurrentAuthoringConfiguration:Equatable {
    public typealias Method = RecurrentSequenceConfiguration.Method
    public static let componentIDs:Set<String> = ["rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell"]
    public static func method(for type:String)->Method? {componentIDs.contains(type) ? Method(rawValue:type.replacingOccurrences(of:"_cell",with:"")):nil}
    public static func primitiveID(_ method:Method,isCell:Bool)->String {method.rawValue+(isCell ? "_cell":"")}
    public let isCell:Bool
    public let method:Method
    public let values:[String:JSONValue]
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard let method=Self.method(for:componentID) else {throw GraphConfigurationError("Choose a supported recurrent sequence or cell method.")}
        let canonical=try GraphMigration.effectiveConfiguration(of:.init(id:"recurrent",type:componentID,config:values),schemaVersion:configurationVersion,registry:registry)
        let parsed=try RecurrentConfiguration(componentID:componentID,values:canonical)
        guard parsed.parameterCount<=2_000_000 else {throw GraphConfigurationError("Recurrent exceeds the 2 million parameter limit. Reduce the feature widths or, for sequences, layers and directions.")}
        self.method=method;self.isCell=parsed.isCell;self.values=parsed.values
    }
    public static func defaults(for method:Method,isCell:Bool=false,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:primitiveID(method,isCell:isCell)) else {throw GraphConfigurationError("The selected recurrent method is unavailable.")}
        return try Self(componentID:primitiveID(method,isCell:isCell),configurationVersion:2,values:Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}),registry:registry).values
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual recurrent sequence or cell.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to method:Method,configuration:[String:JSONValue]?=nil,connectedInputPorts:Set<String>=[],connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,let previous=Self.method(for:type) else {throw GraphConfigurationError("Only an existing recurrent sequence or cell may change methods.")}
        let isCell=type.hasSuffix("_cell")
        let targetType=primitiveID(method,isCell:isCell)
        let validated:Self
        if previous==method,configuration==nil {validated=try Self.configuration(of:node,registry:registry)}
        else {
            var values=try configuration ?? defaults(for:method,isCell:isCell,registry:registry)
            if configuration==nil {
                let current=try Self.configuration(of:node,registry:registry)
                for key in ["input_size","hidden_size","num_layers","bias","bidirectional","dropout","state_mode"] where values[key] != nil {values[key]=current.values[key]}
            }
            validated=try .init(componentID:targetType,configurationVersion:2,values:values,registry:registry)
        }
        let parsed=try RecurrentConfiguration(componentID:targetType,values:validated.values)
        guard connectedInputPorts.isSubset(of:Set(parsed.configuredInputs)),connectedOutputPorts.isSubset(of:Set(parsed.outputPorts)) else {
            throw GraphConfigurationError("This change removes a connected or exposed hidden/cell state port. Disconnect or repair that state connection explicitly first.")
        }
        if let settings=node.parameterSettings {
            try LayerParameterSettings.validate(settings)
            guard previous==method || settings.isEmpty else {throw GraphConfigurationError("Recurrent methods have different gate parameters. Clear the authored weight and bias policies explicitly before changing method.")}
            let choices=try LayerParameterInventory.choices(type:targetType,config:validated.values,version:2,registry:registry)
            for (parameter,value) in settings {
                guard let choice=choices.first(where:{$0.id==parameter}) else {throw GraphConfigurationError("This change removes the authored \(parameter) initialization or trainability policy. Remove that policy explicitly first.")}
                if case .object(let row)=value,case .object(let initialization)=row["initialization"],
                   let algorithm=initialization["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {
                    throw GraphConfigurationError("Xavier and Kaiming require multidimensional weights. Choose a supported initializer for \(parameter).")
                }
            }
        }
        if previous==method,configuration==nil {return node}
        var next=node;next.payload = .modelPrimitive(type:targetType,configurationVersion:2,config:validated.values);return next
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to method:Method,configuration:[String:JSONValue]?=nil,connectedInputPorts:Set<String>=[],connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->GraphNode {
        let wrapped=WorkflowNode(address:.init(domain:.model,localID:node.id),name:node.name,x:node.x,y:node.y,payload:.modelPrimitive(type:node.type,configurationVersion:configurationVersion,config:node.config),parameterSettings:node.parameterSettings)
        let changed=try switching(wrapped,to:method,configuration:configuration,connectedInputPorts:connectedInputPorts,connectedOutputPorts:connectedOutputPorts,registry:registry)
        guard case .modelPrimitive(let type,_,let values)=changed.payload else {throw GraphConfigurationError("Recurrent replacement did not resolve an existing primitive.")}
        var next=node;next.type=type;next.config=values;return next
    }
}
