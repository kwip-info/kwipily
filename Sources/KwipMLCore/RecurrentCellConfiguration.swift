import Foundation

/// Family-restricted authoring surface; common editing owns state and parameter safeguards.
public struct RecurrentCellConfiguration:Equatable {
    public typealias Method = RecurrentSequenceConfiguration.Method
    public static let defaultMethod:Method = .rnn
    public static let componentIDs:Set<String> = ["rnn_cell","lstm_cell","gru_cell"]
    public let method:Method
    public let values:[String:JSONValue]
    public init(componentID:String,configurationVersion:Int,values:[String:JSONValue],registry:ComponentRegistry)throws {
        guard Self.componentIDs.contains(componentID) else {throw GraphConfigurationError("Choose an existing Recurrent Cell tool.")}
        let parsed=try RecurrentAuthoringConfiguration(componentID:componentID,configurationVersion:configurationVersion,values:values,registry:registry)
        method=parsed.method;self.values=parsed.values
    }
    public static func defaults(for method:Method,registry:ComponentRegistry)throws->[String:JSONValue] {
        try RecurrentAuthoringConfiguration.defaults(for:method,isCell:true,registry:registry)
    }
    public static func configuration(of node:WorkflowNode,registry:ComponentRegistry)throws->Self {
        guard case .modelPrimitive(let type,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an actual recurrent component.")}
        return try .init(componentID:type,configurationVersion:version,values:values,registry:registry)
    }
    public static func switching(_ node:WorkflowNode,to method:Method,configuration:[String:JSONValue]?=nil,connectedInputPorts:Set<String>=[],connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->WorkflowNode {
        guard case .modelPrimitive(let type,_,_)=node.payload,componentIDs.contains(type) else {throw GraphConfigurationError("Choose an existing Recurrent Cell tool.")}
        return try RecurrentAuthoringConfiguration.switching(node,to:method,configuration:configuration,connectedInputPorts:connectedInputPorts,connectedOutputPorts:connectedOutputPorts,registry:registry)
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to method:Method,configuration:[String:JSONValue]?=nil,connectedInputPorts:Set<String>=[],connectedOutputPorts:Set<String>=[],registry:ComponentRegistry)throws->GraphNode {
        guard componentIDs.contains(node.type) else {throw GraphConfigurationError("Choose an existing Recurrent Cell tool.")}
        return try RecurrentAuthoringConfiguration.switching(node,configurationVersion:configurationVersion,to:method,configuration:configuration,connectedInputPorts:connectedInputPorts,connectedOutputPorts:connectedOutputPorts,registry:registry)
    }
}
