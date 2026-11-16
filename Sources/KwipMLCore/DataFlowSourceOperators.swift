import Foundation

/// Explicit local source identities. A source never derives filesystem authority from configuration.
public enum DataFlowSourceOperators {
    public static let capturedTable="kwip.data.capturedTable"
    public static let tableInput="kwip.data.sourceCSV"
    public static let fileInput="kwip.data.sourceFiles"
    public static func contains(_ id:String)->Bool {id == tableInput || id == fileInput || id == capturedTable}
    public static func fileConfigurations(in document:WorkflowDocument)->[WorkflowNodeAddress:[String:JSONValue]] {
        Dictionary(uniqueKeysWithValues:document.nodes.compactMap {node in
            guard case .dataOperator(let id,let config)=node.payload,id==fileInput else {return nil}
            return (node.address,config)
        })
    }
    public static func isSource(_ node:WorkflowNode)->Bool {
        guard case .dataOperator(let id,_)=node.payload else {return false}
        return node.address.domain == .data && contains(id)
    }
}
