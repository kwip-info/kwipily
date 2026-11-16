import Foundation
import KwipMLCore

/// Only the worker's structured, captured identity can attribute a runtime failure.
/// A generic message or the last progress node never establishes attribution.
enum WorkflowPreparationFailure {
    static func detail(event:[String:JSONValue],graph:DataFlowDocument,labels:[String:String],registry:DataFlowOperatorRegistry)throws->DataFlowFailureDetail? {
        guard event["event"] == .string("error"),let value=event["dataflowError"] else {return nil}
        var titles:[String:String]=[:]
        for node in graph.nodes {titles[node.operatorID]=registry.operatorDefinition(id:node.operatorID)?.title}
        return try DataFlowFailureDetail.decode(WorkflowArchiveCodec.encode(value),graph:graph,labels:labels,titles:titles)
    }
    static func labels(document:WorkflowDocument,registry:DataFlowOperatorRegistry)->[String:String] {
        Dictionary(uniqueKeysWithValues:document.nodes.filter {$0.address.domain == .data}.map {node -> (String,String) in
            let id:String;if case .dataOperator(let value,_)=node.payload {id=value}else {id=""}
            return (node.address.localID,node.name.isEmpty ? (registry.operatorDefinition(id:id)?.title ?? node.address.localID):String(node.name.prefix(200)))
        })
    }
}
