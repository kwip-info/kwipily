import Foundation
import KwipMLCore

/// Vocabulary authority comes only from the pinned captured graph/state inventory.
/// Unsupported/ambiguous column ancestry returns no token labels, never a guess.
struct TokenVocabularyResolver:Equatable {
    let directory:URL
    let graph:DataFlowDocument
    let states:[String:WorkflowArchiveFile]
    var generationVocabulary:[String:[String]]? = nil
    let endpoint:DataFlowEndpoint
    static func make(manifest:[String:JSONValue],directory:URL,endpoint:DataFlowEndpoint)throws->Self? {
        guard case .array(let entries)=manifest["artifacts"] else {return nil}
        let decoded=try entries.map {try JSONDecoder().decode([String:JSONValue].self,from:JSONEncoder().encode($0))}
        guard let item=decoded.first(where:{$0["path"] == .string("dataflow.json")}),let pin=item["sha256"]?.stringValue,let count=item["bytes"]?.integerValue else {return nil}
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["dataflow.json"],limit:2_000_000)
        guard bytes.count==count,WorkflowArchiveCodec.hash(bytes)==pin else {throw WorkflowFileLeaseError.unsafePath}
        let graph=try JSONDecoder().decode(DataFlowDocument.self,from:bytes)
        guard graph.nodes.contains(where:{$0.operatorID==TextEncodingConfiguration.operatorID || $0.operatorID==SequenceWindowsConfiguration.operatorID}) else {return nil}
        var states:[String:WorkflowArchiveFile]=[:]
        for item in decoded where item["kind"] == .string("state") && item["port"] == .string("state") {
            guard let node=item["nodeID"]?.stringValue,let path=item["path"]?.stringValue,let size=item["bytes"]?.integerValue,let hash=item["sha256"]?.stringValue,states[node]==nil,try WorkflowArchiveCodec.pathParts(path).count==1,(1...20_000_000).contains(size) else {throw WorkflowFileLeaseError.unsafePath}
            states[node] = .init(path:path,bytes:size,sha256:hash)
        }
        return .init(directory:directory,graph:graph,states:states,endpoint:endpoint)
    }
    func at(_ endpoint:DataFlowEndpoint)->Self {.init(directory:directory,graph:graph,states:states,generationVocabulary:generationVocabulary,endpoint:endpoint)}
    /// Pair inspection is available only through unambiguous, column-preserving ancestry.
    func windowConfiguration()->[String:JSONValue]? {
        var current=endpoint,visited=Set<String>()
        while visited.insert(current.node).inserted,visited.count<=128,let node=graph.nodes.first(where:{$0.id==current.node}) {
            if node.operatorID==SequenceWindowsConfiguration.operatorID {return (try? SequenceWindowsConfiguration.validate(node.config)) != nil ? node.config:nil}
            guard ["kwip.data.view","kwip.data.sort","kwip.data.sampleRows","kwip.data.filterRows","kwip.data.unique"].contains(node.operatorID) else {return nil}
            let parents=graph.edges.filter {$0.to.node==node.id && $0.to.port=="table"}
            guard parents.count==1 else {return nil};current=parents[0].from
        }
        return nil
    }
    static func capturedGeneration(directory:URL,column:String,vocabulary:[String])->Self {
        .init(directory:directory,graph:.init(),states:[:],generationVocabulary:[column:vocabulary],endpoint:.init(node:"generation",port:"table"))
    }
    func resolve(column:String)throws->[String]? {
        if let generationVocabulary {return generationVocabulary[column]}
        var current=endpoint,name=column,visited=Set<String>()
        while visited.insert(current.node).inserted,visited.count<=128,let node=graph.nodes.first(where:{$0.id==current.node}) {
            if node.operatorID==SequenceWindowsConfiguration.operatorID {
                guard [node.config["inputColumn"]?.stringValue,node.config["targetColumn"]?.stringValue].contains(name),let source=node.config["tokenColumn"]?.stringValue else {return nil}
                name=source
                let port=node.config["mode"] == .string("partitions") ? current.port:"table"
                let parents=graph.edges.filter {$0.to.node==node.id && $0.to.port==port}
                guard parents.count==1 else {return nil};current=parents[0].from;continue
            }
            if node.operatorID==TextEncodingConfiguration.operatorID {
                guard node.config["outputColumn"] == .string(name),let file=states[node.id] else {return nil}
                let bytes=try DiagnosticFileIO.read(root:directory,parts:[file.path],limit:20_000_000)
                guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256,
                      case .object(let state)=try JSONDecoder().decode(JSONValue.self,from:bytes),state["operator"] == .string(TextEncodingConfiguration.operatorID),state["configuration"] == .object(node.config),case .array(let values)=state["vocabulary"],(1...65536).contains(values.count),values.allSatisfy({$0.stringValue != nil}) else {throw WorkflowFileLeaseError.unsafePath}
                let vocabulary=values.compactMap(\.stringValue)
                guard Set(vocabulary).count==vocabulary.count else {throw WorkflowFileLeaseError.unsafePath}
                return vocabulary
            }
            if node.operatorID=="kwip.data.columns",case .array(let rows)=node.config["rows"] {
                for case .object(let row) in rows where row["outputName"] == .string(name) {
                    guard row["included"] == .bool(true),row["targetType"] == .string("keep"),let source=row["source"]?.stringValue else {return nil};name=source;break
                }
            } else if !["kwip.data.view","kwip.data.sort","kwip.data.sampleRows","kwip.data.partition","kwip.data.filterRows","kwip.data.unique"].contains(node.operatorID) {return nil}
            let parents=graph.edges.filter {$0.to.node==node.id && $0.to.port=="table"}
            guard parents.count==1 else {return nil};current=parents[0].from
        }
        return nil
    }
}
