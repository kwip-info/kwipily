import Foundation
import KwipMLCore

/// Read/edit the same bounded router source without replacing graph IDs, edges or metadata.
enum MixtureRouterPresentation {
    static func object(_ source:String)throws->[String:JSONValue] {
        guard source.unicodeScalars.count<=16384,case .object(let value)=try JSONDecoder().decode(JSONValue.self,from:Data(source.utf8)) else{throw WorkflowArchiveError("Use a bounded router JSON object.")}
        return value
    }
    static func compact(_ source:String)throws->String {
        let value=try object(source),encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let result=String(decoding:try encoder.encode(value),as:UTF8.self)
        guard result.unicodeScalars.count<=4096 else{throw WorkflowArchiveError("Router source exceeds its 4096-character saved limit.")}
        _=try MixtureExpertsConfiguration.pointwiseGraph(.object(value))
        return result
    }
    static func formatted(_ source:String)->String {
        guard let value=try? object(source) else{return source}
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys,.withoutEscapingSlashes]
        return (try? String(decoding:encoder.encode(value),as:UTF8.self)) ?? source
    }
    static func graph(_ source:String)throws->GraphDocument {try MixtureExpertsConfiguration.pointwiseGraph(.object(object(source)))}
    static func setting(_ key:String,to value:JSONValue,nodeID:String,source:String,registry:ComponentRegistry)throws->String {
        var root=try object(source)
        guard case .array(var nodes)=root["nodes"],let index=nodes.firstIndex(where:{if case .object(let node)=$0{return node["id"]?.stringValue==nodeID};return false}),case .object(var node)=nodes[index],let type=node["type"]?.stringValue,let field=registry.component(type:type)?.fields.first(where:{$0.key==key}),field.accepts(value),case .object(var config)=node["config"] else{throw WorkflowArchiveError("Choose a current router layer field and a valid value.")}
        config[key]=value;node["config"] = .object(config);nodes[index] = .object(node);root["nodes"] = .array(nodes)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return try compact(String(decoding:encoder.encode(root),as:UTF8.self))
    }
}
