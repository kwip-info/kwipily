import Foundation

/// JSON data only. Integers preserve Int64 precision, and booleans are never coerced to numbers.
public indirect enum JSONValue:Codable,Equatable,Hashable {
    case null,bool(Bool),integer(Int64),number(Double),string(String),array([JSONValue]),object([String:JSONValue])
    public init(from decoder:Decoder)throws {
        let c=try decoder.singleValueContainer()
        if c.decodeNil(){self = .null}
        else if let v=try? c.decode(Bool.self){self = .bool(v)}
        else if let v=try? c.decode(Int64.self){self = .integer(v)}
        else if let v=try? c.decode(Double.self),v.isFinite{self = .number(v)}
        else if let v=try? c.decode(String.self){self = .string(v)}
        else if let v=try? c.decode([JSONValue].self){self = .array(v)}
        else {self = .object(try c.decode([String:JSONValue].self))}
    }
    public func encode(to encoder:Encoder)throws {
        var c=encoder.singleValueContainer()
        switch self {case .null:try c.encodeNil();case .bool(let v):try c.encode(v);case .integer(let v):try c.encode(v);case .number(let v):try c.encode(v);case .string(let v):try c.encode(v);case .array(let v):try c.encode(v);case .object(let v):try c.encode(v)}
    }
}
public struct DataFlowNode:Codable,Equatable,Identifiable {
    public var id:String;public var operatorID:String;public var config:[String:JSONValue]
    public init(id:String=UUID().uuidString,operatorID:String,config:[String:JSONValue]=[:]){self.id=id;self.operatorID=operatorID;self.config=config}
    enum CodingKeys:String,CodingKey {case id,operatorID="operator",config}
}
public struct DataFlowEndpoint:Codable,Equatable,Hashable {
    public var node:String;public var port:String
    public init(node:String,port:String){self.node=node;self.port=port}
}
public struct DataFlowEdge:Codable,Equatable,Hashable,Identifiable {
    public var from:DataFlowEndpoint;public var to:DataFlowEndpoint
    public var id:String {let values=[from.node,from.port,to.node,to.port];return values.map {"\($0.utf8.count):\($0)"}.joined()}
    public init(from:DataFlowEndpoint,to:DataFlowEndpoint){self.from=from;self.to=to}
}
public struct DataFlowDocument:Codable,Equatable {
    public var schemaVersion:Int;public var nodes:[DataFlowNode];public var edges:[DataFlowEdge]
    public init(schemaVersion:Int=1,nodes:[DataFlowNode]=[],edges:[DataFlowEdge]=[]){self.schemaVersion=schemaVersion;self.nodes=nodes;self.edges=edges}
}
public struct DataFlowEditorNode:Codable,Equatable {
    public var name:String;public var x:Double;public var y:Double
    public init(name:String="",x:Double=0,y:Double=0){self.name=name;self.x=x;self.y=y}
}
/// Private project envelope. Only `graph` is sent to the worker or exported portably.
public struct DataFlowProject:Codable,Equatable {
    public var schemaVersion:Int;public var name:String;public var graph:DataFlowDocument
    public var editor:[String:DataFlowEditorNode];public var sourceBindings:[String:String]
    public init(schemaVersion:Int=1,name:String="Untitled data flow",graph:DataFlowDocument,editor:[String:DataFlowEditorNode]=[:],sourceBindings:[String:String]=[:]){self.schemaVersion=schemaVersion;self.name=name;self.graph=graph;self.editor=editor;self.sourceBindings=sourceBindings}
}
public struct DataFlowContractError:Error,LocalizedError {
    public let message:String
    public var errorDescription:String? {message}
    public let apiIssue:APIConfigurationIssue?
    public init(_ message:String,apiIssue:APIConfigurationIssue?=nil){self.message=message;self.apiIssue=apiIssue}
}
public enum DataFlowCodec {
    public static let maximumBytes=2_000_000
    public static func decodeGraph(_ data:Data)throws->DataFlowDocument {
        try checkBytes(data);try checkPortableKeys(JSONDecoder().decode(JSONValue.self,from:data))
        let graph=try JSONDecoder().decode(DataFlowDocument.self,from:data);try validate(graph);return graph
    }
    public static func encodeGraph(_ graph:DataFlowDocument)throws->Data {try validate(graph);return try encode(graph)}
    public static func decodeProject(_ data:Data)throws->DataFlowProject {
        try checkBytes(data)
        let value=try JSONDecoder().decode(JSONValue.self,from:data)
        guard case .object(let object)=value,Set(object.keys)==Set(["schemaVersion","name","graph","editor","sourceBindings"]),let graph=object["graph"] else {throw DataFlowContractError("Invalid project envelope fields.")}
        try checkPortableKeys(graph)
        let project=try JSONDecoder().decode(DataFlowProject.self,from:data);try validate(project);return project
    }
    public static func encodeProject(_ project:DataFlowProject)throws->Data {try validate(project);return try encode(project)}
    public static func validate(_ graph:DataFlowDocument)throws {
        guard graph.schemaVersion==1,graph.nodes.count<=128,graph.edges.count<=256 else {throw DataFlowContractError("Unsupported schema or graph bounds exceeded.")}
        let ids=Set(graph.nodes.map(\.id));guard ids.count==graph.nodes.count else {throw DataFlowContractError("Duplicate node identities.")}
        for node in graph.nodes {guard validID(node.id),validID(node.operatorID) else {throw DataFlowContractError("Node and operator identifiers require 1–128 characters.")};try checkJSON(.object(node.config),depth:0)}
        var targets=Set<DataFlowEndpoint>();var edges=Set<DataFlowEdge>()
        for edge in graph.edges {guard ids.contains(edge.from.node),ids.contains(edge.to.node),validID(edge.from.port),validID(edge.to.port),edges.insert(edge).inserted,targets.insert(edge.to).inserted else {throw DataFlowContractError("Invalid, duplicate, or multiply connected edge endpoint.")}}
        var remaining=ids,completed=Set<String>()
        while !remaining.isEmpty {let ready=remaining.filter {id in graph.edges.filter {$0.to.node==id}.allSatisfy {completed.contains($0.from.node)}};guard !ready.isEmpty else {throw DataFlowContractError("Dataflow cycle.")};completed.formUnion(ready);remaining.subtract(ready)}
    }
    public static func validate(_ project:DataFlowProject)throws {
        try validate(project.graph);let ids=Set(project.graph.nodes.map(\.id))
        guard project.schemaVersion==1,project.name.count<=500,Set(project.editor.keys).isSubset(of:ids),Set(project.sourceBindings.keys).isSubset(of:ids),project.editor.values.allSatisfy({$0.x.isFinite && $0.y.isFinite && $0.name.count<=500}),project.sourceBindings.values.allSatisfy({!$0.isEmpty && $0.utf8.count<=16_384}) else {throw DataFlowContractError("Invalid private project settings or editor sidecar.")}
    }
    private static func validID(_ v:String)->Bool {!v.isEmpty && v.count<=128}
    private static func checkBytes(_ data:Data)throws {guard data.count<=maximumBytes else {throw DataFlowContractError("Dataflow JSON exceeds 2 MB.")}}
    private static func encode<T:Encodable>(_ value:T)throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys];let data=try encoder.encode(value);try checkBytes(data);return data}
    private static func checkJSON(_ value:JSONValue,depth:Int)throws {
        guard depth<=32 else {throw DataFlowContractError("Configuration nesting exceeds 32 levels.")}
        switch value {case .number(let v):guard v.isFinite else {throw DataFlowContractError("Nonfinite configuration number.")};case .array(let a):guard a.count<=4096 else {throw DataFlowContractError("Configuration array too large.")};for v in a {try checkJSON(v,depth:depth+1)};case .object(let o):guard o.count<=4096 else {throw DataFlowContractError("Configuration object too large.")};for (k,v) in o {guard k.utf8.count<=16_384 else {throw DataFlowContractError("Configuration key too large.")};try checkJSON(v,depth:depth+1)};case .string(let s):guard s.utf8.count<=65_536 else {throw DataFlowContractError("Configuration string too large.")};default:break}
    }
    private static func checkPortableKeys(_ value:JSONValue)throws {
        func object(_ value:JSONValue,_ keys:Set<String>)throws->[String:JSONValue] {guard case .object(let o)=value,Set(o.keys)==keys else {throw DataFlowContractError("Unexpected or missing portable graph fields.")};return o}
        let g=try object(value,["schemaVersion","nodes","edges"])
        guard case .integer(1)=g["schemaVersion"],case .array(let nodes)=g["nodes"],case .array(let edges)=g["edges"] else {throw DataFlowContractError("Invalid portable graph schema.")}
        for n in nodes {_ = try object(n,["id","operator","config"])}
        for e in edges {let o=try object(e,["from","to"]);_ = try object(o["from"]!,["node","port"]);_ = try object(o["to"]!,["node","port"])}
    }
}
