import Foundation
import KwipMLCore

/// Explicit Save history. Automatic authoring persistence and immutable run evidence
/// remain separate authorities. A bounded journal is replaced atomically, never pruned.
final class CanvasVersionJournal {
    enum ChangeKind:String,Codable {case added,removed,configuration,connection,layout,metadata}
    struct Change:Codable,Equatable,Identifiable {
        var component:String;var field:String;var before:String;var after:String;var kind:ChangeKind
        var ordinal=0
        var id:String {String(ordinal)+"/"+component+"/"+field}
    }
    struct Patch:Codable {var path:[String];var value:JSONValue?}
    struct Version:Codable,Identifiable {
        var id:String;var number:Int;var savedAt:Date;var label:String;var hash:String
        var base:JSONValue?;var patch:[Patch];var changes:[Change]
        var dependencies:[FlowDocumentDependency]? = nil
    }
    private struct Envelope:Codable {var schemaVersion=1;var projectID:String;var records:[Version]}
    private let root:URL
    let projectID:String
    private var records:[Version]=[]
    private var sourceBytes:Data?
    var versions:[Version] {records.reversed()}
    var fileURL:URL {root.appendingPathComponent("canvas-versions/"+projectID+".json")}
    private var parts:[String] {["canvas-versions",projectID+".json"]}
    init(root:URL,projectID:String)throws {
        guard UUID(uuidString:projectID)?.uuidString==projectID else {throw WorkflowArchiveError("Invalid canvas history identity.")}
        self.root=root;self.projectID=projectID
        try reload()
    }
    func reload()throws {
        let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:8_000_000)
        if let bytes {
            let value=try JSONDecoder().decode(Envelope.self,from:bytes)
            guard value.schemaVersion==1,value.projectID==projectID,value.records.count<=512,
                  Set(value.records.map(\.id)).count==value.records.count else {throw WorkflowArchiveError("Unsupported canvas history.")}
            records=try Self.verify(value.records,projectID:projectID)
        } else {records=[]}
        sourceBytes=bytes
    }
    func document(version id:String)throws->WorkflowDocument {
        guard let index=records.firstIndex(where:{$0.id==id}) else {throw WorkflowArchiveError("Canvas version unavailable.")}
        return try Self.decode(Self.reconstruct(records,index:index),projectID:projectID)
    }
    var lastDocument:WorkflowDocument? {records.last.flatMap {try? document(version:$0.id)}}
    func matchesSavedDocument(_ document:WorkflowDocument)throws->Bool {
        guard let last=records.last else{return false}
        return try Self.semantic(self.document(version:last.id)) == Self.semantic(document)
    }
    @discardableResult func save(_ document:WorkflowDocument,label:String="",dependencies:[FlowDocumentDependency]?=nil)throws->Version {
        try DiagnosticFileIO.withExclusiveLock(root:root,parts:["canvas-versions",projectID+".lock"]) {
            try saveLocked(document,label:label,dependencies:dependencies)
        }
    }
    private func saveLocked(_ document:WorkflowDocument,label:String,dependencies:[FlowDocumentDependency]?)throws->Version {
        guard try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:8_000_000)==sourceBytes else {throw WorkflowArchiveError("Canvas history changed in another window. Refresh before saving.")}
        guard document.projectID==projectID,document.drafts.isEmpty,label.count<=120 else {throw WorkflowArchiveError("Apply or discard open container drafts before saving a canvas version.")}
        let current=try Self.json(document)
        if let last=records.last,let old=lastDocument,try Self.semantic(old)==Self.semantic(document),last.dependencies==dependencies {return last}
        guard records.count<512 else {throw WorkflowArchiveError("Canvas history reached its 512-version limit. Existing versions are preserved.")}
        let delta=try records.last.map {_ in Self.diff(try Self.json(lastDocument!),current)} ?? []
        var next=Version(id:UUID().uuidString,number:records.count+1,savedAt:Date(),label:label,hash:try Self.hash(current),base:records.isEmpty ? current:nil,patch:delta,changes:lastDocument.map {Self.changes(from:$0,to:document)} ?? [])
        next.dependencies=dependencies
        let proposed=records+[next]
        guard try Self.hash(current)==next.hash else {throw WorkflowArchiveError("Canvas version integrity failure.")}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let bytes=try encoder.encode(Envelope(projectID:projectID,records:proposed))
        guard bytes.count<=8_000_000 else {throw WorkflowArchiveError("Canvas history reached its 8 MB limit. Existing versions are preserved.")}
        guard try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:8_000_000)==sourceBytes else {throw WorkflowArchiveError("Canvas history changed in another window. Refresh before saving.")}
        try DiagnosticFileIO.writeAtomic(bytes,root:root,parts:parts)
        records=proposed;sourceBytes=bytes
        return next
    }
    /// Admission for both the preserved present and restored checkpoint precedes
    /// source mutation. No destructive history rollover to make room.
    func requireRestoreCapacity(current:WorkflowDocument,target:WorkflowDocument,currentDependencies:[FlowDocumentDependency]?,targetDependencies:[FlowDocumentDependency]?)throws {
        var proposed=records
        var previous=lastDocument
        for (source,label,dependencies) in [(current,"Before restore",currentDependencies),(target,"Restored version",targetDependencies)] {
            if let previous,let last=proposed.last,try Self.semantic(previous)==Self.semantic(source),last.dependencies==dependencies {continue}
            guard proposed.count<512 else {throw WorkflowArchiveError("Canvas history has no room for a recoverable restore. Existing versions are preserved.")}
            let value=try Self.json(source)
            let patch=try previous.map {Self.diff(try Self.json($0),value)} ?? []
            var record=Version(id:UUID().uuidString,number:proposed.count+1,savedAt:Date(),label:label,hash:try Self.hash(value),base:proposed.isEmpty ? value:nil,patch:patch,changes:previous.map {Self.changes(from:$0,to:source)} ?? [])
            record.dependencies=dependencies;proposed.append(record);previous=source
        }
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        guard try encoder.encode(Envelope(projectID:projectID,records:proposed)).count<=8_000_000 else {throw WorkflowArchiveError("Canvas history has no room for a recoverable restore. Existing versions are preserved.")}
    }
    private static func json(_ source:WorkflowDocument)throws->JSONValue {try JSONDecoder().decode(JSONValue.self,from:WorkflowCodec.encode(source))}
    private static func bytes(_ value:JSONValue)throws->Data {let e=JSONEncoder();e.outputFormatting=[.sortedKeys];return try e.encode(value)}
    private static func hash(_ value:JSONValue)throws->String {WorkflowArchiveCodec.hash(try bytes(value))}
    private static func semantic(_ source:WorkflowDocument)throws->Data {var value=source;value.revision=0;return try bytes(json(value))}
    private static func decode(_ value:JSONValue,projectID:String)throws->WorkflowDocument {
        let source=try WorkflowCodec.decode(bytes(value))
        guard source.projectID==projectID,source.drafts.isEmpty else {throw WorkflowArchiveError("Canvas history belongs to another project or contains a draft.")};return source
    }
    private static func reconstruct(_ records:[Version],index:Int)throws->JSONValue {
        guard let first=records.first,let initial=first.base,first.patch.isEmpty else {throw WorkflowArchiveError("Canvas history has no valid base.")}
        var value=initial
        for (i,record) in records.prefix(index+1).enumerated() {
            guard record.number==i+1,UUID(uuidString:record.id) != nil,record.label.count<=120,record.patch.count<=20000,(i==0 || record.base==nil) else {throw WorkflowArchiveError("Canvas history chain is invalid.")}
            if i>0 {for patch in record.patch {try apply(patch,to:&value)}}
            guard try hash(value)==record.hash else {throw WorkflowArchiveError("Canvas history failed integrity verification.")}
        };return value
    }
    private static func verify(_ records:[Version],projectID:String)throws->[Version] {
        guard !records.isEmpty else {return []}
        var value=records[0].base ?? .null
        var previous:WorkflowDocument?
        var verified:[Version]=[]
        for (i,r) in records.enumerated() {
            guard r.number==i+1,UUID(uuidString:r.id) != nil,r.label.count<=120,r.patch.count<=20000,(i==0 ? r.base != nil && r.patch.isEmpty:r.base==nil) else {throw WorkflowArchiveError("Canvas history chain is invalid.")}
            if i>0 {for p in r.patch {try apply(p,to:&value)}}
            guard try hash(value)==r.hash else {throw WorkflowArchiveError("Canvas history failed integrity verification.")}
            let source=try decode(value,projectID:projectID)
            let expected=previous.map {changes(from:$0,to:source)} ?? []
            // Display vocabulary may evolve independently of saved source. Never
            // trust serialized summaries or reject exact deltas for wording changes.
            var normalized=r;normalized.changes=expected;verified.append(normalized)
            previous=source
        }
        return verified
    }
    private static func diff(_ lhs:JSONValue,_ rhs:JSONValue,path:[String]=[])->[Patch] {
        if (try? bytes(lhs)) == (try? bytes(rhs)) {return []}
        switch (lhs,rhs) {
        case (.object(let a),.object(let b)):
            return Set(a.keys).union(b.keys).sorted().flatMap {key->[Patch] in
                if let x=a[key],let y=b[key] {return diff(x,y,path:path+[key])}
                return [Patch(path:path+[key],value:b[key])]
            }
        case (.array(let a),.array(let b)) where a.count==b.count:
            return a.indices.flatMap {diff(a[$0],b[$0],path:path+[String($0)])}
        default:return [Patch(path:path,value:rhs)]
        }
    }
    private static func apply(_ patch:Patch,to root:inout JSONValue)throws {
        guard patch.path.count<=64 else {throw WorkflowArchiveError("Canvas delta is too deep.")}
        func update(_ value:inout JSONValue,_ path:ArraySlice<String>)throws {
            guard let key=path.first else {guard let replacement=patch.value else {throw WorkflowArchiveError("Cannot remove canvas root.")};value=replacement;return}
            switch value {
            case .object(var object):
                if path.count==1 {object[key]=patch.value}
                else {guard var child=object[key] else {throw WorkflowArchiveError("Canvas delta target missing.")};try update(&child,path.dropFirst());object[key]=child}
                value = .object(object)
            case .array(var array):
                guard let i=Int(key),array.indices.contains(i),(path.count>1 || patch.value != nil) else {throw WorkflowArchiveError("Canvas delta index invalid.")}
                try update(&array[i],path.dropFirst());value = .array(array)
            default:throw WorkflowArchiveError("Canvas delta target invalid.")
            }
        };try update(&root,patch.path[...])
    }
    static func changes(from old:WorkflowDocument,to new:WorkflowDocument)->[Change] {
        var rows:[Change]=[]
        func add(_ name:String,_ field:String,_ a:String,_ b:String,_ kind:ChangeKind) {rows.append(.init(component:name,field:field,before:a,after:b,kind:kind))}
        if old.name != new.name {add("Canvas","Name",old.name,new.name,.metadata)}
        let a=Dictionary(uniqueKeysWithValues:old.nodes.map {($0.address,$0)}),b=Dictionary(uniqueKeysWithValues:new.nodes.map {($0.address,$0)})
        func title(_ n:WorkflowNode)->String {
            if !n.name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty {return n.name}
            switch n.payload {case .dataOperator(let id,_):return id.split(separator:".").last.map(String.init) ?? "Table tool"
            case .modelPrimitive(let id,_,_):return id.capitalized
            default:return n.address.domain == .control ? "Workflow component":"Model block"}
        }
        for n in new.nodes {
            let name=title(n)
            guard let before=a[n.address] else {add(name,"Component","—","Added",.added);continue}
            if before.name != n.name {add(name,"Name",title(before),name,.metadata)}
            if before.x != n.x || before.y != n.y {add(name,"Position",String(format:"%.0f, %.0f",before.x,before.y),String(format:"%.0f, %.0f",n.x,n.y),.layout)}
            if before.payload != n.payload {
                if let left=try? JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(before.payload)),
                   let right=try? JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(n.payload)) {
                    for delta in diff(left,right) {
                        func value(at path:ArraySlice<String>,in v:JSONValue)->JSONValue? {
                            guard let k=path.first else {return v}
                            if case .object(let o)=v,let c=o[k] {return value(at:path.dropFirst(),in:c)}
                            if case .array(let a)=v,let i=Int(k),a.indices.contains(i) {return value(at:path.dropFirst(),in:a[i])};return nil
                        }
                        func display(_ v:JSONValue?)->String {
                            guard let v else {return "—"}
                            if case .string(let text)=v {return String(text.prefix(160))}
                            if case .object=v {return "Updated fields"}
                            if case .array(let values)=v {return "\(values.count) items"}
                            return String((try? String(data:bytes(v),encoding:.utf8)) ?? "—").prefix(160).description
                        }
                        let labels=["runSetup":"Run settings","trainingDataMapping":"Data mapping","out_features":"Output features","in_features":"Input features","epochs":"Epochs","learningRate":"Learning rate","batchSize":"Batch size","device":"Device","seed":"Seed","type":"Type","configurationVersion":"Configuration version"]
                        let field=delta.path.filter {$0 != "config"}.map {labels[$0] ?? $0.replacingOccurrences(of:"_",with:" ")}.joined(separator:" · ")
                        add(name,field,display(value(at:delta.path[...],in:left)),display(delta.value),.configuration)
                    }
                } else {add(name,"Configuration","Previous settings","Updated settings",.configuration)}
            }
            if before.parameterSettings != n.parameterSettings {add(name,"Weights & biases","Previous settings","Updated settings",.configuration)}
        }
        for n in old.nodes where b[n.address]==nil {add(title(n),"Component","Present","Removed",.removed)}
        func endpoint(_ e:WorkflowEndpoint,_ nodes:[WorkflowNodeAddress:WorkflowNode])->String {(nodes[e.address].map(title) ?? "Component")+" · "+e.port}
        for edge in old.edges where !new.edges.contains(edge) {add(b[edge.to.address].map(title) ?? "Connection","Connection",endpoint(edge.from,a),"Disconnected",.connection)}
        for edge in new.edges where !old.edges.contains(edge) {add(b[edge.to.address].map(title) ?? "Connection","Connection","—",endpoint(edge.from,b)+" → "+edge.to.port,.connection)}
        if old.definitions != new.definitions {add("Containers","Internal structure","Previous definitions","Updated definitions",.configuration)}
        if old.modelInputs != new.modelInputs || old.modelSourceVersions != new.modelSourceVersions || old.guidedBuild != new.guidedBuild {add("Canvas","Model / lesson metadata","Previous metadata","Updated metadata",.metadata)}
        return rows.enumerated().map {index,value in var row=value;row.ordinal=index;return row}
    }
}
