import Foundation
import Darwin
import CryptoKit
import KwipMLCore

/// Read ordered IDs and pinned lineage without expanding row values. The caller
/// checks protected membership/role; the scientific reader checks values and transforms.
final class WorkflowRetainedTablePopulation {
    private static let magic=Data("KWIP-TABLE-PACK".utf8)+Data([0,1,10])
    private let root:URL
    private let prefix:[String]
    private let files:[String:WorkflowArchiveFile]
    private var verified:[String:Stamp]=[:]
    private var objects=0,metadataBytes=0
    private struct Stamp:Equatable {
        let device:dev_t,inode:ino_t,size:off_t,modified:Int64,modifiedNS:Int64,changed:Int64,changedNS:Int64
        init(_ value:stat){device=value.st_dev;inode=value.st_ino;size=value.st_size;modified=Int64(value.st_mtimespec.tv_sec);modifiedNS=Int64(value.st_mtimespec.tv_nsec);changed=Int64(value.st_ctimespec.tv_sec);changedNS=Int64(value.st_ctimespec.tv_nsec)}
    }
    private static func fail()->WorkflowArchiveError {WorkflowArchiveError("Retained evaluation population has invalid pins, geometry or identities.")}
    init(root:URL,prefix:[String],inventory:[WorkflowArchiveFile],maximumBytes:Int)throws {
        guard maximumBytes>=0 else {throw Self.fail()}
        self.root=root;self.prefix=prefix
        let start=prefix.joined(separator:"/")+"/"
        var selected:[String:WorkflowArchiveFile]=[:],total=0
        for file in inventory where file.path.hasPrefix(start) {
            let name=String(file.path.dropFirst(start.count))
            guard PreparationArtifactPolicy.isSegmentName(name) || name=="tables.kwipdata" else {continue}
            guard selected[name]==nil,file.bytes>=Self.magic.count,file.bytes<=maximumBytes-total,WorkflowArchiveCodec.isHash(file.sha256) else {throw Self.fail()}
            total+=file.bytes;selected[name]=file
        }
        files=selected
    }
    private func open(_ name:String)throws->Int32 {
        try Task.checkCancellation()
        guard let file=files[name] else {throw Self.fail()}
        let fd=try DiagnosticFileIO.openRegularFile(root:root,parts:prefix+[name],limit:file.bytes)
        do {
            var before=stat();guard fstat(fd,&before)==0,before.st_size==file.bytes else {throw Self.fail()}
            let stamp=Stamp(before)
            if let previous=verified[name] {guard previous==stamp else {throw Self.fail()}}
            else {
                var hash=SHA256(),offset=0,buffer=[UInt8](repeating:0,count:65_536)
                while offset<file.bytes {
                    try Task.checkCancellation()
                    let count=Darwin.pread(fd,&buffer,min(buffer.count,file.bytes-offset),off_t(offset))
                    if count<0 && errno==EINTR {continue}
                    guard count>0 else {throw Self.fail()}
                    if offset==0 {guard Data(buffer.prefix(count)).starts(with:Self.magic) else {throw Self.fail()}}
                    hash.update(data:Data(buffer.prefix(count)));offset+=count
                }
                var after=stat();guard fstat(fd,&after)==0,Stamp(after)==stamp,
                    hash.finalize().map({String(format:"%02x",$0)}).joined()==file.sha256 else {throw Self.fail()}
                verified[name]=stamp
            }
            return fd
        } catch {Darwin.close(fd);throw error}
    }
    func finish()throws {
        // Also cover segment files used only by the scientific value reader.
        for name in files.keys.sorted(){let fd=try open(name);Darwin.close(fd)}
    }
    private static func integer(_ value:JSONValue?,maximum:Int)throws->Int {
        guard case .integer(let count)=value,count>=0,count<=Int64(maximum) else {throw fail()};return Int(count)
    }
    private static func object(_ value:JSONValue,fields:Set<String>?=nil)throws->[String:JSONValue] {
        guard case .object(let result)=value,fields==nil || Set(result.keys)==fields else {throw fail()};return result
    }
    private static func decode(_ bytes:Data,maximum:Int)throws->JSONValue {
        guard bytes.count<=maximum else {throw fail()}
        // Reject duplicate (including escaped-equivalent) JSON keys before decoding.
        let b=Array(bytes);var i=0,stack:[Set<String>?]=[]
        while i<b.count {
            switch b[i] {
            case 123:stack.append(Set<String>());i+=1
            case 91:stack.append(nil);i+=1
            case 125,93:if !stack.isEmpty {stack.removeLast()};i+=1
            case 34:
                let start=i;i+=1
                while i<b.count {if b[i]==92 {i+=2;continue};if b[i]==34 {i+=1;break};i+=1}
                var next=i;while next<b.count && [9,10,13,32].contains(b[next]) {next+=1}
                if next<b.count,b[next]==58,!stack.isEmpty,var keys=stack[stack.count-1] {
                    let key=try JSONDecoder().decode(String.self,from:Data(b[start..<min(i,b.count)]))
                    guard keys.insert(key).inserted else {throw fail()};stack[stack.count-1]=keys
                }
            default:i+=1
            }
            guard stack.count<=64 else {throw fail()}
        }
        return try JSONDecoder().decode(JSONValue.self,from:bytes)
    }
    private func node(_ reference:JSONValue,owner:String?)throws->(JSONValue,String) {
        try Task.checkCancellation();objects+=1;guard objects<=100_000 else {throw Self.fail()}
        let value=try Self.object(reference,fields:["pack","object"])
        let name:String
        if value["pack"] == .null {guard let owner else {throw Self.fail()};name=owner}
        else {
            let pack=try Self.object(value["pack"]!,fields:["schemaVersion","kind","path","bytes","sha256"])
            guard pack["schemaVersion"] == .integer(1),pack["kind"] == .string("tableArtifactPack"),
                  let path=pack["path"]?.stringValue,let file=files[path],
                  pack["bytes"] == .integer(Int64(file.bytes)),pack["sha256"] == .string(file.sha256) else {throw Self.fail()}
            name=path
        }
        guard let file=files[name] else {throw Self.fail()}
        let range=try Self.object(value["object"]!,fields:["offset","bytes","sha256"])
        let offset=try Self.integer(range["offset"],maximum:file.bytes),length=try Self.integer(range["bytes"],maximum:2_097_152)
        guard offset>=Self.magic.count,length>0,length<=file.bytes-offset,
              let pin=range["sha256"]?.stringValue,WorkflowArchiveCodec.isHash(pin),length<=256_000_000-metadataBytes else {throw Self.fail()}
        metadataBytes+=length
        let fd=try open(name);defer{Darwin.close(fd)}
        var bytes=Data(),buffer=[UInt8](repeating:0,count:min(65_536,length))
        while bytes.count<length {
            try Task.checkCancellation()
            let count=Darwin.pread(fd,&buffer,min(buffer.count,length-bytes.count),off_t(offset+bytes.count))
            if count<0 && errno==EINTR {continue};guard count>0 else {throw Self.fail()}
            bytes.append(contentsOf:buffer.prefix(count))
        }
        var after=stat();guard fstat(fd,&after)==0,Stamp(after)==verified[name],WorkflowArchiveCodec.hash(bytes)==pin else {throw Self.fail()}
        return (try Self.decode(bytes,maximum:2_097_152),name)
    }
    private func identities(_ reference:JSONValue,owner:String,selectionDepth:Int)throws->[String] {
        let (value,children)=try node(reference,owner:owner)
        if case .array(let values)=value {
            guard !values.isEmpty,values.count<=256 else {throw Self.fail()}
            let ids=values.compactMap(\.stringValue);guard ids.count==values.count else {throw Self.fail()};return ids
        }
        let selected=try Self.object(value,fields:["schemaVersion","kind","source","sourceCount","positions"])
        guard selectionDepth>0,selected["schemaVersion"] == .integer(1),selected["kind"] == .string("selectedColumn"),
              case .array(let positions)=selected["positions"],!positions.isEmpty,positions.count<=256 else {throw Self.fail()}
        let count=try Self.integer(selected["sourceCount"],maximum:256);guard count>0 else {throw Self.fail()}
        let indexes=try positions.map{try Self.integer($0,maximum:count-1)}
        guard Set(indexes).count==indexes.count else {throw Self.fail()}
        let source=try identities(selected["source"]!,owner:children,selectionDepth:selectionDepth-1)
        guard source.count==count else {throw Self.fail()};return indexes.map{source[$0]}
    }
    private func rootIndex(_ value:JSONValue,count:Int)throws->(JSONValue,Int) {
        let root=try Self.object(value,fields:["reference","count","height"])
        guard root["count"] == .integer(Int64(count)) else {throw Self.fail()}
        let height=try Self.integer(root["height"],maximum:8)
        _=try Self.object(root["reference"]!,fields:["pack","object"])
        return (root["reference"]!,height)
    }
    private func index(_ reference:JSONValue,owner:String,height:Int,count:Int,selectionDepth:Int)throws->[String] {
        let (value,children)=try node(reference,owner:owner)
        let page=try Self.object(value,fields:["schemaVersion","kind","height","count","entries"])
        guard page["schemaVersion"] == .integer(1),page["kind"] == .string("orderedIndex"),
              page["height"] == .integer(Int64(height)),page["count"] == .integer(Int64(count)),
              case .array(let entries)=page["entries"],entries.count<=128,!entries.isEmpty else {throw Self.fail()}
        var result:[String]=[],total=0
        for item in entries {
            let field=height==0 ? "value":"reference",entry=try Self.object(item,fields:["count",field])
            let span=try Self.integer(entry["count"],maximum:count-total);guard span>0 else {throw Self.fail()};total+=span
            let values=try height==0 ? identities(entry[field]!,owner:children,selectionDepth:selectionDepth):index(entry[field]!,owner:children,height:height-1,count:span,selectionDepth:selectionDepth)
            guard values.count==span else {throw Self.fail()};result+=values
        }
        guard total==count else {throw Self.fail()};return result
    }
    func lineage(_ bytes:Data)throws->[String:Any] {
        let envelope=try Self.object(Self.decode(bytes,maximum:50_000_000))
        let columns:[JSONValue],count:Int,ids:[String]
        var lineage:[String:JSONValue]
        if envelope["kind"] == .string("segmentedTable") {
            guard Set(envelope.keys)==["schemaVersion","kind","columns","table"],envelope["schemaVersion"] == .integer(4),
                  case .array(let outerColumns)=envelope["columns"] else {throw Self.fail()}
            let (value,owner)=try node(envelope["table"]!,owner:nil),table=try Self.object(value)
            let version=try Self.integer(table["schemaVersion"],maximum:2)
            let fields:Set<String>=["schemaVersion","kind","columns","rowCount","columnIndexes","identityIndex","lineage"]
            guard [1,2].contains(version),Set(table.keys)==fields.union(version==2 ? ["selectionDepth"]:[]),table["kind"] == .string("columnTable"),
                  table["columns"] == envelope["columns"],case .array(let indexes)=table["columnIndexes"],indexes.count==outerColumns.count else {throw Self.fail()}
            count=try Self.integer(table["rowCount"],maximum:100_000);columns=outerColumns
            let depth=version==2 ? try Self.integer(table["selectionDepth"],maximum:32):0
            guard version==1 || depth>0 else {throw Self.fail()}
            for column in indexes {_=try rootIndex(column,count:count)}
            let (reference,height)=try rootIndex(table["identityIndex"]!,count:count)
            ids=try index(reference,owner:owner,height:height,count:count,selectionDepth:depth)
            lineage=try Self.object(table["lineage"]!,fields:["role","sourceDigest","membershipDigest","dataVersion","partitionDigest","fittedStateDigests","diagnostics"])
            lineage["observationIDs"] = .array(ids.map(JSONValue.string))
        } else {
            guard envelope["kind"] == .string("table"),case .array(let cells)=envelope["rows"],case .array(let schema)=envelope["columns"] else {throw Self.fail()}
            count=cells.count;columns=schema;lineage=try Self.object(envelope["lineage"] ?? .null)
            guard case .array(let values)=lineage["observationIDs"] else {throw Self.fail()}
            ids=values.compactMap(\.stringValue);guard ids.count==values.count,cells.allSatisfy({if case .array(let row)=$0{return row.count==schema.count};return false}) else {throw Self.fail()}
            let types=try columns.map {column -> String in
                let value=try Self.object(column)
                guard let type=value["dtype"]?.stringValue else {throw Self.fail()};return type
            }
            let version=types.contains(where:{["audio","tensor"].contains($0)}) ? 5:types.contains("int64Sequence") ? 4:types.contains(where:TemporalColumnContract.types.contains) ? 3:types.contains("image") ? 2:1
            guard envelope["schemaVersion"] == .integer(Int64(version)) else {throw Self.fail()}
        }
        guard (1...100_000).contains(count),!columns.isEmpty,columns.count<=5_000_000/count,
              ids.count==count,Set(ids).count==count,ids.allSatisfy({!$0.isEmpty && $0.unicodeScalars.count<=200}),
              ["sourceDigest","partitionDigest","membershipDigest","dataVersion"].allSatisfy({lineage[$0]?.stringValue.map(WorkflowArchiveCodec.isHash)==true}) else {throw Self.fail()}
        let schema=try columns.map{try Self.object($0,fields:["name","dtype","nullable"])}
        let names=schema.compactMap{$0["name"]?.stringValue}
        let supported=Set(["string","int64","float64","bool","image","int64Sequence","audio","tensor"]).union(TemporalColumnContract.types)
        guard names.count==schema.count,Set(names).count==names.count,names.allSatisfy({!$0.isEmpty}),
              schema.allSatisfy({column in
                  guard let type=column["dtype"]?.stringValue,supported.contains(type),case .bool=column["nullable"] else {return false};return true
              }),let result=try JSONSerialization.jsonObject(with:JSONEncoder().encode(lineage)) as? [String:Any] else {throw Self.fail()}
        return result
    }
}
