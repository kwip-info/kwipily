import Foundation
import KwipMLCore

/// Lossless navigation of authoring metadata. Large children are references, never
/// clipped values. Every continuation is bound to the canonical source revision.
enum MCPMetadataRead {
    private static func shallow(_ value:JSONValue,_ remaining:Int)->Bool {
        guard remaining>0 else {return false}
        switch value {
        case .object(let fields):return fields.values.allSatisfy{shallow($0,remaining-1)}
        case .array(let values):return values.allSatisfy{shallow($0,remaining-1)}
        default:return true
        }
    }
    static let argumentKeys = ["readPath", "readOffset", "readLimit", "readRevision"]

    static func response(_ source:[String:JSONValue], arguments:[String:JSONValue], byteLimit:Int) throws -> [String:JSONValue] {
        func refuse(_ code:String) throws -> Never { throw MCPSettings.Refusal(code:code) }
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        // Scrub before navigation: selecting a secret-shaped field must never
        // bypass the central key filter by returning it under a generic value key.
        let unsanitized=JSONValue.object(source)
        guard shallow(unsanitized,128) else {try refuse("metadata_depth_limit")}
        let root=MCPOutputBoundary.sanitize(unsanitized,maximumDepth:129)
        guard case .object(let sanitized)=root else {try refuse("invalid_metadata_read")}
        let encoded=try encoder.encode(root)
        let explicit=argumentKeys.contains{arguments[$0] != nil}
        if !explicit,encoded.count <= byteLimit,shallow(root,20) { return sanitized }
        let revision=WorkflowArchiveCodec.hash(encoded)
        if let expected=arguments["readRevision"],expected != .string(revision) { try refuse("metadata_read_stale") }
        var path:[String]=[]
        if let requested=arguments["readPath"] {
            guard case .array(let parts)=requested,parts.count<=128 else {try refuse("invalid_metadata_read")}
            for part in parts {guard let text=part.stringValue,text.utf8.count<=4096 else {try refuse("invalid_metadata_read")};path.append(text)}
        }
        func number(_ key:String,_ fallback:Int) throws -> Int {
            guard let value=arguments[key] else {return fallback}
            guard case .integer(let n)=value,n>=0,n<=Int64(Int.max) else {try refuse("invalid_metadata_read")};return Int(n)
        }
        let offset=try number("readOffset",0),limit=try number("readLimit",50)
        guard limit>0,limit<=4096 else {try refuse("invalid_metadata_read")}
        if offset>0,arguments["readRevision"] == nil {try refuse("metadata_revision_required")}
        var selected=root
        for part in path {
            switch selected {
            case .object(let fields):guard let value=fields[part] else {try refuse("metadata_path_unavailable")};selected=value
            case .array(let values):guard let index=Int(part),String(index)==part,values.indices.contains(index) else {try refuse("metadata_path_unavailable")};selected=values[index]
            default:try refuse("metadata_path_unavailable")
            }
        }
        func kind(_ value:JSONValue)->String {
            switch value {case .object:return "object";case .array:return "array";case .string:return "string";default:return "scalar"}
        }
        var page:[String:JSONValue] = ["readRevision":.string(revision),"readPath":.array(path.map(JSONValue.string)),"offset":.integer(Int64(offset)),"kind":.string(kind(selected))]
        var next:Int?,total=1
        switch selected {
        case .string(let text):
            let scalars=text.unicodeScalars;total=scalars.count
            guard offset<=total else {try refuse("invalid_metadata_read")}
            let chunk=scalars.dropFirst(offset).prefix(min(limit,4096))
            page["value"] = .string(String(String.UnicodeScalarView(chunk)))
            page["offsetUnit"] = .string("unicodeScalar")
            let end=offset+chunk.count;next=end<total ? end:nil
        case .object,.array:
            let values:[(String,JSONValue)]
            if case .object(let fields)=selected {values=fields.keys.sorted().map{($0,fields[$0]!)}}
            else if case .array(let items)=selected {values=items.enumerated().map{(String($0.offset),$0.element)}}
            else {values=[]}
            total=values.count;guard offset<=total else {try refuse("invalid_metadata_read")}
            var entries:[JSONValue]=[],used=0
            for (key,value) in values.dropFirst(offset).prefix(min(limit,100)) {
                var entry:[String:JSONValue]=["key":.string(key),"readPath":.array((path+[key]).map(JSONValue.string)),"kind":.string(kind(value))]
                // Bound each child independently so arbitrarily large configurations,
                // lesson blocks and graph nodes remain individually navigable.
                if shallow(value,20),try encoder.encode(value).count<=4096 {entry["value"]=value;entry["complete"]=true}
                else {entry["complete"]=false}
                let row=JSONValue.object(entry),bytes=try encoder.encode(row).count
                if used+bytes>max(4096,byteLimit-8192) {break}
                entries.append(row);used+=bytes
            }
            guard !entries.isEmpty || offset==total else {try refuse("metadata_entry_too_large")}
            page["entries"] = .array(entries)
            let end=offset+entries.count;next=end<total ? end:nil
        default:
            guard offset==0 else {try refuse("invalid_metadata_read")}
            page["value"]=selected
        }
        page["total"] = .integer(Int64(total));page["complete"] = .bool(next == nil)
        page["nextOffset"] = next.map{.integer(Int64($0))} ?? .null
        if let next {page["nextRead"] = .object(["readPath":.array(path.map(JSONValue.string)),"readOffset":.integer(Int64(next)),"readLimit":.integer(Int64(limit)),"readRevision":.string(revision)])}
        let result:[String:JSONValue] = ["metadataRead":.object(page)]
        guard try encoder.encode(result).count<=byteLimit else {try refuse("metadata_page_too_large")}
        return result
    }
}
