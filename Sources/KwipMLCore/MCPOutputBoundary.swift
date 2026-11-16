import Foundation

/// Observation limits apply after native capture, never by rerunning a source.
/// Returned strings and cells remain untrusted data. Private filesystem identities
/// and credential-shaped fields are not part of the MCP observation contract.
public enum MCPOutputBoundary {
    /// Fail explicitly rather than silently clipping schemas, graph state or receipts.
    public static func responseByteLimit(action:String)->Int {
        switch action {
        case "tools_list": return 300_000 // protocol schemas, never captured rows
        case "catalog_get", "workflow_inspect": return 65_536
        default: return 32_768
        }
    }

    public static func sanitize(_ value:JSONValue,depth:Int=0,truncateValues:Bool=false,maximumDepth:Int=32)->JSONValue {
        guard depth<maximumDepth else{return .string("[depth limit]")}
        switch value {
        case .object(let fields):
            let privateKeys:Set<String>=["password","token","accesstoken","refreshtoken","authorization","credential","credentials","bookmark","bookmarkdata","sourcepath","indexpath","indexurl","absolutepath","localpath"]
            return .object(fields.mapValues{sanitize($0,depth:depth+1,truncateValues:truncateValues,maximumDepth:maximumDepth)}.filter{!privateKeys.contains($0.key.lowercased())})
        case .array(let values):return .array((truncateValues ? Array(values.prefix(1000)):values).map{sanitize($0,depth:depth+1,truncateValues:truncateValues,maximumDepth:maximumDepth)})
        case .string(let text):
            if text.hasPrefix("file://") || text.hasPrefix("/Users/") || text.hasPrefix("/Volumes/") || text.hasPrefix("/private/"){return .string("[local path]")}
            return .string(truncateValues ? String(text.prefix(8192)):text)
        default:return value
        }
    }
}
