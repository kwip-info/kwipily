import Foundation

/// Versioned JSON REST settings; secrets and origin grants remain private.
public enum APITableSettings {
    public static let defaults:[String:JSONValue] = try! JSONDecoder().decode([String:JSONValue].self,from:Data(#"{"schemaVersion":1,"kind":"api","direction":"input","url":"","method":"GET","auth":"none","credentialHeader":"X-API-Key","query":{},"headers":{},"allowLoopback":false,"recordsPath":"","mapping":[],"missing":"error","pagination":"none","cursorPath":"/next_cursor","cursorParameter":"cursor","pageParameter":"page","pageStart":1,"pageSize":100,"nextPath":"/next","maximumPages":100,"maximumBytes":10000000,"maximumRows":100000,"timeoutSeconds":120,"requestTimeout":30,"retries":2,"requestsPerSecond":5,"delivery":"row","batchRows":100,"arrayPath":"","maximumRequests":256,"idempotencyHeader":"","responsePath":"","constants":[]}"#.utf8))
    public static let limits:[(String,Int,Int)] = [("schemaVersion",1,1),("maximumPages",1,1000),("maximumBytes",1024,50_000_000),("maximumRows",1,500_000),("timeoutSeconds",1,300),("requestTimeout",1,120),("retries",0,3),("requestsPerSecond",1,50),("batchRows",1,8192),("maximumRequests",1,256),("pageSize",1,8192),("pageStart",0,1_000_000)]
    public static let choices:[(String,[String])] = [("kind",["api"]),("direction",["input","output"]),("auth",["none","bearer","basic","apiKey"]),("missing",["error","null"]),("pagination",["none","cursor","page","offset","nextLink"]),("delivery",["row","batch"])]
    public static let types=["string","int64","float64","bool","datetime","json"]
    public static func isAPI(_ c:[String:JSONValue])->Bool {c["destination"] == .string("api")}
    public static func isExternal(_ c:[String:JSONValue])->Bool {isAPI(c) || DatabaseOutputSettings.isDatabase(c)}
    public static func sensitive(_ name:String)->Bool {["authorization","cookie","token","password","secret","api-key","apikey","api_key","access_key","signature"].contains {name.lowercased().contains($0)}}
    public static func origin(_ c:[String:JSONValue])throws->String {
        guard let text=c["url"]?.stringValue,let u=URLComponents(string:text),let scheme=u.scheme,let host=u.host,scheme=="https" || scheme=="http",u.user==nil,u.password==nil,u.fragment==nil,u.query==nil else {throw DataFlowContractError("Enter an HTTPS API URL without credentials, query or fragment. Use the query grid.",apiIssue:.url)}
        let loopback=host=="127.0.0.1" || host=="[::1]" || host=="::1"
        guard scheme=="https" || loopback && c["allowLoopback"] == .bool(true),!host.isEmpty,text.utf8.count<=2048 else {throw DataFlowContractError("HTTP requires an explicitly enabled literal loopback address.")}
        return scheme+"://"+host.lowercased()+":"+String(u.port ?? (scheme=="https" ? 443:80))
    }
    public static func normalized(_ raw:[String:JSONValue],requireMapping:Bool=true)throws->[String:JSONValue] {
        guard Set(raw.keys).isSubset(of:Set(defaults.keys)) else {throw DataFlowContractError("Unknown API configuration field.",apiIssue:.fields)}
        let c=defaults.merging(raw){_,v in v};_ = try origin(c)
        for (key,low,high) in limits {
            guard case .integer(let v)=c[key],v>=low,v<=high else {throw DataFlowContractError("API limit outside supported range: "+key)}
        }
        guard case .bool = c["allowLoopback"] else {throw DataFlowContractError("Invalid API transport permission.")}
        for (key,values) in choices {
            guard let v=c[key]?.stringValue,values.contains(v) else {throw DataFlowContractError("Choose a supported API "+key+".")}
        }
        guard let method=c["method"]?.stringValue,(c["direction"] == .string("input") ? ["GET"]:["POST","PUT","PATCH","DELETE"]).contains(method) else {throw DataFlowContractError("Choose a supported request method.")}
        for key in ["headers","query"] {
            guard case .object(let fields)=c[key],fields.count<=32 else {throw DataFlowContractError("Use at most32 request fields.",apiIssue:.requestFields)}
            for (name,value) in fields {
                guard !name.isEmpty,name.utf8.count<=128,!sensitive(name),let v=value.stringValue,v.utf8.count<=1024,!(name+v).contains(where:{"\r\n\0".contains($0)}) else {throw DataFlowContractError("Secrets belong in private credentials; use bounded request fields.",apiIssue:.requestFields)}
                if key=="headers" {guard name.allSatisfy({$0.isASCII && ($0.isLetter || $0.isNumber || $0=="-")}),!["host","content-length","content-type","connection","transfer-encoding","expect","upgrade","accept-encoding"].contains(name.lowercased()) else {throw DataFlowContractError("Reserved HTTP header.",apiIssue:.requestFields)}}
            }
        }
        for key in ["recordsPath","cursorPath","nextPath","arrayPath","responsePath"] {guard let p=c[key]?.stringValue,p.utf8.count<=1024,!p.contains("\0"),p.isEmpty || p.hasPrefix("/") else {throw DataFlowContractError("Use JSON Pointer paths such as /data/items.")}}
        for key in ["cursorParameter","pageParameter","credentialHeader","idempotencyHeader"] {guard let v=c[key]?.stringValue,v.utf8.count<=128,v.allSatisfy({$0.isASCII && ($0.isLetter || $0.isNumber || $0=="-" || $0=="_")}) else {throw DataFlowContractError("Use a simple request field name.")}}
        guard c["cursorParameter"] != .string(""),c["pageParameter"] != .string("") else {throw DataFlowContractError("Name the pagination query parameter.")}
        if c["auth"] == .string("apiKey") {guard let name=c["credentialHeader"]?.stringValue,!name.isEmpty,!["host","content-length","content-type","connection","transfer-encoding","cookie","authorization"].contains(name.lowercased()) else {throw DataFlowContractError("Choose a custom API credential header.")}}
        if let name=c["idempotencyHeader"]?.stringValue,!name.isEmpty {guard !sensitive(name),!["host","content-length","content-type","connection","transfer-encoding"].contains(name.lowercased()) else {throw DataFlowContractError("Choose an endpoint-supported idempotency header.")}}
        guard case .array(let rows)=c["mapping"],rows.count<=128,!requireMapping || !rows.isEmpty else {throw DataFlowContractError("Map at least one column.",apiIssue:.mapping)}
        var names=Set<String>(),sources=Set<String>()
        for item in rows {guard case .object(let r)=item,Set(r.keys)==["source","target","type"],let s=r["source"]?.stringValue,let t=r["target"]?.stringValue,[s,t].allSatisfy({!$0.isEmpty && $0.utf8.count<=256 && !$0.contains("\0")}),let dtype=r["type"]?.stringValue,types.contains(dtype),names.insert(t).inserted,sources.insert(s).inserted,(c["direction"] == .string("input") ? s:t).hasPrefix("/") else {throw DataFlowContractError("Map distinct source/destination fields with a supported type.",apiIssue:.mapping)}}
        if c["direction"] == .string("output") {for name in names {guard !names.contains(where:{$0 != name && ($0.hasPrefix(name+"/") || name.hasPrefix($0+"/"))}) else {throw DataFlowContractError("JSON fields cannot shadow their parent.",apiIssue:.mapping)}}}
        guard case .array(let constants)=c["constants"],constants.count<=32,c["direction"] == .string("output") || constants.isEmpty else {throw DataFlowContractError("Constants belong to output JSON payloads.")}
        for constant in constants {guard case .object(let row)=constant,Set(row.keys)==["target","json"],let target=row["target"]?.stringValue,target.hasPrefix("/"),target.utf8.count<=256,!names.contains(target),!names.contains(where:{$0.hasPrefix(target+"/") || target.hasPrefix($0+"/")}),let text=row["json"]?.stringValue,text.utf8.count<=1024,(try? JSONDecoder().decode(JSONValue.self,from:Data(text.utf8))) != nil else {throw DataFlowContractError("Choose distinct JSON destinations and a valid constant JSON value.")};names.insert(target)}
        return c
    }
}
