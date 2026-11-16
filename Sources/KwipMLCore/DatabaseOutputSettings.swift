import Foundation

/// Optional Table Output destination. Legacy graphs omit it and remain file writers.
public enum DatabaseOutputSettings {
    public static let types:[String:[String]]=["sqlite":["INTEGER","REAL","TEXT"],"postgresql":["BIGINT","DOUBLE PRECISION","TEXT","BOOLEAN","DATE","TIMESTAMPTZ","TIME"],"mysql":["BIGINT","DOUBLE","VARCHAR(255)","TEXT","BOOLEAN","DATE","DATETIME(6)","TIME(6)"]]
    public static let defaults:[String:JSONValue]=["schemaVersion":1,"backend":.string("sqlite"),"host":.string("localhost"),"port":5432,"database":.string(""),"schema":.string("main"),"table":.string(""),"tlsMode":.string("verifyFull"),"mode":.string("append"),"batchRows":2048,"timeoutSeconds":30,"mapping":.array([]),"keys":.array([])]
    public static func isDatabase(_ config:[String:JSONValue])->Bool {config["destination"] == .string("database")}
    public static func requiresLocalFolder(_ config:[String:JSONValue])->Bool {
        if APITableSettings.isAPI(config) {return false}
        guard isDatabase(config) else {return true}
        guard case .object(let db)=config["database"] else {return false}
        return (db["backend"] ?? .string("sqlite")) == .string("sqlite")
    }
    public static func normalized(_ input:[String:JSONValue],requireMapping:Bool=true)throws->[String:JSONValue] {
        guard Set(input.keys).isSubset(of:Set(defaults.keys)) else {throw DataFlowContractError("Unsupported database output setting.")}
        var c=defaults.merging(input){_,v in v}
        guard c["schemaVersion"] == .integer(1),let backend=c["backend"]?.stringValue,types[backend] != nil,["create","append","upsert"].contains(c["mode"]?.stringValue ?? "") else {throw DataFlowContractError("Choose a supported engine and write mode.")}
        for (key,bounds) in [("batchRows",1...8192),("timeoutSeconds",1...300)] {
            guard case .integer(let n)=c[key],bounds.contains(Int(n)) else {throw DataFlowContractError("Database batch/timeout is outside its supported range.")}
        }
        var endpoint=c.filter {Set(["backend","host","port","database","schema","table","tlsMode","timeoutSeconds"]).contains($0.key)}
        if endpoint["table"] == .string("") && !requireMapping {endpoint["table"] = .string("_metadata")}
        _ = try DatabaseSourceSettings.normalized(endpoint)
        guard case .array(let mapping)=c["mapping"],mapping.count<=128,!requireMapping || !mapping.isEmpty else {throw DataFlowContractError("Include at least one mapped source column.")}
        var sources=Set<String>(),targets=Set<String>()
        for item in mapping {
            guard case .object(let row)=item,Set(row.keys)==["source","target","type","nullable"],let source=row["source"]?.stringValue,let target=row["target"]?.stringValue,[source,target].allSatisfy({!$0.isEmpty && !$0.contains("\0") && $0.utf8.count<=200}),sources.insert(source).inserted,targets.insert(backend == "postgresql" ? target:target.lowercased()).inserted,types[backend]!.contains(row["type"]?.stringValue ?? ""),case .bool = row["nullable"] else {throw DataFlowContractError("Map distinct known source/destination names and supported types.")}
        }
        for identifier in ([c["schema"]?.stringValue ?? "",c["table"]?.stringValue ?? ""]+mapping.compactMap {if case .object(let r)=$0 {return r["target"]?.stringValue};return nil}) {
            guard backend != "postgresql" || identifier.utf8.count<=63, backend != "mysql" || identifier.count<=64 else {throw DataFlowContractError("Destination identifier exceeds the engine limit.")}
        }
        guard case .array(let rawKeys)=c["keys"],rawKeys.count<=128,rawKeys.allSatisfy({$0.stringValue != nil}),Set(rawKeys.compactMap(\.stringValue)).count==rawKeys.count else {throw DataFlowContractError("Choose a complete unique key.")}
        let keys=Set(rawKeys.compactMap(\.stringValue))
        guard !requireMapping || keys.isSubset(of:Set(mapping.compactMap {if case .object(let r)=$0 {return r["target"]?.stringValue};return nil})),!requireMapping || c["mode"] != .string("upsert") || !keys.isEmpty else {throw DataFlowContractError("Map every member of an existing unique key for Upsert.")}
        if backend=="postgresql",c["schema"] == .string("main") {c["schema"] = .string("public")}
        if backend=="mysql" {c["schema"]=c["database"]}
        return c
    }
    public static func suggestedType(_ dtype:String,backend:String)->String {
        if ["int64","int32","int16","uint8"].contains(dtype) {return backend=="sqlite" ? "INTEGER":"BIGINT"}
        if ["float32","float64"].contains(dtype) {return backend=="sqlite" ? "REAL":backend=="mysql" ? "DOUBLE":"DOUBLE PRECISION"}
        if dtype=="bool" {return backend=="sqlite" ? "INTEGER":"BOOLEAN"}
        if dtype=="date",backend != "sqlite" {return "DATE"}
        if dtype=="datetime",backend != "sqlite" {return backend=="postgresql" ? "TIMESTAMPTZ":"DATETIME(6)"}
        if dtype=="time",backend != "sqlite" {return backend=="postgresql" ? "TIME":"TIME(6)"}
        return "TEXT"
    }
}
