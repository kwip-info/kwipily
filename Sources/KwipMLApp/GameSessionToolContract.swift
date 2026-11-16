import Foundation
import KwipMLCore

enum GameSessionToolContract {
    static func fileLimit(path:String)throws->Int {
        switch path {case "input/session.json":return 2_000_000;case "input/A.pt","input/B.pt":return 100_000_000;default:throw WorkflowFileLeaseError.unsafePath}
    }
    static func validate(arguments:[String],files:[WorkflowArchiveFile],directory:URL?)throws {
        guard arguments.count==2,arguments[0]=="--game-session",arguments[1].hasPrefix("/") else {throw WorkflowFileLeaseError.unsafePath}
        let root=directory ?? URL(fileURLWithPath:arguments[1]).deletingLastPathComponent().deletingLastPathComponent()
        guard arguments[1]==root.appendingPathComponent("input/session.json").path,(1...3).contains(files.count),Set(files.map(\.path)).count==files.count,files.contains(where:{$0.path=="input/session.json"}) else {throw WorkflowFileLeaseError.unsafePath}
        for file in files {guard file.bytes>0,file.bytes<=((try fileLimit(path:file.path))),WorkflowArchiveCodec.isHash(file.sha256) else {throw WorkflowFileLeaseError.unsafePath}}
    }
    static func verifyInputs(files:[WorkflowArchiveFile],directory:URL)throws {
        for file in files {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:fileLimit(path:file.path))}
        let raw=try DiagnosticFileIO.read(root:directory,parts:["input","session.json"],limit:2_000_000)
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:raw)
        guard Set(inner.keys)==["schemaVersion","definition","configuration","policies","validateOnly"],inner["schemaVersion"] == 1,case .object(let config)=inner["configuration"],case .object(let policies)=inner["policies"],case .object(let rules)=inner["definition"],case .bool=inner["validateOnly"] else {throw WorkflowFileLeaseError.unsafePath}
        _=try GameSessionConfiguration(config);try GameRuleConfiguration.validate(["definition":.object(rules)])
        var expected:Set<String>=["input/session.json"]
        for (role,value) in policies {
            guard ["A","B"].contains(role),case .object(let p)=value,Set(p.keys).isSubset(of:["graph","initialization","sourceIdentity","checkpointSHA256"]),p["graph"] != nil,let identity=p["sourceIdentity"]?.stringValue,WorkflowArchiveCodec.isHash(identity) else {throw WorkflowFileLeaseError.unsafePath}
            if let pin=p["checkpointSHA256"]?.stringValue {let name="input/"+role+".pt";guard files.contains(where:{$0.path==name && $0.sha256==pin}) else {throw WorkflowFileLeaseError.unsafePath};expected.insert(name)}
        }
        guard Set(files.map(\.path))==expected else {throw WorkflowFileLeaseError.unsafePath}
    }
    nonisolated static func verifyResult(directory:URL,request:WorkflowToolRequest,completion:Data,window:ExecutionBudgetWindow)throws->URL {
        try Task.checkCancellation();_=try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
        try request.validate(directory:directory,verifyInputs:true)
        let event=try JSONDecoder().decode([String:JSONValue].self,from:completion)
        guard event["event"] == .string("gameSessionCompleted"),event["operationID"] == .string(request.operationID),event["requestIdentity"] == .string(request.requestIdentity),case .array(let entries)=event["files"] else {throw WorkflowFileLeaseError.unsafePath}
        let files=try entries.map{try JSONDecoder().decode(WorkflowArchiveFile.self,from:JSONEncoder().encode($0))}
        guard files.count==4,Set(files.map(\.path))==["artifacts/session.json","artifacts/summary.table.json","artifacts/episodes.table.json","artifacts/moves.table.json"],files.reduce(0,{$0+$1.bytes})<=8_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        for file in files {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:8_000_000)}
        let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["artifacts","session.json"],limit:8_000_000))
        guard result["schemaVersion"] == 1,result["kind"] == .string("kwip.game-session.v1"),case .object=result["identity"],case .array=result["episodes"],case .array=result["moves"] else {throw WorkflowFileLeaseError.unsafePath}
        var ports:[String:JSONValue]=[:]
        for port in ["summary","episodes","moves"] {
            let table=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["artifacts",port+".table.json"],limit:8_000_000))
            guard table["kind"] == .string("table"),case .array(let columns)=table["columns"],case .array(let rows)=table["rows"] else {throw WorkflowFileLeaseError.unsafePath}
            let sample=rows.prefix(8).map{value->JSONValue in guard case .array(let cells)=value else{return .array([])};return .array(cells.map{if case .string(let text)=$0{return .string(String(text.prefix(200)))};return $0})}
            ports[port] = .object(["kind":.string("table"),"preview":.object(["columns":.array(columns),"rows":.array(sample),"rowCount":.integer(Int64(rows.count))])])
        }
        let admission:[String:JSONValue]=["summary":result["summary"] ?? .object([:]),"ports":.object(ports)]
        try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(admission),root:directory,parts:["artifacts","admission.json"],limit:64_000)
        try DiagnosticFileIO.writeImmutable(completion,root:directory,parts:["game-session-outcome.json"],limit:1_000_000)
        return directory.appendingPathComponent("artifacts")
    }
}
