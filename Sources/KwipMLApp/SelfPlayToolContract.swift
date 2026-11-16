import Foundation
import KwipMLCore

enum SelfPlayToolContract {
    static let tablePorts=SelfPlayConfiguration.baseTablePorts
    static func tablePorts(document:WorkflowDocument,address:WorkflowNodeAddress)->[String] {guard let node=document.nodes.first(where:{$0.address==address}),case .train(let raw)=node.payload,let c=try? SelfPlayConfiguration(raw)else{return tablePorts};return SelfPlayConfiguration.tablePorts(c.values)+(document.nodes.contains(where:{if case .modelPrimitive("mixture_experts",_,_)=$0.payload{return true};return false}) ? ["routing","routingSample"]:[])}
    static func replayLimit(_ path:String)throws->Int {
        if path=="latest.json" {return 128}
        if path.range(of:"^chunks/[a-f0-9]{64}\\.jsonl$",options:.regularExpression) != nil {return 4_000_000}
        if path.range(of:"^checkpoints/iteration-[0-9]{2}/(model|optimizer|rng)\\.pt$",options:.regularExpression) != nil {return 100_000_000}
        if path.range(of:"^checkpoints/iteration-[0-9]{2}/manifest\\.json$",options:.regularExpression) != nil {return 2_000_000}
        if path.range(of:"^checkpoints/iteration-[0-9]{2}/PINNED$",options:.regularExpression) != nil {return 64}
        throw WorkflowFileLeaseError.unsafePath
    }
    static func fileLimit(path:String)throws->Int {
        if ["input/self-play.json","input/document.json","input/campaign-grant.json"].contains(path) {return 2_000_000}
        if path.range(of:"^input/opponents/[A-F0-9-]{36}/(model.pt|model.json|identity.json)$",options:.regularExpression) != nil {return path.hasSuffix("model.pt") ? 100000000:2000000}
        guard path.hasPrefix("input/resume/") else {throw WorkflowFileLeaseError.unsafePath}
        return try replayLimit(String(path.dropFirst("input/resume/".count)))
    }
    static func validate(arguments:[String],files:[WorkflowArchiveFile],directory:URL?)throws {
        guard arguments.count==2,arguments[0]=="--self-play",arguments[1].hasPrefix("/") else {throw WorkflowFileLeaseError.unsafePath}
        let root=directory ?? URL(fileURLWithPath:arguments[1]).deletingLastPathComponent().deletingLastPathComponent()
        guard arguments[1]==root.appendingPathComponent("input/self-play.json").path,(1...600).contains(files.count),Set(files.map(\.path)).count==files.count,files.contains(where:{$0.path=="input/self-play.json"}),files.reduce(Int64(0),{$0+Int64($1.bytes)})<=1_000_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        for f in files {guard f.bytes>0,f.bytes<=((try fileLimit(path:f.path))),WorkflowArchiveCodec.isHash(f.sha256) else {throw WorkflowFileLeaseError.unsafePath}}
    }
    static func verifyInputs(files:[WorkflowArchiveFile],directory:URL)throws {
        for f in files {try WorkflowPreviewToolContract.verifyFile(file:f,directory:directory,limit:fileLimit(path:f.path))}
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["input","self-play.json"],limit:2_000_000))
        guard Set(inner.keys)==["schemaVersion","definition","graph","configuration","initialization","trainability","documentSHA256","trainAddress","parentOperationID","resume","validateOnly"],inner["schemaVersion"] == 1,case .object(let config)=inner["configuration"],case .object(let definition)=inner["definition"],case .object=inner["graph"],case .bool(let resume)=inner["resume"],case .bool=inner["validateOnly"],resume==files.contains(where:{$0.path=="input/resume/latest.json"}) else {throw WorkflowFileLeaseError.unsafePath}
        _=try SelfPlayConfiguration(["selfPlay":.object(config)]);try GameRuleConfiguration.validate(["definition":.object(definition)])
        guard let pin=inner["documentSHA256"]?.stringValue,files.contains(where:{$0.path=="input/document.json" && $0.sha256==pin})else{throw WorkflowFileLeaseError.unsafePath}
        guard resume || files.allSatisfy({["input/self-play.json","input/document.json","input/campaign-grant.json"].contains($0.path) || $0.path.hasPrefix("input/opponents/")}) else {throw WorkflowFileLeaseError.unsafePath}
    }
    nonisolated static func verifyResult(directory:URL,request:WorkflowToolRequest,completion:Data,window:ExecutionBudgetWindow)throws->URL {
        try Task.checkCancellation();_=try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime);try request.validate(directory:directory,verifyInputs:true)
        let event=try JSONDecoder().decode([String:JSONValue].self,from:completion)
        guard event["event"] == .string("selfPlayCompleted"),event["operationID"] == .string(request.operationID),event["requestIdentity"] == .string(request.requestIdentity),case .array(let entries)=event["files"] else {throw WorkflowFileLeaseError.unsafePath}
        let files=try entries.map{try JSONDecoder().decode(WorkflowArchiveFile.self,from:JSONEncoder().encode($0))}
        let captured=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["input","self-play.json"],limit:2000000))
        guard case .object(let settings)=captured["configuration"]else{throw WorkflowFileLeaseError.unsafePath}
        let doc=try WorkflowCodec.decode(DiagnosticFileIO.read(root:directory,parts:["input","document.json"],limit:2_000_000))
        let outputPorts=SelfPlayConfiguration.tablePorts(settings)+(doc.nodes.contains(where:{if case .modelPrimitive("mixture_experts",_,_)=$0.payload{return true};return false}) ? ["routing","routingSample"]:[])
        let required=Set(["artifacts/self-play.json"]+outputPorts.map{"artifacts/"+$0+".table.json"})
        guard required.isSubset(of:Set(files.map(\.path))),Set(files.map(\.path)).count==files.count,files.count<=605,files.reduce(Int64(0),{$0+Int64($1.bytes)})<=1_000_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        var tablesBytes=0
        for f in files {
            let limit:Int
            if ["artifacts/model.pt","artifacts/model.py","artifacts/model.json","artifacts/inference-manifest.json"].contains(f.path) {limit=f.path=="artifacts/model.pt" ? 100_000_000:2_000_000}
            else if required.contains(f.path) {limit=8_000_000;tablesBytes+=f.bytes}
            else {guard f.path.hasPrefix("artifacts/replay/")else{throw WorkflowFileLeaseError.unsafePath};limit=try replayLimit(String(f.path.dropFirst("artifacts/replay/".count)))}
            try WorkflowPreviewToolContract.verifyFile(file:f,directory:directory,limit:limit)
        }
        guard tablesBytes<=8_000_000 else {throw WorkflowFileLeaseError.unsafePath}
        let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["artifacts","self-play.json"],limit:8_000_000))
        guard result["kind"] == .string("kwip.self-play.v1"),case .object(let summary)=result["summary"] else {throw WorkflowFileLeaseError.unsafePath}
        var ports:[String:JSONValue]=["model":.object(["kind":.string("trainedModel"),"task":.string("gamePolicyValue")])]
        for port in outputPorts {
            let table=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["artifacts",port+".table.json"],limit:8_000_000))
            guard case .array(let rows)=table["rows"],case .array(let columns)=table["columns"] else {throw WorkflowFileLeaseError.unsafePath}
            ports[port] = .object(["kind":.string("table"),"preview":.object(["columns":.array(columns),"rows":.array(Array(rows.prefix(4))),"rowCount":.integer(Int64(rows.count))])])
        }
        let admission:[String:JSONValue]=["summary":.object(summary),"ports":.object(ports)]
        try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(admission),root:directory,parts:["artifacts","admission.json"],limit:64_000)
        try DiagnosticFileIO.writeImmutable(completion,root:directory,parts:["self-play-outcome.json"],limit:1_000_000)
        return directory.appendingPathComponent("artifacts")
    }
}
