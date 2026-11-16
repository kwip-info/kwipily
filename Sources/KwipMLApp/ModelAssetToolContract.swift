import Foundation
import KwipMLCore

enum ModelAssetToolContract {
    static func isSegment(_ path:String)->Bool {path.hasPrefix("input/") && PreparationArtifactPolicy.isSegmentName(String(path.dropFirst(6)))}
    static func limit(_ path:String)throws->Int {
        if isSegment(path) {return TableTransferLimits.artifactBytes}
        if ["input/training.table.json","input/validation.table.json","input/evaluation.table.json","input/evaluationPopulation.table.json"].contains(path) {return TableTransferLimits.artifactBytes}
        if path=="input/model-request.json" || ["input/prompts.table.json","input/training.table.json","input/validation.table.json"].contains(path) {return 8_000_000}
        if path.hasPrefix("input/adapter/"),ModelAdapterFiles.names.contains(String(path.dropFirst(14))) {return path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000}
        if ModelFineTuningFiles.resumeInput(path) {return path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000}
        guard path.hasPrefix("source/"),path.dropFirst(7)=="kwip-manifest.json" || ModelHubRequest.allowedAssetFile(String(path.dropFirst(7))) else {throw WorkflowFileLeaseError.unsafePath}
        return path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000
    }
    static func validate(arguments:[String],files:[WorkflowArchiveFile],directory:URL?)throws {
        guard arguments.count==2,arguments[0]=="--model-asset",(4...512).contains(files.count),Set(files.map(\.path)).count==files.count else {throw WorkflowFileLeaseError.unsafePath}
        let path=arguments[1],root=directory ?? URL(fileURLWithPath:path).deletingLastPathComponent().deletingLastPathComponent()
        guard path==root.appendingPathComponent("input/model-request.json").path,root.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent=="workflow-operations",UUID(uuidString:root.lastPathComponent)?.uuidString==root.lastPathComponent,UUID(uuidString:root.deletingLastPathComponent().lastPathComponent)?.uuidString==root.deletingLastPathComponent().lastPathComponent else {throw WorkflowFileLeaseError.unsafePath}
        var total=0
        for f in files {guard f.bytes>0,f.bytes <= (try limit(f.path)),WorkflowArchiveCodec.isHash(f.sha256) else {throw WorkflowFileLeaseError.unsafePath};total+=f.bytes}
        guard total<=4_120_000_000,Set(["input/model-request.json"]).isSubset(of:Set(files.map(\.path))) && ((Set(["source/config.json"]).isSubset(of:Set(files.map(\.path))) && !Set(["source/tokenizer.json","source/preprocessor_config.json"]).isDisjoint(with:Set(files.map(\.path)))) || Set(["source/rl_agent_config.json","source/encoder/config.json","source/tokenizer/tokenizer.json","source/tokenizer/tokenizer_config.json"]).isSubset(of:Set(files.map(\.path)))),files.contains(where:{$0.path.hasSuffix(".safetensors")}) else {throw WorkflowFileLeaseError.unsafePath}
    }
    static func verify(_ request:WorkflowToolRequest,directory:URL)throws {
        try validate(arguments:request.arguments,files:request.inputFiles,directory:directory)
        for file in request.inputFiles {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:limit(file.path))}
        let sourceConfig=try JSONDecoder().decode([String:JSONValue].self,from:WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("source"),name:request.inputFiles.contains(where:{$0.path=="source/rl_agent_config.json"}) ? "rl_agent_config.json":"config.json",limit:8_000_000))
        guard ModelAssetFiles.requiredSourceFiles(sourceConfig).map({"source/"+$0}).allSatisfy({name in request.inputFiles.contains{$0.path==name}}) else {throw WorkflowFileLeaseError.unsafePath}
        try verifyCaptureVersion(request,directory:directory)
    }
    /// Called after the captured inner request itself has passed its file pin.
    static func verifyCaptureVersion(_ request:WorkflowToolRequest,directory:URL)throws {
        let value=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["input","model-request.json"],limit:2_000_000))
        guard Set(value.keys)==["schemaVersion","projectID","operationID","requestIdentity","action","settings","sourceIdentity"],[JSONValue.integer(1),.integer(2),.integer(3),.integer(4)].contains(value["schemaVersion"] ?? .null),value["projectID"] == .string(request.projectID),value["operationID"] == .string(request.operationID),value["requestIdentity"] == .string(request.requestIdentity),let action=value["action"]?.stringValue,["inspect","tokenize","generate","export","train","attachAdapter","prepare","evaluate"].contains(action),case .object=value["settings"],let identity=value["sourceIdentity"]?.stringValue,WorkflowArchiveCodec.isHash(identity) else {throw WorkflowFileLeaseError.unsafePath}
        // Temporary worker-source snapshots are removed after exit. Cold receipts
        // verify their pinned original base separately, never reopen expired grants.
        let selectedNames=Set(request.inputFiles.map(\.path))
        if selectedNames.contains("source/rl_agent_config.json") {
            guard Set(["source/rl_agent_config.json","source/encoder/config.json","source/tokenizer/tokenizer.json","source/tokenizer/tokenizer_config.json","source/model.safetensors"]).isSubset(of:selectedNames) else {throw WorkflowFileLeaseError.unsafePath}
        }
        let names=Set(request.inputFiles.map(\.path)),allTables:Set<String>=["input/training.table.json","input/validation.table.json","input/evaluation.table.json","input/evaluationPopulation.table.json","input/prompts.table.json"]
        let expected:Set<String>=action=="evaluate" ? ["input/training.table.json","input/evaluation.table.json","input/evaluationPopulation.table.json"]:["train","prepare"].contains(action) ? ["input/training.table.json","input/validation.table.json"]:action=="generate" ? ["input/prompts.table.json"]:[]
        guard names.intersection(allTables).isSubset(of:expected)else{throw WorkflowFileLeaseError.unsafePath}
        guard case .object(let settings)=value["settings"]else{throw WorkflowFileLeaseError.unsafePath}
        let resumed=names.filter{$0.hasPrefix("input/resume/")}
        guard resumed.isEmpty && settings["resumeMode"]==nil || [JSONValue.integer(3),.integer(4)].contains(value["schemaVersion"] ?? .null) && action=="train" && settings["resumeMode"] == .string("exact") && Set(["input/resume/training.json","input/resume/result-manifest.json","input/resume/latest/kwip-continuation.json"]).isSubset(of:resumed) && !resumed.contains(where:{$0.contains("/model/")})else{throw WorkflowFileLeaseError.unsafePath}
        guard action=="attachAdapter" || !names.contains(where:{$0.hasPrefix("input/adapter/")})else{throw WorkflowFileLeaseError.unsafePath}
        let segments=names.filter {isSegment($0)}
        guard segments.isEmpty || value["schemaVersion"] == .integer(4) else {throw WorkflowFileLeaseError.unsafePath}
        if value["schemaVersion"] == .integer(4) {guard !segments.isEmpty else {throw WorkflowFileLeaseError.unsafePath}}
        if value["schemaVersion"] == .integer(1) {
            guard request.inputFiles.reduce(0,{$0+$1.bytes})<=4_020_000_000,request.inputFiles.allSatisfy({$0.path.hasSuffix(".safetensors") || $0.bytes<=8_000_000}) else {throw WorkflowFileLeaseError.unsafePath}
        } else if [JSONValue.integer(3),.integer(4)].contains(value["schemaVersion"] ?? .null) {
            let names=Set(request.inputFiles.map(\.path))
            guard ["train","prepare","evaluate"].contains(action),!names.contains("input/prompts.table.json") else{throw WorkflowFileLeaseError.unsafePath}
            if action=="evaluate" {guard Set(["input/training.table.json","input/evaluation.table.json","input/evaluationPopulation.table.json"]).isSubset(of:names) else{throw WorkflowFileLeaseError.unsafePath}}
            else {guard Set(["input/training.table.json","input/validation.table.json"]).isSubset(of:names) else{throw WorkflowFileLeaseError.unsafePath}}
        } else {
            guard action=="train",Set(["input/training.table.json","input/validation.table.json"]).isSubset(of:Set(request.inputFiles.map(\.path))),!request.inputFiles.contains(where:{$0.path=="input/prompts.table.json"}) else {throw WorkflowFileLeaseError.unsafePath}
        }
    }
}
