import Foundation
import KwipMLCore

enum WorkflowGenerationToolContract {
    static let sourceNames:Set<String>=["model.py","checkpoint.pt","inference.py","input-contract.json","README.md","requirements.txt","training_recipe.py","rate-trace.json","objective_recipe.py","initialization_recipe.py","gradient_policy.py","trainability_recipe.py","data_loading_recipe.py","named_inputs.py","tokenizers.json","portable_text_encoding.py"]
    static func fileLimit(_ path:String)throws->Int {
        if path == "input/generation-kernel.py" {return 200_000}
        if path == "input/prompts.table.json" {return 8_000_000}
        if path == "input/generation.json" {return 2_000_000}
        if path == "source/export-manifest.json" {return 2_000_000}
        guard path.hasPrefix("source/"),sourceNames.contains(String(path.dropFirst(7))) else {throw WorkflowFileLeaseError.unsafePath}
        return path == "source/checkpoint.pt" ? 100_000_000:path == "source/rate-trace.json" ? 4_000_000:path == "source/data_loading_recipe.py" ? 131_072:2_000_000
    }
    static func validate(arguments:[String],files:[WorkflowArchiveFile],directory:URL?)throws {
        guard arguments.count == 2,arguments[0] == "--generate-text",(9...20).contains(files.count) else {throw WorkflowFileLeaseError.unsafePath}
        let path=arguments[1],root=directory ?? URL(fileURLWithPath:path).deletingLastPathComponent().deletingLastPathComponent()
        guard path.hasPrefix("/"),!path.contains("//"),!path.split(separator:"/").contains(where:{$0 == "." || $0 == ".."}),path == root.appendingPathComponent("input/generation.json").path,root.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations",UUID(uuidString:root.lastPathComponent)?.uuidString == root.lastPathComponent,UUID(uuidString:root.deletingLastPathComponent().lastPathComponent)?.uuidString == root.deletingLastPathComponent().lastPathComponent else {throw WorkflowFileLeaseError.unsafePath}
        var names=Set<String>(),total=0
        for file in files {
            guard names.insert(file.path).inserted,file.bytes > 0,file.bytes <= (try fileLimit(file.path)),WorkflowArchiveCodec.isHash(file.sha256) else {throw WorkflowFileLeaseError.unsafePath}
            total += file.bytes
        }
        guard total <= 140_000_000,Set(["input/generation.json","source/export-manifest.json","source/model.py","source/checkpoint.pt","source/inference.py","source/input-contract.json"]).isSubset(of:names) else {throw WorkflowFileLeaseError.unsafePath}
    }
    static func verifyInputs(request:WorkflowToolRequest,directory:URL)throws {
        try validate(arguments:request.arguments,files:request.inputFiles,directory:directory)
        for file in request.inputFiles {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:fileLimit(file.path))}
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["input","generation.json"],limit:2_000_000))
        guard inner["schemaVersion"] == .integer(1),inner["projectID"] == .string(request.projectID),inner["operationID"] == .string(request.operationID),inner["requestIdentity"] == .string(request.requestIdentity),inner["sourceManifestSHA256"]?.stringValue == request.inputFiles.first(where:{$0.path == "source/export-manifest.json"})?.sha256 else {throw WorkflowFileLeaseError.unsafePath}
        struct Source:Decodable {let schemaVersion:Int;let kind:String;let files:[WorkflowArchiveFile]}
        let source=try JSONDecoder().decode(Source.self,from:DiagnosticFileIO.read(root:directory,parts:["source","export-manifest.json"],limit:2_000_000))
        let expected=source.files.map{WorkflowArchiveFile(path:"source/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}
        guard source.schemaVersion == 1,source.kind == "kwipmlTrainedModel",Set(expected.map(\.path)).count==expected.count,expected.allSatisfy({request.inputFiles.contains($0)}),request.inputFiles.count==expected.count+3+(request.inputFiles.contains(where:{$0.path=="input/prompts.table.json"}) ? 1:0) else {throw WorkflowFileLeaseError.unsafePath}
    }
}
