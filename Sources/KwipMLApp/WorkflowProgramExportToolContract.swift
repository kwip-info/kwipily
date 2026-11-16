import Foundation
import KwipMLCore

enum WorkflowProgramExportToolContract {
    static let sourceNames:Set<String>=["model.py","checkpoint.pt","inference.py","input-contract.json","README.md","requirements.txt","training_recipe.py","rate-trace.json","objective_recipe.py","initialization_recipe.py","gradient_policy.py","trainability_recipe.py","data_loading_recipe.py"]
    static func fileLimit(_ path:String)throws->Int {
        if path == "input/program-export.json" {return 262_144}
        if path == "source/export-manifest.json" {return 2_000_000}
        guard path.hasPrefix("source/"),sourceNames.contains(String(path.dropFirst(7))) else {throw WorkflowFileLeaseError.unsafePath}
        return path == "source/checkpoint.pt" ? 100_000_000:path == "source/rate-trace.json" ? 4_000_000:path == "source/data_loading_recipe.py" ? 131_072:2_000_000
    }
    static func validate(arguments:[String],files:[WorkflowArchiveFile],directory:URL?)throws {
        guard arguments.count == 4,arguments[0] == "--export-program",arguments[2] == "--program-request-sha256",WorkflowArchiveCodec.isHash(arguments[3]),(6...15).contains(files.count) else {throw WorkflowFileLeaseError.unsafePath}
        let path=arguments[1],root=directory ?? URL(fileURLWithPath:path).deletingLastPathComponent().deletingLastPathComponent()
        guard path.hasPrefix("/"),!path.contains("//"),!path.split(separator:"/").contains(where:{$0 == "." || $0 == ".."}),path == root.appendingPathComponent("input/program-export.json").path,root.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations",UUID(uuidString:root.lastPathComponent)?.uuidString == root.lastPathComponent,UUID(uuidString:root.deletingLastPathComponent().lastPathComponent)?.uuidString == root.deletingLastPathComponent().lastPathComponent else {throw WorkflowFileLeaseError.unsafePath}
        var names=Set<String>(),total=0
        for file in files {
            guard names.insert(file.path).inserted,file.bytes > 0,file.bytes <= (try fileLimit(file.path)),WorkflowArchiveCodec.isHash(file.sha256) else {throw WorkflowFileLeaseError.unsafePath}
            total += file.bytes
        }
        guard total <= 130_000_000,Set(["input/program-export.json","source/export-manifest.json","source/model.py","source/checkpoint.pt","source/inference.py","source/input-contract.json"]).isSubset(of:names),files.first(where:{$0.path == "input/program-export.json"})?.sha256 == arguments[3] else {throw WorkflowFileLeaseError.unsafePath}
    }
    static func verifyInputs(request:WorkflowToolRequest,directory:URL)throws {
        try validate(arguments:request.arguments,files:request.inputFiles,directory:directory)
        for file in request.inputFiles {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:fileLimit(file.path))}
        let inner=try WorkflowProgramExportRequest.decode(DiagnosticFileIO.read(root:directory,parts:["input","program-export.json"],limit:262_144))
        guard inner.projectID == request.projectID,inner.exportID == request.operationID,inner.requestIdentity == request.requestIdentity,inner.sourceDirectory == directory.appendingPathComponent("source").path,inner.executionBudget == nil || inner.executionBudget == request.executionBudget,inner.sourceManifestSHA256 == request.inputFiles.first(where:{$0.path == "source/export-manifest.json"})?.sha256 else {throw WorkflowFileLeaseError.unsafePath}
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["source","export-manifest.json"],limit:2_000_000)
        struct Source:Decodable {let schemaVersion:Int;let kind:String;let runID:String;let requestIdentity:String;let modelManifestSHA256:String;let files:[WorkflowArchiveFile]}
        let source=try JSONDecoder().decode(Source.self,from:bytes)
        guard source.schemaVersion == 1,source.kind == "kwipmlTrainedModel",UUID(uuidString:source.runID)?.uuidString == source.runID,WorkflowArchiveCodec.isHash(source.requestIdentity),WorkflowArchiveCodec.isHash(source.modelManifestSHA256),(4...13).contains(source.files.count),Set(source.files.map(\.path)).count == source.files.count,source.files.allSatisfy({sourceNames.contains($0.path)}) else {throw WorkflowFileLeaseError.unsafePath}
        let expected=source.files.map{WorkflowArchiveFile(path:"source/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}
        guard Set(expected.map(\.path)) == Set(request.inputFiles.filter{$0.path != "input/program-export.json" && $0.path != "source/export-manifest.json"}.map(\.path)),expected.allSatisfy({request.inputFiles.contains($0)}) else {throw WorkflowFileLeaseError.unsafePath}
        let contract=try JSONDecoder().decode(JSONValue.self,from:DiagnosticFileIO.read(root:directory,parts:["source","input-contract.json"],limit:2_000_000))
        guard case .object(let object)=contract,[JSONValue.integer(1),.integer(2)].contains(object["schemaVersion"] ?? .null),case .object(let input)=object["input"],["float32","int64"].contains(input["dtype"]?.stringValue ?? ""),case .array(let axes)=input["shape"],(2...8).contains(axes.count),axes.first == .string("N"),axes.dropFirst().allSatisfy({if case .integer(let n)=$0 {return (1...4096).contains(n)};return false}) else {throw WorkflowFileLeaseError.unsafePath}
    }
}
