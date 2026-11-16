import Foundation
import KwipMLCore

enum WorkflowGenerationExport {
    static func stage(_ record:WorkflowGenerationRecord,root:URL,to destination:URL)throws->WorkflowPendingExportPublication {
        let verified=try record.verify(root:root),directory=record.directory(root:root)
        let request=try JSONDecoder().decode(WorkflowToolRequest.self,from:DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:2_000_000))
        var files:[String:Data]=[:]
        for file in request.inputFiles where file.path.hasPrefix("source/") {
            files[String(file.path.dropFirst(7))]=try DiagnosticFileIO.read(root:directory,parts:WorkflowArchiveCodec.pathParts(file.path),limit:WorkflowGenerationToolContract.fileLimit(file.path))
        }
        files["text_generation.py"]=try DiagnosticFileIO.read(root:directory,parts:["input","generation-kernel.py"],limit:200_000)
        guard let replay=Bundle.module.url(forResource:"generation_replay",withExtension:"py") else {throw WorkflowFileLeaseError.unsafePath}
        files["generate.py"]=try Data(contentsOf:replay)
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["input","generation.json"],limit:2_000_000))
        guard case .object(let configuration)=inner["configuration"],case .array(let rows)=verified.table["rows"],case .array(let columns)=verified.table["columns"] else {throw WorkflowFileLeaseError.unsafePath}
        let c=try TextGenerationConfiguration(configuration),name=c.tableMode ? c.values["promptColumn"]?.stringValue:"prompt"
        guard let index=columns.firstIndex(where:{if case .object(let v)=$0 {return v["name"]?.stringValue==name};return false}) else {throw WorkflowFileLeaseError.unsafePath}
        let prompts=try rows.map {row->JSONValue in guard case .array(let v)=row,index<v.count,case .string=v[index] else {throw WorkflowFileLeaseError.unsafePath};return v[index]}
        files["generation-input.json"]=try WorkflowArchiveCodec.encode(["prompts":JSONValue.array(prompts),"settings":.object(c.workerSettings)])
        files["GENERATION-README.md"]=Data("# Captured text generation\n\nInstall the package requirements, then run `python generate.py`. It replays the captured prompts, saved tokenizer and exact bounded decoding algorithm and prints JSON. `captured-generation.json` is the recorded comparison. CPU replay requires the same supported Torch/runtime versions; GPU numerics may differ.\n\nThe generation manifest pins every replay file. Editing captured inputs deliberately invalidates that exact replay. For a new request, import `generate` from `text_generation.py` and use `load_model` from `inference.py`; keep the captured tokenizer and model contract. No source documents, training or network access are needed.\n".utf8)
        files["captured-generation.json"]=try WorkflowArchiveCodec.encode(verified.result)
        struct Manifest:Encodable {let schemaVersion=1;let kind="kwipmlTextGeneration";let files:[WorkflowArchiveFile]}
        let manifest=Manifest(files:files.sorted{$0.key<$1.key}.map{.init(path:$0.key,bytes:$0.value.count,sha256:WorkflowArchiveCodec.hash($0.value))})
        files["generation-manifest.json"]=try WorkflowArchiveCodec.encode(manifest)
        return try WorkflowExportPublication.stageGeneration(files,to:destination)
    }
}
