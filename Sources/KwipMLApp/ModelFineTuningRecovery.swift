import Foundation
import KwipMLCore

/// A failed/cancelled operation is never presented as a completed inference result.
/// Only its independently pinned, atomically committed epoch may be resumed.
struct ModelFineTuningRecovery:Codable,Identifiable {
    let operationID:String
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let base:ModelAssetReference
    let epochs:[Int]
    let files:[WorkflowArchiveFile]
    let finishedAt:Date
    var id:String{operationID}
    func directory(_ root:URL)->URL{root.appendingPathComponent("workflow-operations/\(document.projectID)/\(operationID)")}
    func capture(root:URL,epoch:Int=0)throws->ModelResumeCapture {
        guard let selected=epoch==0 ? epochs.max():epochs.first(where:{$0==epoch}) else{throw WorkflowArchiveError("Choose a retained completed epoch.")}
        let dir=directory(root),prefix="output/training/checkpoints/epoch-"+String(format:"%06d",selected)+"/"
        guard UUID(uuidString:id)?.uuidString==id,base.projectID==document.projectID,base.hasValidPins,epochs.count<=10 else{throw WorkflowFileLeaseError.unsafePath}
        let request=try WorkflowToolRequest.decode(DiagnosticFileIO.read(root:dir,parts:["request.json"],limit:256_000))
        try request.validate(directory:dir,verifyInputs:false)
        guard request.operationID==id,request.projectID==document.projectID,request.toolMode == .modelAsset else{throw WorkflowFileLeaseError.unsafePath}
        for file in request.inputFiles where file.path.hasPrefix("input/"){try WorkflowPreviewToolContract.verifyFile(file:file,directory:dir,limit:ModelAssetToolContract.limit(file.path))}
        for file in base.files{try WorkflowPreviewToolContract.verifyFile(file:file,directory:base.directory(root:root),limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}
        let selectedFiles=files.filter{$0.path.hasPrefix(prefix)}
        guard !selectedFiles.isEmpty,selectedFiles.count<=128 else{throw WorkflowFileLeaseError.unsafePath}
        for file in selectedFiles{try WorkflowPreviewToolContract.verifyFile(file:file,directory:dir,limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}
        let folder=dir.appendingPathComponent(prefix)
        let reportBytes=try DiagnosticFileIO.read(root:folder,parts:["training.json"],limit:8_000_000)
        let report=try JSONDecoder().decode([String:JSONValue].self,from:reportBytes)
        guard report["kind"] == .string("kwipFineTuning"),report["baseIdentity"] == .string(base.identity),report["completedEpoch"] == .integer(Int64(selected)),report["continuation"] == .string("cpuEpochBoundary") else{throw WorkflowArchiveError("Recovery boundary needs the exact base and CPU runtime.")}
        let leaves=selectedFiles.map{WorkflowArchiveFile(path:String($0.path.dropFirst(prefix.count)),bytes:$0.bytes,sha256:$0.sha256)}
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:folder,parts:["result-manifest.json"],limit:8_000_000))
        let innerFiles=try JSONDecoder().decode([WorkflowArchiveFile].self,from:WorkflowArchiveCodec.encode(inner["files"] ?? .null))
        guard inner["schemaVersion"] == 1,innerFiles==leaves.filter{$0.path != "result-manifest.json"}.sorted(by:{$0.path<$1.path}) else{throw WorkflowArchiveError("Recovery epoch differs from its committed file inventory.")}
        let mapped=leaves.map{WorkflowArchiveFile(path:"latest/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}+[.init(path:"training.json",bytes:reportBytes.count,sha256:WorkflowArchiveCodec.hash(reportBytes))]
        let manifest=try WorkflowArchiveCodec.encode(["schemaVersion":JSONValue.integer(1),"files":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(mapped.sorted{$0.path<$1.path}))])
        return .init(source:folder,files:leaves,report:report,reportBytes:reportBytes,manifestBytes:manifest)
    }
    static func collect(directory:URL,address:WorkflowNodeAddress,document:WorkflowDocument,base:ModelAssetReference,root:URL)throws->Self? {
        let checkpoints=directory.appendingPathComponent("output/training/checkpoints")
        let names=(try? FileManager.default.contentsOfDirectory(atPath:checkpoints.path)) ?? []
        let epochs=names.compactMap{name->Int? in guard name.range(of:"^epoch-[0-9]{6}$",options:.regularExpression) != nil else{return nil};return Int(name.suffix(6))}.sorted()
        guard !epochs.isEmpty,epochs.count<=10 else{return nil}
        var files:[WorkflowArchiveFile]=[]
        for epoch in epochs {
            let prefix="output/training/checkpoints/epoch-"+String(format:"%06d",epoch)+"/",folder=directory.appendingPathComponent(prefix)
            let manifestBytes=try DiagnosticFileIO.read(root:folder,parts:["result-manifest.json"],limit:8_000_000)
            let manifest=try JSONDecoder().decode([String:JSONValue].self,from:manifestBytes)
            let leaves=try JSONDecoder().decode([WorkflowArchiveFile].self,from:WorkflowArchiveCodec.encode(manifest["files"] ?? .null))
            guard manifest["schemaVersion"]==1,leaves.count<=128,leaves.allSatisfy({ModelFineTuningFiles.leaf($0.path)}) else{throw WorkflowFileLeaseError.unsafePath}
            files += (leaves+[.init(path:"result-manifest.json",bytes:manifestBytes.count,sha256:WorkflowArchiveCodec.hash(manifestBytes))]).map{.init(path:prefix+$0.path,bytes:$0.bytes,sha256:$0.sha256)}
        }
        let record=Self(operationID:directory.lastPathComponent,address:address,document:document,base:base,epochs:epochs,files:files,finishedAt:Date())
        _=try record.capture(root:root)
        try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:directory,parts:["model-training-recovery.json"],limit:2_000_000)
        return record
    }
    static func scan(root:URL,project:String)->[Self] {
        guard UUID(uuidString:project) != nil else{return []}
        let directory=root.appendingPathComponent("workflow-operations/"+project)
        return ((try? FileManager.default.contentsOfDirectory(atPath:directory.path)) ?? []).sorted().suffix(1000).compactMap{name in guard UUID(uuidString:name) != nil,let bytes=try? DiagnosticFileIO.read(root:directory,parts:[name,"model-training-recovery.json"],limit:2_000_000),let value=try? JSONDecoder().decode(Self.self,from:bytes),value.id==name,value.document.projectID==project else{return nil};return value}.sorted{$0.finishedAt>$1.finishedAt}
    }
}
