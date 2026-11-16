import Foundation
import KwipMLCore

/// A task-provider receipt uses the same retained operation ownership and file
/// verifier as other workers. It never impersonates a canvas graph checkpoint.
struct ModelFineTuningRecord:Codable,Equatable,Identifiable {
    let schemaVersion:Int
    let operationID:String
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let sourceAddress:WorkflowNodeAddress
    let base:ModelAssetReference
    let finishedAt:Date
    let files:[WorkflowArchiveFile]
    let progressFile:WorkflowArchiveFile?
    var id:String {operationID}
    func directory(_ root:URL)->URL {root.appendingPathComponent("workflow-operations/\(document.projectID)/\(operationID)")}
    func verify(root:URL)throws->[String:JSONValue] {
        guard schemaVersion==1,UUID(uuidString:operationID)?.uuidString==operationID,base.projectID==document.projectID,base.hasValidPins,files.count<=512,Set(files.map(\.path)).count==files.count,files.contains(where:{$0.path=="output/model-metadata.json"}),document.nodes.contains(where:{$0.address==address && FineTuningConfiguration.isExternal($0)}) else {throw WorkflowArchiveError("Invalid fine-tuning receipt identity, base pins or file inventory.")}
        try WorkflowCodec.validate(document)
        let path=directory(root)
        let request=try WorkflowToolRequest.decode(DiagnosticFileIO.read(root:path,parts:["request.json"],limit:256_000));do {try request.validate(directory:path,verifyInputs:false)}catch {throw WorkflowArchiveError("Captured request: "+error.localizedDescription)}
        guard request.operationID==operationID,request.projectID==document.projectID,request.toolMode == .modelAsset else {throw WorkflowFileLeaseError.unsafePath}
        for file in request.inputFiles where file.path.hasPrefix("input/") {do {try WorkflowPreviewToolContract.verifyFile(file:file,directory:path,limit:ModelAssetToolContract.limit(file.path))}catch {throw WorkflowArchiveError("Captured training input "+file.path+": "+error.localizedDescription)}}
        do {try ModelAssetToolContract.verifyCaptureVersion(request,directory:path)}catch {throw WorkflowArchiveError("Captured source/task contract: "+error.localizedDescription)}
        for file in base.files {do {try WorkflowPreviewToolContract.verifyFile(file:file,directory:base.directory(root:root),limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}catch {throw WorkflowArchiveError("Immutable base "+file.path+": "+error.localizedDescription)}}
        var total:Int64=0
        for file in files {
            try Task.checkCancellation();guard ["output/model-metadata.json","output/table.json"].contains(file.path) || ModelAssetOperation.trainingOutput(file.path) else {throw WorkflowArchiveError("Unrecognized captured training output: "+file.path)}
            total+=Int64(file.bytes);guard total<=12_100_000_000 else {throw WorkflowFileLeaseError.unsafePath}
            do {try WorkflowPreviewToolContract.verifyFile(file:file,directory:path,limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}catch {throw WorkflowArchiveError("Captured training output "+file.path+": "+error.localizedDescription)}
        }
        let report=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:path,parts:["output","training","training.json"],limit:8_000_000))
        guard report["baseIdentity"] == .string(base.identity),report["kind"] == .string("kwipFineTuning") else {throw WorkflowArchiveError("Training report base identity or kind differs from the pinned input.")}
        if let progressFile {
            guard progressFile.path==TrainingProgressArchive.path else {throw WorkflowFileLeaseError.unsafePath}
            try WorkflowPreviewToolContract.verifyFile(file:progressFile,directory:path,limit:8_000_000)
            let archive=try TrainingProgressArchive.decode(DiagnosticFileIO.read(root:path,parts:WorkflowArchiveCodec.pathParts(progressFile.path),limit:8_000_000))
            guard case .array(let values)=report["metrics"] else {throw WorkflowFileLeaseError.unsafePath}
            let initial=Int(report["initialEpoch"]?.integerValue ?? 0)
            let metrics=values.compactMap {value -> EpochMetric? in guard case .object(let row)=value,let epoch=row["epoch"]?.integerValue,let train=row["trainingLoss"]?.doubleValue,let valid=row["validationLoss"]?.doubleValue else {return nil};guard epoch>initial else{return nil};return .init(epoch:Int(epoch),trainLoss:train,validationLoss:valid,accuracy:nil)}
            try archive.validate(request:request.requestIdentity,worker:operationID,metrics:metrics)
        }
        var metadata=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:path,parts:["output","model-metadata.json"],limit:8_000_000))
        if case .object(let settings)=report["settings"] {metadata["trainingDevice"]=settings["device"]}
        metadata["paused"]=report["paused"];metadata["initialEpoch"]=report["initialEpoch"];metadata["continuation"]=report["continuation"];metadata["completedEpoch"]=report["completedEpoch"];metadata["selectedEpoch"]=report["selectedEpoch"];metadata["trainableParameters"]=report["trainableParameters"];metadata["trainingTotalParameters"]=report["totalParameters"]
        if files.contains(where:{$0.path=="output/table.json"}) {
            let table=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:path,parts:["output","table.json"],limit:8_000_000))
            guard case .array(let rows)=table["rows"],case .array(let columns)=table["columns"] else {throw WorkflowArchiveError("Completed prediction table is malformed.")}
            metadata["predictionSummary"] = .object(["kind":.string("table"),"rowCount":.integer(Int64(rows.count)),"columnCount":.integer(Int64(columns.count)),"columns":.array(columns)])
        }
        return metadata
    }
    func report(root:URL)throws->[String:JSONValue] {
        _=try verify(root:root)
        return try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory(root),parts:["output","training","training.json"],limit:8_000_000))
    }
    static func scan(root:URL,project:String)->[Self] {
        guard UUID(uuidString:project) != nil else {return []}
        let folder=root.appendingPathComponent("workflow-operations/"+project)
        let names=(try? FileManager.default.contentsOfDirectory(atPath:folder.path)) ?? []
        var records:[Self]=[]
        for name in names.sorted().suffix(1000) {
            if Task.isCancelled {break}
            guard UUID(uuidString:name) != nil,let bytes=try? DiagnosticFileIO.read(root:folder,parts:[name,"model-training-outcome.json"],limit:2_000_000),let record=try? JSONDecoder().decode(Self.self,from:bytes),record.operationID==name,record.document.projectID==project else {continue}
            records.append(record)
        }
        return records.sorted{$0.finishedAt>$1.finishedAt}
    }
}
