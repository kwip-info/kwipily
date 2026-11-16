import Foundation
import KwipMLCore

/// Imported causal evaluation uses the same retained operation/file authority and
/// durable exposure journal; it is never a training or pristine-holdout claim.
struct ModelCausalEvaluationRecord:Codable,Identifiable {
    let operationID:String
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let sourceAddress:WorkflowNodeAddress
    let model:ModelAssetReference
    let files:[WorkflowArchiveFile]
    let finishedAt:Date
    var id:String{operationID}
    func directory(_ root:URL)->URL{root.appendingPathComponent("workflow-operations/\(document.projectID)/\(id)")}
    func verify(root:URL)throws->[String:JSONValue] {
        guard UUID(uuidString:id)?.uuidString==id,model.hasValidPins,model.projectID==document.projectID,Set(files.map(\.path))==["output/tokens.json","output/table.json"],files.count==2 else{throw WorkflowFileLeaseError.unsafePath}
        try WorkflowCodec.validate(document)
        let selected=try WorkflowTrainedModelSource.resolve(in:document,consumer:address)
        guard selected.address==sourceAddress else{throw WorkflowFileLeaseError.unsafePath}
        let dir=directory(root),request=try WorkflowToolRequest.decode(DiagnosticFileIO.read(root:dir,parts:["request.json"],limit:256_000))
        try request.validate(directory:dir,verifyInputs:false)
        guard request.operationID==id,request.projectID==document.projectID,request.toolMode == .modelAsset else{throw WorkflowFileLeaseError.unsafePath}
        for file in request.inputFiles where file.path.hasPrefix("input/"){try WorkflowPreviewToolContract.verifyFile(file:file,directory:dir,limit:ModelAssetToolContract.limit(file.path))}
        try ModelAssetToolContract.verifyCaptureVersion(request,directory:dir)
        let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:dir,parts:["input","model-request.json"],limit:2_000_000))
        guard inner["action"] == .string("evaluate"),inner["sourceIdentity"] == .string(model.identity) else{throw WorkflowFileLeaseError.unsafePath}
        for file in model.files{try WorkflowPreviewToolContract.verifyFile(file:file,directory:model.directory(root:root),limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)}
        for file in files{try WorkflowPreviewToolContract.verifyFile(file:file,directory:dir,limit:8_000_000)}
        let summary=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:dir,parts:["output","tokens.json"],limit:8_000_000))
        guard [JSONValue.string("causalEvaluation"),.string("typedDecisionEvaluation")].contains(summary["kind"] ?? .null),summary["modelIdentity"] == .string(model.identity),["validation","test"].contains(summary["role"]?.stringValue ?? ""),let tokens=summary[summary["kind"] == .string("typedDecisionEvaluation") ? "scoredItems":"scoredTokens"]?.integerValue,tokens>0,let loss=summary["loss"]?.doubleValue,loss.isFinite,loss>=0 else{throw WorkflowArchiveError("Causal evaluation summary is incomplete.")}
        return summary
    }
    func disclose(root:URL,exposureRoot:URL?)throws->[String:JSONValue] {
        var summary=try verify(root:root)
        guard case .array(let ids)=summary["observationIDs"],let source=summary["sourceDigest"]?.stringValue else{throw WorkflowFileLeaseError.unsafePath}
        let population=try WorkflowEvaluationExposureStore.Population(checkpointSHA256:model.identity,sourceSHA256:source,observationIDs:ids.compactMap(\.stringValue))
        let receipt=try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(self))
        let prior=try WorkflowEvaluationDisclosure.recordImported(evaluationID:id,projectID:document.projectID,receipt:receipt,population:population,root:root,exposureRoot:exposureRoot)
        summary["priorInspections"] = .integer(Int64(prior.inspectionCount));summary["priorOverlap"] = .integer(Int64(prior.sameSourceInspectedOverlap));summary.removeValue(forKey:"observationIDs")
        return summary
    }
    static func scan(root:URL,project:String)->[Self] {
        guard UUID(uuidString:project) != nil else{return []}
        let folder=root.appendingPathComponent("workflow-operations/"+project)
        return ((try? FileManager.default.contentsOfDirectory(atPath:folder.path)) ?? []).sorted().suffix(1000).compactMap{name in
            guard UUID(uuidString:name) != nil,let bytes=try? DiagnosticFileIO.read(root:folder,parts:[name,"model-evaluation-outcome.json"],limit:2_000_000),let value=try? JSONDecoder().decode(Self.self,from:bytes),value.id==name,value.document.projectID==project else{return nil};return value
        }.sorted{$0.finishedAt>$1.finishedAt}
    }
}
