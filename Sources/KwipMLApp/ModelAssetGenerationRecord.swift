import Foundation
import KwipMLCore

/// Imported-model generations retain their own source identity; never pretend
/// they came from an app training checkpoint or rewrite old generation records.
struct ModelAssetGenerationRecord:Codable,Equatable {
    let schemaVersion:Int
    let operationID:String
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let sourceAddress:WorkflowNodeAddress
    let reference:ModelAssetReference
    let finishedAt:Date
    let files:[WorkflowArchiveFile]
    func directory(_ root:URL)->URL {root.appendingPathComponent("workflow-operations/\(document.projectID)/\(operationID)")}
    func verify(root:URL)throws->[String:JSONValue] {
        guard schemaVersion==1,UUID(uuidString:operationID)?.uuidString==operationID,reference.projectID==document.projectID,reference.hasValidPins,Set(files.map(\.path))==["output/table.json","output/generation.json"],files.count==2 else {throw WorkflowFileLeaseError.unsafePath}
        try WorkflowCodec.validate(document)
        for file in files {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory(root),limit:8_000_000)}
        let metadata=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory(root),parts:["output","generation.json"],limit:8_000_000))
        guard metadata["modelIdentity"] == .string(reference.identity) else {throw WorkflowArchiveError("Generation model provenance changed.")}
        return try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory(root),parts:["output","table.json"],limit:8_000_000))
    }
}
