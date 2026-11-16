import Foundation
import KwipMLCore

struct WorkflowGenerationRecord:Codable,Identifiable {
    var schemaVersion=1
    var status="completed"
    var message:String? = nil
    let id:String
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let reference:WorkflowFrozenModelReference
    let runtime:WorkflowRuntimeEvidence
    let finishedAt:Date
    let files:[WorkflowArchiveFile]
    enum CodingKeys:String,CodingKey {case schemaVersion,status,message,id,address,document,reference,runtime,finishedAt,files}
    init(schemaVersion:Int=1,status:String="completed",message:String?=nil,id:String,address:WorkflowNodeAddress,document:WorkflowDocument,reference:WorkflowFrozenModelReference,runtime:WorkflowRuntimeEvidence,finishedAt:Date,files:[WorkflowArchiveFile]) {
        self.schemaVersion=schemaVersion;self.status=status;self.message=message;self.id=id;self.address=address;self.document=document;self.reference=reference;self.runtime=runtime;self.finishedAt=finishedAt;self.files=files
    }
    init(from decoder:Decoder)throws {
        let c=try decoder.container(keyedBy:CodingKeys.self)
        self.init(schemaVersion:try c.decode(Int.self,forKey:.schemaVersion),status:try c.decodeIfPresent(String.self,forKey:.status) ?? "completed",message:try c.decodeIfPresent(String.self,forKey:.message),id:try c.decode(String.self,forKey:.id),address:try c.decode(WorkflowNodeAddress.self,forKey:.address),document:try c.decode(WorkflowDocument.self,forKey:.document),reference:try c.decode(WorkflowFrozenModelReference.self,forKey:.reference),runtime:try c.decode(WorkflowRuntimeEvidence.self,forKey:.runtime),finishedAt:try c.decode(Date.self,forKey:.finishedAt),files:try c.decode([WorkflowArchiveFile].self,forKey:.files))
    }
    struct Verified {let result:[String:JSONValue];let table:[String:JSONValue];let artifact:TableViewQuerySession.Artifact}
    func directory(root:URL)->URL {root.appendingPathComponent("workflow-operations/\(document.projectID)/\(id)")}
    func verify(root:URL)throws->Verified {
        guard status=="completed" else {throw WorkflowArchiveError(message ?? "Generation did not complete.")}
        guard schemaVersion==1,UUID(uuidString:id)?.uuidString==id,reference.hasValidPins,reference.projectID==document.projectID,Set(files.map(\.path))==["output/table.json","output/generation.json"],files.count==2 else {throw WorkflowFileLeaseError.unsafePath}
        let directory=directory(root:root)
        let request=try JSONDecoder().decode(WorkflowToolRequest.self,from:DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:262_144))
        guard request.operationID==id,request.projectID==document.projectID,request.toolMode == .textGeneration else {throw WorkflowFileLeaseError.unsafePath}
        try request.validate(directory:directory,verifyInputs:true)
        let source=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["source","export-manifest.json"],limit:2_000_000))
        guard source["runID"] == .string(reference.runID),source["requestIdentity"] == .string(reference.requestIdentity),source["modelManifestSHA256"] == .string(reference.modelManifestSHA256) else {throw WorkflowFileLeaseError.unsafePath}
        guard request.inputFiles.first(where:{$0.path=="source/checkpoint.pt"})?.sha256==reference.checkpointSHA256 else {throw WorkflowFileLeaseError.unsafePath}
        for file in files {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:8_000_000)}
        let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["output","generation.json"],limit:8_000_000))
        let table=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["output","table.json"],limit:8_000_000))
        guard case .array(let vocabulary)=result["vocabulary"],(2...512).contains(vocabulary.count),vocabulary.allSatisfy({$0.stringValue != nil}),let node=document.nodes.first(where:{$0.address==address}),case .generateText(let values)=node.payload,let column=try TextGenerationConfiguration(values).values["idsColumn"]?.stringValue,let file=files.first(where:{$0.path=="output/table.json"}) else {throw WorkflowFileLeaseError.unsafePath}
        return .init(result:result,table:table,artifact:.init(indexURL:directory.appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table",tokenVocabulary:.capturedGeneration(directory:directory,column:column,vocabulary:vocabulary.compactMap(\.stringValue))))
    }
    static func list(root:URL,projectID:String)throws->[Self] {
        let directory=root.appendingPathComponent("workflow-operations/\(projectID)")
        guard FileManager.default.fileExists(atPath:directory.path) else {return []}
        let candidates=try FileManager.default.contentsOfDirectory(at:directory,includingPropertiesForKeys:[.isDirectoryKey,.isSymbolicLinkKey])
        guard candidates.count<=20_000 else {throw WorkflowArchiveError("Too many local operations to scan; use temporary-data cleanup first.")}
        var values:[Self]=[]
        for entry in candidates where UUID(uuidString:entry.lastPathComponent)?.uuidString==entry.lastPathComponent {
            try Task.checkCancellation()
            guard let resource=try? entry.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey]),resource.isDirectory==true,resource.isSymbolicLink != true,FileManager.default.fileExists(atPath:entry.appendingPathComponent("generation-outcome.json").path) else {continue}
            guard let raw=try? DiagnosticFileIO.read(root:entry,parts:["generation-outcome.json"],limit:2_000_000),let value=try? JSONDecoder().decode(Self.self,from:raw),value.id==entry.lastPathComponent,value.document.projectID==projectID,value.schemaVersion==1 else {continue}
            values.append(value)
        }
        return Array(values.sorted{$0.finishedAt>$1.finishedAt}.prefix(100))
    }
}
