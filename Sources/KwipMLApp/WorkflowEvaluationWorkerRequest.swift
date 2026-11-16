import Foundation
import KwipMLCore

/// Worker-owned evaluation contract: paths are fixed relative to request.json.
struct WorkflowEvaluationWorkerRequest:Codable,Equatable {
    var schemaVersion:Int=1
    var kind:String="frozenEvaluation"
    let evaluationID:String
    let projectID:String
    let requestIdentity:String
    let snapshotReceiptSHA256:String
    let settings:WorkflowEvaluationSettings
    var executionBudget:ResolvedExecutionBudget? = nil
    var retainComponentOutputs:Bool? = nil

    func validate()throws {
        try WorkflowWorkerLaunchBudget.validatePolicy(schemaVersion:schemaVersion,executionBudget:executionBudget)
        guard kind == "frozenEvaluation",UUID(uuidString:evaluationID) != nil,
              UUID(uuidString:projectID) != nil,WorkflowArchiveCodec.isHash(requestIdentity),
              WorkflowArchiveCodec.isHash(snapshotReceiptSHA256) else {
            throw WorkflowArchiveError("Invalid frozen evaluation worker request identity.")
        }
        guard retainComponentOutputs == nil || retainComponentOutputs == false else {throw WorkflowArchiveError("Unsupported evaluation retention override.")}
        try settings.validate()
    }
    func encoded()throws->Data {try validate();return try WorkflowArchiveCodec.encode(self)}
    func requestSHA256()throws->String {try WorkflowArchiveCodec.hash(encoded())}
    static func decode(_ bytes:Data)throws->Self {
        guard bytes.count<=256_000,let object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],
              let settings=object["settings"] as? [String:Any],(Set(settings.keys)==["device","batchSize","previewLimit"] || Set(settings.keys)==["device","batchSize","previewLimit","settingsVersion","partitionRole"]) else {
            throw WorkflowArchiveError("Evaluation request has unknown, missing or oversized fields.")
        }
        let value=try JSONDecoder().decode(Self.self,from:bytes)
        var fields:Set<String>=["schemaVersion","kind","evaluationID","projectID","requestIdentity","snapshotReceiptSHA256","settings"]
        if object["retainComponentOutputs"] != nil {
            guard try JSONDecoder().decode([String:JSONValue].self,from:bytes)["retainComponentOutputs"] == .bool(false) else {throw WorkflowArchiveError("Evaluation retention override must be false.")}
            fields.insert("retainComponentOutputs")
        }
        try WorkflowWorkerLaunchBudget.validatePolicyFields(object,required:fields,schemaVersion:value.schemaVersion,executionBudget:value.executionBudget)
        try value.validate();return value
    }
}

struct WorkflowEvaluationWorkerInput {
    let request:WorkflowEvaluationWorkerRequest
    let snapshot:WorkflowFrozenModelSnapshot
    let capture:WorkflowArchiveFile

    private struct Capture:Codable {
        let schemaVersion:Int
        let kind:String
        let evaluationID:String
        let requestIdentity:String
        let bindingRevision:UInt64
        let workerRequest:WorkflowArchiveFile
        let workflow:WorkflowArchiveFile
        let runtimeEvidence:WorkflowArchiveFile
        let snapshotReceipt:WorkflowArchiveFile
    }
    /// Called off the main actor. A supplied owner retains exclusive staging authority;
    /// otherwise this service exclusively allocates the directory. Partial evidence is preserved.
    static func publish(request:WorkflowEvaluationRequest,evaluationID:UUID,directory:URL,owner:WorkflowEvaluationOwner? = nil)throws->Self {
        try request.verifyCapturedInputs()
        guard UUID(uuidString:directory.lastPathComponent) != nil else {throw WorkflowArchiveError("Evaluation run directory requires a UUID identity.")}
        let worker=WorkflowEvaluationWorkerRequest(schemaVersion:request.executionBudget == nil ? 1 : 2,evaluationID:evaluationID.uuidString,projectID:request.document.projectID,
            requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshot.receiptSHA256,settings:request.plan.settings,executionBudget:request.executionBudget,retainComponentOutputs:request.historyPolicy.componentOutputs ? nil:false)
        let requestBytes=try worker.encoded()
        try Task.checkCancellation()
        if let owner {
            try owner.validateDirectory(directory);try owner.check()
            guard owner.metadata.evaluationID==worker.evaluationID,owner.metadata.projectID==worker.projectID,
                  owner.metadata.requestIdentity==worker.requestIdentity,owner.metadata.requestSHA256==WorkflowArchiveCodec.hash(requestBytes) else {
                throw WorkflowArchiveError("Evaluation input does not match its reserved ownership.")
            }
        }
        else {try DiagnosticFileIO.createExclusiveDirectory(root:directory.deletingLastPathComponent(),name:directory.lastPathComponent)}
        try DiagnosticFileIO.createExclusiveDirectory(root:directory,name:"input")
        let input=directory.appendingPathComponent("input")
        for file in request.snapshot.files {
            try Task.checkCancellation()
            let bytes=try DiagnosticFileIO.read(root:request.snapshot.directory,parts:WorkflowArchiveCodec.pathParts(file.path),limit:WorkflowFrozenModelSnapshot.fileLimit(file.path))
            guard bytes.count == file.bytes,WorkflowArchiveCodec.hash(bytes) == file.sha256 else {throw WorkflowArchiveError("Frozen input changed before run capture.")}
            try DiagnosticFileIO.writeFrozenEvaluationArtifact(bytes,root:input,path:file.path)
        }
        let receipt=try DiagnosticFileIO.read(root:request.snapshot.directory,parts:["snapshot.json"],limit:256_000)
        guard WorkflowArchiveCodec.hash(receipt)==worker.snapshotReceiptSHA256 else {throw WorkflowArchiveError("Frozen snapshot receipt changed before run capture.")}
        try DiagnosticFileIO.writeImmutable(receipt,root:input,parts:["snapshot.json"],limit:256_000)
        let snapshot=WorkflowFrozenModelSnapshot(reference:request.reference,directory:input,files:request.snapshot.files,receiptSHA256:worker.snapshotReceiptSHA256)
        try snapshot.verify()
        func write(_ bytes:Data,_ path:String,_ limit:Int)throws->WorkflowArchiveFile {
            try Task.checkCancellation()
            try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:[path],limit:limit)
            return .init(path:path,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        }
        let workflow=try write(WorkflowCodec.encode(request.document),"workflow.json",2_000_000)
        let runtime=try write(WorkflowArchiveCodec.encode(request.runtimeEvidence),"runtime-evidence.json",256_000)
        let workerFile=WorkflowArchiveFile(path:"request.json",bytes:requestBytes.count,sha256:WorkflowArchiveCodec.hash(requestBytes))
        let capture=Capture(schemaVersion:1,kind:"frozenEvaluationCapture",evaluationID:worker.evaluationID,
            requestIdentity:request.requestIdentity,bindingRevision:request.bindingRevision,workerRequest:workerFile,
            workflow:workflow,runtimeEvidence:runtime,snapshotReceipt:.init(path:"input/snapshot.json",bytes:receipt.count,sha256:worker.snapshotReceiptSHA256))
        let captureFile=try write(WorkflowArchiveCodec.encode(capture),"capture.json",256_000)
        // Publish the executable request only after all immutable inputs/evidence exist.
        _=try write(requestBytes,"request.json",256_000)
        return .init(request:worker,snapshot:snapshot,capture:captureFile)
    }
}
