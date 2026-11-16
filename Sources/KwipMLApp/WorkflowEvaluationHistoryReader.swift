import Foundation
import Darwin
import KwipMLCore

struct WorkflowEvaluationHistoryReference:Hashable {let projectID:String;let evaluationID:UUID}
enum WorkflowEvaluationHistoryVerification {case recordedUnverified,inUse,incomplete,unavailable}
struct WorkflowEvaluationHistoryEntry {
    let reference:WorkflowEvaluationHistoryReference
    let status:WorkflowEvaluationStatus?
    let verification:WorkflowEvaluationHistoryVerification
    let message:String?
    var finishedAt:Date?=nil
    var partitionRole:String?=nil
}
struct WorkflowEvaluationHistoryListing {let entries:[WorkflowEvaluationHistoryEntry];let issues:[String];let truncated:Bool}
struct VerifiedWorkflowEvaluationRecord {
    let outcome:WorkflowEvaluationOutcome
    let result:WorkflowEvaluationArtifacts.Verified?
    let captureIncomplete:Bool
}
enum WorkflowEvaluationHistoryLoad {
    case inUse,incomplete(String),terminal(VerifiedWorkflowEvaluationRecord),unavailable(String)
}

/// Independent saved evaluation evidence. Never supplies a training comparison candidate.
enum WorkflowEvaluationHistoryReader {
    private struct Capture:Decodable {
        let schemaVersion:Int;let kind:String;let evaluationID:String;let requestIdentity:String;let bindingRevision:UInt64
        let workerRequest:WorkflowArchiveFile;let workflow:WorkflowArchiveFile;let runtimeEvidence:WorkflowArchiveFile;let snapshotReceipt:WorkflowArchiveFile
    }
    private struct Receipt:Decodable {let schemaVersion:Int;let reference:WorkflowFrozenModelReference;let testEndpoint:DataFlowEndpoint;let partitionRole:String?;let files:[WorkflowArchiveFile]}
    static func load(_ reference:WorkflowEvaluationHistoryReference,root:URL)->WorkflowEvaluationHistoryLoad {
        do {
            let directory=try path(reference,root)
            let inspection=try WorkflowEvaluationOwnership.acquireInspection(directory:directory,projectID:reference.projectID,evaluationID:reference.evaluationID.uuidString)
            defer {inspection.close()}
            return try load(reference,inspection:inspection)
        } catch WorkflowFileLeaseError.busy {return .inUse}
        catch {return .unavailable(error.localizedDescription)}
    }
    /// Verifies saved evidence under an already-retained inactive lease. The caller
    /// retains and rechecks that lease through asynchronous export staging.
    static func load(_ reference:WorkflowEvaluationHistoryReference,inspection:WorkflowEvaluationInspection)throws->WorkflowEvaluationHistoryLoad {
        try inspection.check()
        guard inspection.metadata.projectID==reference.projectID,
              inspection.metadata.evaluationID==reference.evaluationID.uuidString else {throw failure("Retained evaluation belongs to another history entry.")}
        let value=try loadCaptured(reference,directory:inspection.directory,metadata:inspection.metadata)
        try inspection.check()
        return value
    }
    private static func loadCaptured(_ reference:WorkflowEvaluationHistoryReference,directory:URL,metadata:WorkflowEvaluationOwnershipMetadata)throws->WorkflowEvaluationHistoryLoad {
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:directory,parts:["outcome.json"],limit:256000) else {return .incomplete("No terminal outcome was published.")}
        let outcome=try decodeOutcome(bytes,reference,metadata)
        guard let captureFile=outcome.capture else {
            return .terminal(.init(outcome:outcome,result:nil,captureIncomplete:true))
        }
        let captureBytes=try pinned(captureFile,path:"capture.json",limit:256000,root:directory)
        try keys(captureBytes,["schemaVersion","kind","evaluationID","requestIdentity","bindingRevision","workerRequest","workflow","runtimeEvidence","snapshotReceipt"])
        let capture=try JSONDecoder().decode(Capture.self,from:captureBytes)
        guard capture.schemaVersion==1,capture.kind=="frozenEvaluationCapture",capture.evaluationID==outcome.evaluationID,capture.requestIdentity==outcome.requestIdentity else {throw failure("Capture identity differs.")}
        let requestBytes=try pinned(capture.workerRequest,path:"request.json",limit:256000,root:directory)
        let request=try WorkflowEvaluationWorkerRequest.decode(requestBytes)
        guard WorkflowArchiveCodec.hash(requestBytes)==metadata.requestSHA256,request.requestIdentity==outcome.requestIdentity,request.evaluationID==outcome.evaluationID,request.projectID==outcome.projectID else {throw failure("Worker request identity differs.")}
        let workflow=try WorkflowCodec.decode(pinned(capture.workflow,path:"workflow.json",limit:2000000,root:directory))
        guard workflow.projectID==reference.projectID else {throw failure("Captured workflow belongs to another project.")}
        _ = try JSONSerialization.jsonObject(with:pinned(capture.runtimeEvidence,path:"runtime-evidence.json",limit:256000,root:directory))
        let receiptBytes=try pinned(capture.snapshotReceipt,path:"input/snapshot.json",limit:256000,root:directory)
        let receiptObject=try JSONSerialization.jsonObject(with:receiptBytes) as? [String:Any]
        try keys(receiptBytes,receiptObject?["schemaVersion"] as? Int == 2 ? ["schemaVersion","reference","testEndpoint","partitionRole","files"]:["schemaVersion","reference","testEndpoint","files"])
        let receipt=try JSONDecoder().decode(Receipt.self,from:receiptBytes)
        guard ((receipt.schemaVersion==1 && receipt.partitionRole==nil) || (receipt.schemaVersion==2 && receipt.partitionRole=="validation")),receipt.reference.hasValidPins,receipt.reference.projectID==reference.projectID,
              WorkflowArchiveCodec.hash(receiptBytes)==request.snapshotReceiptSHA256,(1...320).contains(receipt.files.count),Set(receipt.files.map(\.path)).count==receipt.files.count else {throw failure("Snapshot receipt is invalid.")}
        let required:Set<String>=["outcome.json","capture.json","workflow.json","model-registry.json","data-registry.json","runtime-evidence.json","training/artifacts/manifest.json","training/artifacts/checkpoint.pt","preparation/artifacts/manifest.json"]
        guard required.isSubset(of:Set(receipt.files.map(\.path))) else {throw failure("Snapshot inventory is incomplete.")}
        var total=0
        for file in receipt.files {
            let limit=try WorkflowFrozenModelSnapshot.fileLimit(file.path)
            try validFile(file,limit:limit)
            guard file.bytes<=340000000-total else {throw failure("Snapshot exceeds its byte budget.")};total+=file.bytes
        }
        let snapshot=WorkflowFrozenModelSnapshot(reference:receipt.reference,directory:directory.appendingPathComponent("input"),files:receipt.files,receiptSHA256:request.snapshotReceiptSHA256)
        try snapshot.verify()
        try validateSnapshotAuthority(snapshot)
        var result:WorkflowEvaluationArtifacts.Verified?
        if outcome.status == .completed {
            let verified=try WorkflowEvaluationArtifacts.verify(request:request,snapshot:snapshot,root:directory)
            guard verified.manifestSHA256==outcome.manifestSHA256,verified.files.sorted(by:{$0.path<$1.path})==outcome.artifacts.sorted(by:{$0.path<$1.path}) else {throw failure("Terminal outcome differs from verified artifacts.")}
            result=verified
        }
        guard try DiagnosticFileIO.read(root:directory,parts:["outcome.json"],limit:256000)==bytes else {throw failure("Outcome changed during inspection.")}
        return .terminal(.init(outcome:outcome,result:result,captureIncomplete:false))
    }
    static func list(root:URL,projectID:String,limit:Int=1000)throws->WorkflowEvaluationHistoryListing {
        guard root.isFileURL,UUID(uuidString:projectID)?.uuidString==projectID,(1...10000).contains(limit) else {throw failure("Invalid history scope or scan budget.")}
        let project=root.appendingPathComponent("workflow-evaluations").appendingPathComponent(projectID)
        let fd=open(project.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW_ANY)
        guard fd>=0 else {if errno==ENOENT {return .init(entries:[],issues:[],truncated:false)};throw failure("History directory is unsafe or unavailable.")}
        guard let stream=fdopendir(fd) else {Darwin.close(fd);throw failure("Cannot enumerate history.")}
        defer {closedir(stream)}
        var entries:[WorkflowEvaluationHistoryEntry]=[],issues:[String]=[],count=0,truncated=false
        while true {
            errno=0
            guard let entry=readdir(stream) else {if errno != 0 {throw failure("History scan failed.")};break}
            let name=withUnsafePointer(to:&entry.pointee.d_name) {$0.withMemoryRebound(to:CChar.self,capacity:Int(MAXNAMLEN)+1) {String(cString:$0)}}
            if name=="." || name==".." {continue}
            if count==limit {truncated=true;issues.append("Scan budget reached; this is not a complete newest-first history.");break};count+=1
            guard let id=UUID(uuidString:name),id.uuidString==name else {issues.append("Unrecognized history entry.");continue}
            let reference=WorkflowEvaluationHistoryReference(projectID:projectID,evaluationID:id)
            do {
                let directory=try path(reference,root)
                let value=try WorkflowEvaluationOwnership.inspect(directory:directory,projectID:projectID,evaluationID:name) { metadata -> WorkflowEvaluationHistoryEntry in
                    guard let bytes=try DiagnosticFileIO.readIfPresent(root:directory,parts:["outcome.json"],limit:256000) else {return .init(reference:reference,status:nil,verification:.incomplete,message:"No terminal outcome.")}
                    let outcome=try decodeOutcome(bytes,reference,metadata)
                    // Read only the pinned request for the list label. Metrics remain behind disclosure.
                    var role:String?
                    if let requestBytes=try? DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:256000),
                       WorkflowArchiveCodec.hash(requestBytes)==metadata.requestSHA256,
                       let request=try? WorkflowEvaluationWorkerRequest.decode(requestBytes),
                       request.projectID==reference.projectID,request.evaluationID==name {
                        role=request.settings.partitionRole
                    }
                    return .init(reference:reference,status:outcome.status,verification:.recordedUnverified,message:"Recorded status; select to verify captured evidence.",finishedAt:outcome.finishedAt,partitionRole:role)
                }
                entries.append(value)
            } catch WorkflowFileLeaseError.busy {entries.append(.init(reference:reference,status:nil,verification:.inUse,message:"Evaluation ownership is active."))}
            catch {entries.append(.init(reference:reference,status:nil,verification:.unavailable,message:error.localizedDescription))}
        }
        entries.sort {
            if $0.finishedAt != $1.finishedAt {return ($0.finishedAt ?? .distantPast) > ($1.finishedAt ?? .distantPast)}
            return $0.reference.evaluationID.uuidString < $1.reference.evaluationID.uuidString
        }
        return .init(entries:entries,issues:issues,truncated:truncated)
    }
    /// Reconstruct parent authority from the copied archive, independently of the live archive.
    private static func validateSnapshotAuthority(_ snapshot:WorkflowFrozenModelSnapshot)throws {
        let inventory=Dictionary(uniqueKeysWithValues:snapshot.files.map {($0.path,$0)})
        let reference=snapshot.reference
        let pins=["outcome.json":reference.outcomeSHA256,"training/artifacts/manifest.json":reference.modelManifestSHA256,"preparation/artifacts/manifest.json":reference.preparationManifestSHA256,"training/artifacts/checkpoint.pt":reference.checkpointSHA256]
        for (path,pin) in pins {guard inventory[path]?.sha256==pin else {throw failure("Frozen parent pin differs from snapshot inventory.")}}
        func object(_ path:String)throws->[String:Any] {
            guard let file=inventory[path],let value=try JSONSerialization.jsonObject(with:pinned(file,path:path,limit:WorkflowFrozenModelSnapshot.fileLimit(path),root:snapshot.directory)) as? [String:Any] else {throw failure("Frozen parent evidence is missing.")}
            return value
        }
        let outcome=try object("outcome.json"),capture=try object("capture.json")
        guard outcome["status"] as? String=="completed" else {throw failure("Frozen parent outcome is not completed.")}
        let identities=["projectID":reference.projectID,"directoryID":reference.directoryID,"runID":reference.runID,"requestIdentity":reference.requestIdentity]
        for (key,expected) in identities {
            guard outcome[key] as? String==expected,capture[key] as? String==expected else {throw failure("Frozen parent archive identity differs from selection.")}
        }
        guard outcome["modelManifestSHA256"] as? String==reference.modelManifestSHA256,outcome["preparationManifestSHA256"] as? String==reference.preparationManifestSHA256 else {throw failure("Frozen parent outcome pins differ from selection.")}
        let preparation=try object("preparation/artifacts/manifest.json")
        guard let artifacts=preparation["artifacts"] as? [[String:Any]],artifacts.count<=320 else {throw failure("Preparation artifact inventory is invalid.")}
        var declared=Set<String>()
        for entry in artifacts {
            guard let relative=entry["path"] as? String else {throw failure("Preparation artifact path is missing.")}
            let path="preparation/artifacts/"+relative
            _ = try WorkflowFrozenModelSnapshot.fileLimit(path)
            guard declared.insert(path).inserted,let file=inventory[path],let bytes=entry["bytes"] as? Int,let hash=entry["sha256"] as? String,file.bytes==bytes,file.sha256==hash else {throw failure("Preparation artifact is omitted, duplicated or differs from the snapshot.")}
        }
        let fixed:Set<String>=["outcome.json","capture.json","workflow.json","model-registry.json","data-registry.json","runtime-evidence.json","training/artifacts/manifest.json","training/artifacts/checkpoint.pt","preparation/artifacts/manifest.json"]
        guard Set(inventory.keys)==fixed.union(declared) else {throw failure("Snapshot inventory differs from the full preparation inventory.")}
    }
    private static func decodeOutcome(_ bytes:Data,_ reference:WorkflowEvaluationHistoryReference,_ metadata:WorkflowEvaluationOwnershipMetadata)throws->WorkflowEvaluationOutcome {
        try keys(bytes,["schemaVersion","kind","evaluationID","projectID","requestIdentity","status","startedAt","finishedAt","artifacts"],optional:["capture","manifestSHA256","error","failureCode"])
        let value=try JSONDecoder().decode(WorkflowEvaluationOutcome.self,from:bytes)
        guard value.schemaVersion==1,value.kind=="frozenEvaluationOutcome",value.evaluationID==reference.evaluationID.uuidString,value.projectID==reference.projectID,value.requestIdentity==metadata.requestIdentity,
              value.startedAt.timeIntervalSince1970.isFinite,value.finishedAt.timeIntervalSince1970.isFinite,(0...253402300799).contains(value.startedAt.timeIntervalSince1970),(0...253402300799).contains(value.finishedAt.timeIntervalSince1970),(value.error?.utf8.count ?? 0)<=16000 else {throw failure("Invalid terminal outcome identity or values.")}
        guard value.failureCode == nil || value.status == .failed else { throw failure("Failure code requires failed evaluation.") }
        if value.status == .completed {
            guard value.capture != nil,let hash=value.manifestSHA256,WorkflowArchiveCodec.isHash(hash),value.error==nil,validCompletedInventory(value.artifacts) else {throw failure("Completed outcome lacks required evidence.")}
        } else {
            guard value.manifestSHA256==nil,value.artifacts.isEmpty else {throw failure("Noncompleted outcome publishes success artifacts.")}
        }
        if let capture=value.capture {try validFile(capture,limit:256000);guard capture.path=="capture.json" else {throw failure("Unexpected capture path.")}}
        for artifact in value.artifacts {try validFile(artifact,limit:100000000)}
        guard Set(value.artifacts.map(\.path)).count==value.artifacts.count else {throw failure("Duplicate terminal artifacts.")}
        return value
    }
    private static func validCompletedInventory(_ files:[WorkflowArchiveFile])->Bool {
        let original:Set<String>=["artifacts/manifest.json","artifacts/summary.json","artifacts/predictions.jsonl"]
        let paths=Set(files.map(\.path))
        return files.count==paths.count && (paths==original || paths==original.union(["artifacts/prediction-table.json"]))
    }
    private static func path(_ reference:WorkflowEvaluationHistoryReference,_ root:URL)throws->URL {
        guard root.isFileURL,UUID(uuidString:reference.projectID)?.uuidString==reference.projectID else {throw failure("Invalid evaluation history reference.")}
        return root.appendingPathComponent("workflow-evaluations").appendingPathComponent(reference.projectID).appendingPathComponent(reference.evaluationID.uuidString)
    }
    private static func keys(_ data:Data,_ required:Set<String>,optional:Set<String>=[])throws {
        guard let value=try JSONSerialization.jsonObject(with:data) as? [String:Any],required.isSubset(of:Set(value.keys)),Set(value.keys).isSubset(of:required.union(optional)) else {throw failure("Unknown or missing evidence fields.")}
        func checkFiles(_ item:Any)throws {
            if let object=item as? [String:Any] {
                if object["path"] != nil {guard Set(object.keys)==["path","bytes","sha256"] else {throw failure("Unknown artifact inventory fields.")}}
                for child in object.values {try checkFiles(child)}
            } else if let array=item as? [Any] {for child in array {try checkFiles(child)}}
        }
        try checkFiles(value)
    }
    private static func validFile(_ file:WorkflowArchiveFile,limit:Int)throws {
        _ = try WorkflowArchiveCodec.pathParts(file.path)
        guard (0...limit).contains(file.bytes),WorkflowArchiveCodec.isHash(file.sha256) else {throw failure("Invalid artifact inventory entry.")}
    }
    private static func pinned(_ file:WorkflowArchiveFile,path:String,limit:Int,root:URL)throws->Data {
        try validFile(file,limit:limit)
        guard file.path==path else {throw failure("Unexpected captured evidence path.")}
        let bytes=try DiagnosticFileIO.read(root:root,parts:WorkflowArchiveCodec.pathParts(path),limit:limit)
        guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256 else {throw failure("Captured evidence changed: "+path)}
        return bytes
    }
    private static func failure(_ message:String)->WorkflowArchiveError {.init(message)}
}
