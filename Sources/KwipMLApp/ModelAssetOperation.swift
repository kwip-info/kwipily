import Foundation
import KwipMLCore

@MainActor final class ModelAssetOperation {
    struct Result {let directory:URL;let files:[WorkflowArchiveFile];let runtime:RuntimeLocation;let operationID:String}
    private var handle:WorkflowWorkerHandle?
    private var cancelPreparation:(()->Void)?
    private var verification:Task<Void,Error>?
    private var cancelled=false
    private(set) var recoveryDirectory:URL?
    private var activeDirectory:URL?
    private var pauseRequested=false
    func pauseAfterEpoch(){pauseRequested=true;if let activeDirectory {try? DiagnosticFileIO.writeImmutable(Data("pause".utf8),root:activeDirectory,parts:["pause-after-epoch"],limit:64)}}
    nonisolated static func trainingOutput(_ path:String)->Bool {
        let p=path.split(separator:"/").map(String.init)
        guard p.count>=3,p[0]=="output",p[1]=="training" else {return false}
        if p.count==3 {return ["training.json","result-manifest.json"].contains(p[2])}
        if p.count==5,p[2]=="checkpoints",p[3].range(of:"^epoch-[0-9]{6}$",options:.regularExpression) != nil {return ModelFineTuningFiles.leaf(p[4])}
        if p.count==5,p[2]=="model",ModelHubRequest.nestedModelFiles.contains(p.dropFirst(3).joined(separator:"/")) {return true}
        guard p.count==4,["model","latest","base","adapter"].contains(p[2]) else {return false}
        return ModelHubRequest.allowedFile(p[3]) || (p[2]=="latest" && ModelTaskArtifactNames.continuationTensor(p[3])) || ["kwip-manifest.json","adapter_config.json","adapter_model.safetensors","kwip-adapter.json","kwip-continuation.json","training-state.safetensors","training.json","result-manifest.json"].contains(p[3])
    }
    func cancel() {cancelled=true;cancelPreparation?();verification?.cancel();handle?.cancel()}
    func run(action:String,settings:[String:JSONValue],source:URL,files:[WorkflowArchiveFile],projectID:String,root:URL,accountingRoot:URL?,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,extraInputs:[String:Data]=[:],tableInputs:[ModelTableCapture.Input]=[],historyPolicy:RunHistoryPolicy?=nil,budgetWindow:ExecutionBudgetWindow?=nil,adapter:URL?=nil,adapterFiles:[WorkflowArchiveFile]=[],resume:ModelResumeCapture?=nil,resumeWeightsOnly:Bool=false,isCurrent:@escaping()->Bool,onProgress:@escaping(String)->Void,onTrainingProgress:@escaping(TrainingProgressPoint)throws->Void={_ in})async throws->Result {
        guard !cancelled else {throw CancellationError()}
        let operation=UUID().uuidString
        let window=try budgetWindow ?? ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
        struct Prepared {let owner:WorkflowOperationOwner;let request:WorkflowToolRequest;let storage:WorkflowStorageAdmission.Reservation?;let source:WorkflowFrozenSnapshotDirectory}
        let preparation=Task.detached(priority:.userInitiated) {()->Prepared in
            try Task.checkCancellation()
            let tables:[WorkflowArchiveFile]
            if tableInputs.isEmpty {tables=[]} else {tables=try ModelTableCapture.inventory(tableInputs)}
            guard tables.isEmpty || ["train","prepare","evaluate"].contains(action) && extraInputs.isEmpty else {throw WorkflowFileLeaseError.unsafePath}
            struct Identity:Codable {let tableCaptureVersion:Int?;let action:String;let settings:[String:JSONValue];let files:[WorkflowArchiveFile];let extraInputs:[String:String];let runtime:WorkflowRuntimeEvidence}
            var inputHashes=extraInputs.mapValues{WorkflowArchiveCodec.hash($0)}
            for file in tables {inputHashes[file.path]=file.sha256}
            for file in adapterFiles {inputHashes["adapter/"+file.path]=file.sha256}
            for file in resume?.inputFiles(weightsOnly:resumeWeightsOnly) ?? [] {inputHashes[file.path]=file.sha256}
            let runtimeEvidence=try WorkflowRuntimeEvidence.capture(runtime)
            let identity=try WorkflowArchiveCodec.hash(WorkflowArchiveCodec.encode(Identity(tableCaptureVersion:tables.isEmpty ? nil:2,action:action,settings:settings,files:files,extraInputs:inputHashes,runtime:runtimeEvidence)))
            let directory=root.appendingPathComponent("workflow-operations/\(projectID)/\(operation)")
            let inner:[String:JSONValue]=["schemaVersion":.integer(tables.contains(where:{$0.path.hasSuffix(".kwipdata")}) ? 4:resume != nil || ["prepare","evaluate"].contains(action) ? 3:tables.isEmpty ? 1:2),"projectID":.string(projectID),"operationID":.string(operation),"requestIdentity":.string(identity),"action":.string(action),"settings":.object(settings),"sourceIdentity":.string(try ModelAssetFiles.sourceIdentity(files.filter{$0.path != "kwip-manifest.json"}))]
            let bytes=try WorkflowArchiveCodec.encode(inner)
            guard Set(extraInputs.keys).isSubset(of:["input/prompts.table.json","input/training.table.json","input/validation.table.json"]),extraInputs.values.allSatisfy({$0.count<=8_000_000}) else {throw WorkflowFileLeaseError.unsafePath}
            let supplemental=adapterFiles.map{WorkflowArchiveFile(path:"input/adapter/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}
            guard adapterFiles.allSatisfy({ModelAdapterFiles.names.contains($0.path)}),adapterFiles.isEmpty == (adapter==nil) else {throw WorkflowFileLeaseError.unsafePath}
            let additional=extraInputs.sorted{$0.key<$1.key}.map {WorkflowArchiveFile(path:$0.key,bytes:$0.value.count,sha256:WorkflowArchiveCodec.hash($0.value))}
            let captured=(resume?.inputFiles(weightsOnly:resumeWeightsOnly) ?? [])+tables+supplemental+additional+files.map {WorkflowArchiveFile(path:"source/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [.init(path:"input/model-request.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))]
            let request=WorkflowToolRequest(operationID:operation,projectID:projectID,requestIdentity:identity,toolMode:.modelAsset,arguments:["--model-asset",directory.appendingPathComponent("input/model-request.json").path],inputFiles:captured,executionBudget:policy.budget,remainingSeconds:min(Double(action=="train" ? policy.budget.maximumRunSeconds:300),try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)))
            try request.validate(directory:directory,verifyInputs:false)
            let storage:WorkflowStorageAdmission.Reservation?
            if let limit=policy.storageLimitBytes {storage=try WorkflowStorageAdmission(workspace:root,accountingRoot:accountingRoot).reserve(scopes:[.init(namespace:.workflowOperations,components:[projectID,operation])],peakBytes:captured.reduce(Int64(action=="train" ? files.reduce(32_000_000){$0+$1.bytes*16}+Int((settings["checkpointCount"]?.integerValue ?? 0)>0 ? settings["checkpointBytes"]?.integerValue ?? 0:0):action=="export" ? files.reduce(10_000_000){$0+$1.bytes}:16_000_000)){$0+Int64($1.bytes)},limitBytes:limit)} else {storage=nil}
            let owner=try WorkflowOperationOwnership.create(workspace:root,request:request)
            var stagedSource:WorkflowFrozenSnapshotDirectory?
            do {
                if let storage {try storage.bind(to:.init(scope:.init(namespace:.workflowOperations,components:[projectID,operation]),directoryIdentity:owner.metadata.directoryIdentity,ownerLockIdentity:owner.metadata.ownerLockIdentity,activityLockIdentity:owner.metadata.activityLockIdentity))}
                try FileManager.default.createDirectory(at:directory.appendingPathComponent("source"),withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
                let sourceOwnership=try WorkflowFrozenSnapshotDirectory(directory:directory.appendingPathComponent("source"),removeOnDeinit:false);stagedSource=sourceOwnership
                try ModelAssetFiles.copy(files,from:source,to:directory.appendingPathComponent("source"))
                try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:["input","model-request.json"],limit:2_000_000)
                for (name,bytes) in extraInputs {try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:WorkflowArchiveCodec.pathParts(name),limit:8_000_000)}
                if !tables.isEmpty {try ModelTableCapture.copy(tableInputs,files:tables,to:directory.appendingPathComponent("input"))}
                try resume?.copy(to:directory,weightsOnly:resumeWeightsOnly)
                if let adapter {
                    let target=directory.appendingPathComponent("input/adapter")
                    try FileManager.default.createDirectory(at:target,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
                    try ModelAdapterFiles.copy(adapterFiles,from:adapter,to:target)
                }
                if let historyPolicy {try RunHistoryPolicyStore(root:root,projectID:projectID).capture(id:operation,policy:historyPolicy)}
                try owner.publishRequest(request)
                guard try runtimeEvidence.matchesCurrent(runtime) else {throw WorkflowArchiveError("Model runtime changed during capture.")}
                return .init(owner:owner,request:request,storage:storage,source:sourceOwnership)
            } catch {stagedSource?.remove();owner.close();storage?.close();throw error}
        }
        cancelPreparation={preparation.cancel()};defer {cancelPreparation=nil}
        let prepared=try await withTaskCancellationHandler(operation:{try await preparation.value},onCancel:{preparation.cancel()})
        recoveryDirectory=prepared.owner.directory
        activeDirectory=prepared.owner.directory
        if pauseRequested {pauseAfterEpoch()}
        defer{activeDirectory=nil}
        guard !cancelled,isCurrent() else {prepared.source.remove();prepared.owner.close();prepared.storage?.close();throw CancellationError()}
        return try await withTaskCancellationHandler(operation:{
            try await withCheckedThrowingContinuation {continuation in
                var completed:[WorkflowArchiveFile]?,failure:String?
                do {
                    handle=try WorkflowToolExecution.launchPrepared(owner:prepared.owner,request:prepared.request,runtime:runtime,ownershipSHA256:prepared.owner.prepareLaunch(),storageReservation:prepared.storage,finalize:{code,message in
                        guard code==0,isCurrent(),let files=completed else {throw WorkflowArchiveError(failure ?? message)}
                        guard !files.isEmpty,files.count<=512,Set(files.map(\.path)).count==files.count else {throw WorkflowFileLeaseError.unsafePath}
                        let directory=prepared.owner.directory
                        let verification:Task<Void,Error>=Task.detached(priority:.userInitiated) {
                        var total=0
                        for file in files {
                            let allowed=file.path.hasPrefix("output/model/") ? (ModelHubRequest.allowedFile(String(file.path.dropFirst(13))) || file.path=="output/model/kwip-manifest.json"):["output/model-metadata.json","output/tokens.json","output/table.json","output/generation.json"].contains(file.path)
                            guard allowed || (action=="train" && Self.trainingOutput(file.path)) else {throw WorkflowArchiveError("Unreviewed model result path: "+file.path)}
                            total+=file.bytes;guard total <= (action=="train" ? 12_100_000_000:2_020_000_000) else {throw WorkflowFileLeaseError.unsafePath}
                            do {try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:file.path.hasSuffix(".safetensors") ? 2_000_000_000:8_000_000)} catch {throw WorkflowArchiveError("Model result verification failed for "+file.path+": "+error.localizedDescription)}
                        }
                        }
                        self.verification=verification;defer {self.verification=nil}
                        try await withTaskCancellationHandler(operation:{try await verification.value},onCancel:{verification.cancel()})
                        guard !self.cancelled,isCurrent() else {throw CancellationError()}

                    },retainInputsOnSuccess:["train","evaluate"].contains(action),retainModelRecoveryInputs:action=="train" && (settings["checkpointCount"]?.integerValue ?? 0)>0,onLine:{line in
                        do {
                            guard let data=line.data(using:.utf8),let event=try JSONSerialization.jsonObject(with:data) as? [String:Any],let name=event["event"] as? String else {throw WorkflowFileLeaseError.unsafePath}
                            if name=="error" {throw WorkflowArchiveError(event["message"] as? String ?? "Model operation failed.")}
                            if name=="trainingProgress",action=="train" {let point=try TrainingProgressPoint.decode(data);guard point.runID==operation else {throw WorkflowFileLeaseError.unsafePath};try onTrainingProgress(point);return}
                            guard event["operationID"] as? String==operation else {throw WorkflowFileLeaseError.unsafePath}
                            if name=="modelAssetCompleted",completed==nil,event["requestIdentity"] as? String==prepared.request.requestIdentity,let raw=event["files"] {completed=try JSONDecoder().decode([WorkflowArchiveFile].self,from:JSONSerialization.data(withJSONObject:raw))}
                            else if name=="modelAssetProgress",let n=event["completed"] as? Int,let total=event["total"] as? Int {onProgress("Generating · \(n)/\(total) prompts")}
                            else {throw WorkflowFileLeaseError.unsafePath}
                        } catch {failure=error.localizedDescription;self.handle?.cancel()}
                    },onExit:{code,message in
                        self.handle=nil;prepared.source.remove()
                        if code==0,!self.cancelled,isCurrent(),let files=completed {continuation.resume(returning:Result(directory:prepared.owner.directory,files:files,runtime:runtime,operationID:operation))}
                        else {continuation.resume(throwing:WorkflowArchiveError(self.cancelled ? "Model operation cancelled.":failure ?? message))}
                    })
                } catch {prepared.source.remove();prepared.owner.close();prepared.storage?.close();continuation.resume(throwing:error)}
            }
        },onCancel:{Task {@MainActor in self.cancel()}})
    }
}
