import Foundation
import Combine
import KwipMLCore

/// Inference owns one managed operation through exit and immutable output verification.
/// It never launches training or reads original source files.
@MainActor final class WorkflowGenerationSession:ObservableObject {
    @Published private(set) var isBusy=false
    @Published private(set) var statusMessage="Ready"
    @Published private(set) var artifact:TableViewQuerySession.Artifact?
    private(set) var directory:URL?
    private(set) var result:[String:JSONValue]?
    private(set) var outputMetadata:[String:JSONValue]?
    private(set) var record:WorkflowGenerationRecord?
    let address:WorkflowNodeAddress
    let document:WorkflowDocument
    let runtime:RuntimeLocation
    private var task:Task<Void,Never>?
    private var handle:WorkflowWorkerHandle?
    private var failure:String?
    private var completedFiles:[WorkflowArchiveFile]?
    private var cancelled=false
    private var retained:WorkflowGenerationSession?
    init(address:WorkflowNodeAddress,document:WorkflowDocument,runtime:RuntimeLocation) {self.address=address;self.document=document;self.runtime=runtime}
    func start(reference:WorkflowFrozenModelReference,configuration:TextGenerationConfiguration,prompts:TableViewQuerySession.Artifact?,workspace:URL,accountingRoot:URL?,policy:ResolvedExecutionPreferences,reservation:WorkflowExecutionReservation,isCurrent:@escaping()->Bool,onTerminal:@escaping()->Void)throws {
        guard !isBusy,let token=reservation.acquire() else {throw WorkflowArchiveError("Wait for the current workspace operation.")}
        isBusy=true;retained=self;statusMessage="Verifying captured model…"
        task=Task { [self] in
            var storage:WorkflowStorageAdmission.Reservation?
            var operationOwner:WorkflowOperationOwner?
            var stage="Model capture"
            do {
                let window=try ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
                let source=try await WorkflowHistoryVerificationWork.run {
                    guard let id=UUID(uuidString:reference.directoryID),try WorkflowFrozenModelSnapshot.select(.init(projectID:reference.projectID,directoryID:id),root:workspace)==reference else {throw WorkflowArchiveError("Selected model changed; select it again.")}
                    return try WorkflowHistoryReader.stageTrainedExport(.init(projectID:reference.projectID,directoryID:id),root:workspace,accountingRoot:accountingRoot,storageLimitBytes:policy.storageLimitBytes)
                }
                defer {source.remove()}
                try Task.checkCancellation();guard isCurrent() else {throw CancellationError()}
                stage="Generation capture"
                let operation=UUID().uuidString,root=URL(fileURLWithPath:try WorkflowExportPublication.canonicalParentPath(workspace),isDirectory:true),project=document.projectID
                let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
                let sourcePin=WorkflowArchiveCodec.hash(source.manifestBytes)
                struct Identity:Encodable {let operation:String;let source:String;let configuration:[String:JSONValue];let promptPin:String?;let runtime:WorkflowRuntimeEvidence}
                let runtimeEvidence=try WorkflowRuntimeEvidence.capture(runtime)
                let identity=WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(Identity(operation:operation,source:sourcePin,configuration:configuration.values,promptPin:prompts?.sourceSHA256,runtime:runtimeEvidence)))
                let inner:[String:JSONValue] = ["schemaVersion":.integer(1),"projectID":.string(project),"operationID":.string(operation),"requestIdentity":.string(identity),"sourceManifestSHA256":.string(sourcePin),"configuration":.object(configuration.values)]
                let raw=try WorkflowArchiveCodec.encode(inner)
                var files=source.files.map{WorkflowArchiveFile(path:"source/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [.init(path:"source/export-manifest.json",bytes:source.manifestBytes.count,sha256:sourcePin),.init(path:"input/generation.json",bytes:raw.count,sha256:WorkflowArchiveCodec.hash(raw))]
                let kernel=try DiagnosticFileIO.read(root:runtime.workingDirectoryURL,parts:["runner","text_generation.py"],limit:200_000)
                guard runtimeEvidence.workerFiles["runner/text_generation.py"]==WorkflowArchiveCodec.hash(kernel) else {throw WorkflowFileLeaseError.unsafePath}
                files.append(.init(path:"input/generation-kernel.py",bytes:kernel.count,sha256:WorkflowArchiveCodec.hash(kernel)))
                var promptBytes:Data?
                if configuration.tableMode {
                    guard let prompts,prompts.inputKind == "table" else {throw WorkflowArchiveError("Run the prompt source first and connect its complete captured table. Indexed-only View sources are not admitted as prompts.")}
                    let bytes=try DiagnosticFileIO.read(root:prompts.indexURL.deletingLastPathComponent(),parts:[prompts.indexURL.lastPathComponent],limit:8_000_000)
                    guard WorkflowArchiveCodec.hash(bytes)==prompts.sourceSHA256 else {throw WorkflowFileLeaseError.unsafePath}
                    promptBytes=bytes;files.append(.init(path:"input/prompts.table.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)))
                }
                let request=WorkflowToolRequest(operationID:operation,projectID:project,requestIdentity:identity,toolMode:.textGeneration,arguments:["--generate-text",directory.appendingPathComponent("input/generation.json").path],inputFiles:files,executionBudget:policy.budget,remainingSeconds:min(300,try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)))
                try request.validate(directory:directory,verifyInputs:false)
                if let limit=policy.storageLimitBytes {storage=try WorkflowStorageAdmission(workspace:root,accountingRoot:accountingRoot).reserve(scopes:[.init(namespace:.workflowOperations,components:[project,operation])],peakBytes:files.reduce(Int64(80_000_000)){$0+Int64($1.bytes)},limitBytes:limit)}
                stage="Generation ownership"
                let held=try WorkflowOperationOwnership.create(workspace:root,request:request);operationOwner=held
                if let storage {try storage.bind(to:.init(scope:.init(namespace:.workflowOperations,components:[project,operation]),directoryIdentity:held.metadata.directoryIdentity,ownerLockIdentity:held.metadata.ownerLockIdentity,activityLockIdentity:held.metadata.activityLockIdentity))}
                stage="Captured model publication"
                try source.publish(to:held.directory.appendingPathComponent("source"))
                try DiagnosticFileIO.writeImmutable(raw,root:held.directory,parts:["input","generation.json"],limit:2_000_000)
                try DiagnosticFileIO.writeImmutable(kernel,root:held.directory,parts:["input","generation-kernel.py"],limit:200_000)
                if let promptBytes {try DiagnosticFileIO.writeImmutable(promptBytes,root:held.directory,parts:["input","prompts.table.json"],limit:8_000_000)}
                stage="Generation admission"
                try held.publishRequest(request);try Task.checkCancellation();guard isCurrent() else {throw CancellationError()}
                guard try runtimeEvidence.matchesCurrent(runtime) else {throw WorkflowArchiveError("Generation runtime changed during capture.")}
                self.directory=held.directory;statusMessage="Generating…";stage="Generation launch"
                handle=try WorkflowToolExecution.launchPrepared(owner:held,request:request,runtime:runtime,ownershipSHA256:held.prepareLaunch(),storageReservation:storage,finalize:{[self] code,message in
                    guard code==0,!cancelled,failure==nil,isCurrent(),let completedFiles,Set(completedFiles.map(\.path))==["output/table.json","output/generation.json"],completedFiles.count==2 else {throw WorkflowArchiveError(failure ?? (message.isEmpty ? "Generation did not complete.":message))}
                    for f in completedFiles {try WorkflowPreviewToolContract.verifyFile(file:f,directory:held.directory,limit:8_000_000)}
                    guard let table=completedFiles.first(where:{$0.path=="output/table.json"}) else {throw WorkflowFileLeaseError.unsafePath}
                    let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:held.directory,parts:["output","generation.json"],limit:8_000_000))
                    guard result["schemaVersion"] == .integer(1),case .array(let vocabulary)=result["vocabulary"],vocabulary.allSatisfy({$0.stringValue != nil}) else {throw WorkflowFileLeaseError.unsafePath}
                    let tableBytes=try DiagnosticFileIO.read(root:held.directory,parts:["output","table.json"],limit:8_000_000)
                    let tableValue=try JSONDecoder().decode([String:JSONValue].self,from:tableBytes)
                    guard case .array(let rows)=tableValue["rows"],!rows.isEmpty,rows.count<=128,case .array(let generatedRows)=result["rows"],rows.count==generatedRows.count,result["settings"] == (try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(configuration.workerSettings))) else {throw WorkflowFileLeaseError.unsafePath}
                    let record=WorkflowGenerationRecord(id:operation,address:address,document:document,reference:reference,runtime:runtimeEvidence,finishedAt:Date(),files:completedFiles)
                    try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:held.directory,parts:["generation-outcome.json"],limit:2_000_000)
                    self.record=record;self.outputMetadata=tableValue.filter{$0.key != "rows"};self.result=result
                    self.artifact = .init(indexURL:held.directory.appendingPathComponent(table.path),indexSHA256:table.sha256,sourceSHA256:table.sha256,inputKind:"table",tokenVocabulary:.capturedGeneration(directory:held.directory,column:configuration.values["idsColumn"]!.stringValue!,vocabulary:vocabulary.compactMap(\.stringValue)))
                },retainInputsOnSuccess:true,onLine:{[self] line in
                    do {
                        guard let data=line.data(using:.utf8),case .object(let event)=try JSONDecoder().decode(JSONValue.self,from:data),let name=event["event"]?.stringValue else {throw WorkflowFileLeaseError.unsafePath}
                        if name=="error" {throw WorkflowArchiveError(event["message"]?.stringValue ?? "Generation failed.")}
                        guard event["operationID"] == .string(operation) else {throw WorkflowFileLeaseError.unsafePath}
                        if name=="generationProgress",let n=event["completed"]?.integerValue,let total=event["total"]?.integerValue {statusMessage="Generating · \(n)/\(total) prompts"}
                        else if name=="generationCompleted",event["requestIdentity"] == .string(identity),completedFiles==nil,let value=event["files"] {completedFiles=try JSONDecoder().decode([WorkflowArchiveFile].self,from:JSONEncoder().encode(value))}
                        else {throw WorkflowFileLeaseError.unsafePath}
                    } catch {failure=error.localizedDescription;handle?.cancel()}
                },onExit:{[self] code,message in
                    handle=nil;isBusy=false
                    if code != 0 || cancelled {artifact=nil;result=nil;statusMessage=cancelled ? "Generation cancelled.":failure ?? message
                        let failed=WorkflowGenerationRecord(status:cancelled ? "cancelled":"failed",message:statusMessage,id:operation,address:address,document:document,reference:reference,runtime:runtimeEvidence,finishedAt:Date(),files:[])
                        try? DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(failed),root:held.directory,parts:["generation-outcome.json"],limit:2_000_000);record=failed
                        try? FileManager.default.removeItem(at:held.directory.appendingPathComponent("source"));try? FileManager.default.removeItem(at:held.directory.appendingPathComponent("output"))}
                    else {statusMessage="Generation complete"}
                    reservation.release(token);onTerminal();retained=nil
                })
                operationOwner=nil;storage=nil
            } catch {
                if let operationOwner,(try? operationOwner.check()) != nil {try? FileManager.default.removeItem(at:operationOwner.directory.appendingPathComponent("source"));try? FileManager.default.removeItem(at:operationOwner.directory.appendingPathComponent("input"))}
                operationOwner?.close();storage?.close();isBusy=false;statusMessage=stage+": "+error.localizedDescription
                reservation.release(token);onTerminal();retained=nil
            }
            task=nil
        }
    }
    static func restored(_ record:WorkflowGenerationRecord,directory:URL,runtime:RuntimeLocation,verified:WorkflowGenerationRecord.Verified)->WorkflowGenerationSession {
        let session=WorkflowGenerationSession(address:record.address,document:record.document,runtime:runtime)
        session.record=record;session.directory=directory;session.result=verified.result;session.outputMetadata=verified.table.filter{$0.key != "rows"};session.artifact=verified.artifact;session.statusMessage="Captured generation";return session
    }
    func cancel() {cancelled=true;task?.cancel();handle?.cancel()}
}
