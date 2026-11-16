import Foundation
import Darwin
import KwipMLCore

/// The generic retained operation owns the worker until its terminal callback and output verification.
actor WorkflowProgramExportManagedTransport:WorkflowProgramExportTransport {
    private let expectedTorchVersion:String?
    private let accountingRoot:URL?
    private let storageLimitBytes:Int64?
    private let launcher:WorkflowWorkerLauncher
    init(expectedTorchVersion:String?=nil,accountingRoot:URL?=nil,storageLimitBytes:Int64?=nil,launcher:@escaping WorkflowWorkerLauncher=WorkflowExecutionSession.launchWorker) {
        self.expectedTorchVersion=expectedTorchVersion;self.accountingRoot=accountingRoot;self.storageLimitBytes=storageLimitBytes;self.launcher=launcher
    }
    func capture(source:WorkflowTrainedExportPackage,workspace:URL,runtime:RuntimeLocation,
                 configuration:ProgramExportConfiguration,executionBudget:ResolvedExecutionBudget?,
                 budgetWindow:ExecutionBudgetWindow?)async throws->WorkflowProgramExportPackage {
        guard let window=budgetWindow,executionBudget == window.policy,!source.inputContract.isBag,
              let torch=expectedTorchVersion ?? runtime.packageVersions["torch"],!torch.isEmpty,torch.utf8.count <= 128 else {throw WorkflowArchiveError("Program export requires a qualified Torch runtime and the original execution budget window. Packet models use trained packet export.")}
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
        guard let physical=realpath(workspace.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        let root=URL(fileURLWithPath:String(cString:physical));free(physical)
        let project=source.reference.projectID,operation=UUID().uuidString
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        let evidence=try WorkflowRuntimeEvidence.capture(runtime)
        struct Identity:Encodable {let operationID:String;let projectID:String;let source:WorkflowProgramExportIdentity;let sourceManifestSHA256:String;let configuration:[String:JSONValue];let runtime:WorkflowRuntimeEvidence;let budget:ResolvedExecutionBudget}
        let selected=WorkflowProgramExportIdentity(runID:source.runID,requestIdentity:source.requestIdentity,modelManifestSHA256:source.modelManifestSHA256)
        let sourcePin=WorkflowArchiveCodec.hash(source.manifestBytes)
        let identity=WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(Identity(operationID:operation,projectID:project,source:selected,sourceManifestSHA256:sourcePin,configuration:configuration.values,runtime:evidence,budget:window.policy)))
        let inner=WorkflowProgramExportRequest(schemaVersion:2,projectID:project,exportID:operation,requestIdentity:identity,sourceDirectory:directory.appendingPathComponent("source").path,sourceManifestSHA256:sourcePin,configuration:configuration.values,executionBudget:window.policy)
        // The path does not exist until ownership is created. Full decode/validation is mandatory at publication and launch.
        let innerBytes=try WorkflowArchiveCodec.encode(inner),innerPin=WorkflowArchiveCodec.hash(innerBytes)
        let files=source.files.map{WorkflowArchiveFile(path:"source/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [WorkflowArchiveFile(path:"source/export-manifest.json",bytes:source.manifestBytes.count,sha256:sourcePin),WorkflowArchiveFile(path:"input/program-export.json",bytes:innerBytes.count,sha256:innerPin)]
        let request=WorkflowToolRequest(operationID:operation,projectID:project,requestIdentity:identity,toolMode:.programExport,arguments:["--export-program",directory.appendingPathComponent("input/program-export.json").path,"--program-request-sha256",innerPin],inputFiles:files,executionBudget:window.policy,remainingSeconds:try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime))
        try request.validate(directory:directory,verifyInputs:false)
        let reservation:WorkflowStorageAdmission.Reservation?
        if let storageLimitBytes {
            reservation=try WorkflowStorageAdmission(workspace:root,accountingRoot:accountingRoot).reserve(scopes:[.init(namespace:.workflowOperations,components:[project,operation])],peakBytes:files.reduce(Int64(103_310_000)){$0+Int64($1.bytes)},limitBytes:storageLimitBytes)
        } else {reservation=nil}
        let owner:WorkflowOperationOwner
        do {owner=try WorkflowOperationOwnership.create(workspace:root,request:request)} catch {reservation?.close();throw error}
        let lifetime:WorkflowExportStorageLifetime
        do {
            if let reservation {try reservation.bind(to:.init(scope:.init(namespace:.workflowOperations,components:[project,operation]),directoryIdentity:owner.metadata.directoryIdentity,ownerLockIdentity:owner.metadata.ownerLockIdentity,activityLockIdentity:owner.metadata.activityLockIdentity))}
            lifetime=try WorkflowExportStorageLifetime(directory:owner.directory,reservation:reservation)
        } catch {owner.close();reservation?.close();throw error}
        do {
            try Task.checkCancellation()
            try source.publish(to:owner.directory.appendingPathComponent("source"))
            try DiagnosticFileIO.writeImmutable(innerBytes,root:owner.directory,parts:["input","program-export.json"],limit:262_144)
            try owner.publishRequest(request)
            guard try evidence.matchesCurrent(runtime) else {throw WorkflowArchiveError("Worker runtime changed before program export.")}
            _ = try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
            let ownership=try owner.prepareLaunch()
            let relay=await ProgramExportRelay(owner:owner,request:request,runtime:runtime,ownership:ownership,window:window,configuration:configuration,expected:selected,sourcePin:sourcePin,torch:torch,lifetime:lifetime,launcher:launcher)
            return try await withTaskCancellationHandler(operation:{try await relay.run()},onCancel:{Task {@MainActor in relay.cancel()}})
        } catch {owner.close();lifetime.remove();throw error}
    }
}

@MainActor private final class ProgramExportRelay {
    private let owner:WorkflowOperationOwner,request:WorkflowToolRequest,runtime:RuntimeLocation,ownership:String
    private let window:ExecutionBudgetWindow,configuration:ProgramExportConfiguration,expected:WorkflowProgramExportIdentity,sourcePin:String,torch:String,lifetime:WorkflowExportStorageLifetime,launcher:WorkflowWorkerLauncher
    private var handle:WorkflowWorkerHandle?,continuation:CheckedContinuation<WorkflowProgramExportPackage,Error>?
    private var started=false,completion:String?,failure:String?,cancelled=false,result:WorkflowProgramExportPackage?
    private var deadlineTask:Task<Void,Never>?
    init(owner:WorkflowOperationOwner,request:WorkflowToolRequest,runtime:RuntimeLocation,ownership:String,window:ExecutionBudgetWindow,configuration:ProgramExportConfiguration,expected:WorkflowProgramExportIdentity,sourcePin:String,torch:String,lifetime:WorkflowExportStorageLifetime,launcher:@escaping WorkflowWorkerLauncher) {
        self.owner=owner;self.request=request;self.runtime=runtime;self.ownership=ownership;self.window=window;self.configuration=configuration;self.expected=expected;self.sourcePin=sourcePin;self.torch=torch;self.lifetime=lifetime;self.launcher=launcher
    }
    func run()async throws->WorkflowProgramExportPackage {
        try Task.checkCancellation()
        let remaining=try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
        deadlineTask=Task { [weak self] in
            do {try await Task.sleep(nanoseconds:UInt64(remaining * 1_000_000_000))} catch {return}
            guard let self,self.continuation != nil else {return}
            self.failure="Program export exceeded its original execution deadline.";self.handle?.cancel()
        }
        return try await withCheckedThrowingContinuation {continuation in
            self.continuation=continuation
            do {
                handle=try WorkflowToolExecution.launchPrepared(owner:owner,request:request,runtime:runtime,ownershipSHA256:ownership,launcher:launcher,finalize:{[self] code,message in try await finalize(code:code,message:message)},onLine:{[self] in consume($0)},onExit:{[self] in finish(code:$0,message:$1)})
                if cancelled || failure != nil {handle?.cancel()}
            } catch {deadlineTask?.cancel();self.continuation=nil;continuation.resume(throwing:error)}
        }
    }
    func cancel() {cancelled=true;handle?.cancel()}
    private func consume(_ line:String) {
        guard failure == nil else {return}
        do {
            guard let bytes=line.data(using:.utf8),bytes.count <= 1_000_000,case .object(let event)=try JSONDecoder().decode(JSONValue.self,from:bytes),let name=event["event"]?.stringValue else {throw WorkflowFileLeaseError.unsafePath}
            if name == "error" {throw WorkflowArchiveError(event["message"]?.stringValue ?? "Program export failed.")}
            let base:Set<String>=["event","projectID","exportID","requestIdentity"]
            guard event["projectID"] == .string(request.projectID),event["exportID"] == .string(request.operationID),event["requestIdentity"] == .string(request.requestIdentity) else {throw WorkflowFileLeaseError.unsafePath}
            if name == "programExportStarted" {
                guard !started,completion == nil,Set(event.keys) == base else {throw WorkflowFileLeaseError.unsafePath};started=true
            } else if name == "programExportCompleted" {
                guard started,completion == nil,Set(event.keys) == base.union(["manifestSHA256"]),let pin=event["manifestSHA256"]?.stringValue,WorkflowArchiveCodec.isHash(pin) else {throw WorkflowFileLeaseError.unsafePath};completion=pin
            } else {throw WorkflowFileLeaseError.unsafePath}
        } catch {failure=error.localizedDescription;handle?.cancel()}
    }
    private func finalize(code:Int32,message:String)async throws {
        guard code == 0,!cancelled,failure == nil,let completion else {throw WorkflowArchiveError(failure ?? (code == 124 ? "Program export exceeded its execution deadline.":message.isEmpty ? "Program export did not complete.":message))}
        try owner.check();_ = try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
        let directory=owner.directory,expected=self.expected,configuration=self.configuration,sourcePin=self.sourcePin,torch=self.torch
        let task=Task.detached(priority:.userInitiated) {try WorkflowProgramExportPackage.verify(directory:directory.appendingPathComponent("package"),expected:expected,configuration:configuration,sourceManifestSHA256:sourcePin,manifestSHA256:completion,expectedTorchVersion:torch,cleanupDirectory:directory)}
        var package=try await task.value
        try owner.check();_ = try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
        guard !cancelled else {throw CancellationError()}
        package.storageLifetime=lifetime;result=package
    }
    private func finish(code:Int32,message:String) {
        handle=nil;deadlineTask?.cancel();deadlineTask=nil
        guard let continuation else {return};self.continuation=nil
        if code == 0,!cancelled,failure == nil,let result {continuation.resume(returning:result)}
        else if cancelled || (code == 130 && failure == nil) {continuation.resume(throwing:CancellationError())}
        else {continuation.resume(throwing:WorkflowArchiveError(failure ?? (code == 124 ? "Program export exceeded its execution deadline.":message.isEmpty ? "Program export failed verification.":message)))}
    }
}
