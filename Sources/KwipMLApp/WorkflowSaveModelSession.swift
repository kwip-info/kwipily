import Foundation
import Combine
import KwipMLCore

struct WorkflowSaveModelReceipt:Equatable {
    let reference:WorkflowFrozenModelReference
    let format:WorkflowSaveModelConfiguration.Format
    let destination:URL
    let files:[WorkflowArchiveFile]
    var savedAt:Date?=nil
    var bytes:Int64 {files.reduce(0) {$0+Int64($1.bytes)}}
}

/// A staged, verified external artifact. Only the owner performs the final commit.
final class WorkflowSaveModelPrepared {
    let receipt:WorkflowSaveModelReceipt
    private let publication:WorkflowPendingExportPublication
    init(receipt:WorkflowSaveModelReceipt,publication:WorkflowPendingExportPublication) {self.receipt=receipt;self.publication=publication}
    func commit()throws {try publication.commit()}
    func close() {publication.close()}
    deinit {close()}
}

protocol WorkflowSaveModelIO:Actor {
    func stage(reference:WorkflowFrozenModelReference,format:WorkflowSaveModelConfiguration.Format,
               root:URL,destination:URL,accountingRoot:URL?,storageLimitBytes:Int64) async throws->WorkflowSaveModelPrepared
}

actor WorkflowSaveModelWriter:WorkflowSaveModelIO {
    func stage(reference:WorkflowFrozenModelReference,format:WorkflowSaveModelConfiguration.Format,
               root:URL,destination:URL,accountingRoot:URL?,storageLimitBytes:Int64) async throws->WorkflowSaveModelPrepared {
        try await WorkflowHistoryVerificationWork.run {
            try Self.stageVerified(reference:reference,format:format,root:root,destination:destination,
                                   accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
        }
    }
    private static func stageVerified(reference:WorkflowFrozenModelReference,format:WorkflowSaveModelConfiguration.Format,
               root:URL,destination:URL,accountingRoot:URL?,storageLimitBytes:Int64)throws->WorkflowSaveModelPrepared {
        try Task.checkCancellation()
        guard reference.hasValidPins,let directoryID=UUID(uuidString:reference.directoryID),storageLimitBytes>=0 else {throw WorkflowArchiveError("Save model requires an exact verified completed model reference.")}
        let history=WorkflowHistoryReference(projectID:reference.projectID,directoryID:directoryID)
        let workspace=URL(fileURLWithPath:try WorkflowExportPublication.canonicalParentPath(root),isDirectory:true)
        let sourceURL=workspace.appendingPathComponent("workflow-runs").appendingPathComponent(reference.projectID).appendingPathComponent(directoryID.uuidString)
        let source=try WorkflowFrozenSnapshotDirectory(directory:sourceURL,removeOnDeinit:false)
        let pin=try WorkflowStorageAdmission(workspace:workspace,accountingRoot:accountingRoot).reserve(
            scopes:[.init(namespace:.workflowRuns,components:[reference.projectID,directoryID.uuidString])],peakBytes:1,limitBytes:storageLimitBytes)
        defer {pin.close()}
        func verify()throws {
            try Task.checkCancellation();try source.check()
            guard try WorkflowFrozenModelSnapshot.select(history,root:workspace)==reference else {throw WorkflowArchiveError("The selected model's run, outcome, preparation or checkpoint pins changed. Select it again.")}
        }
        try verify()
        let publication:WorkflowPendingExportPublication
        let inventory:[WorkflowArchiveFile]
        switch format {
        case .huggingFace:throw WorkflowArchiveError("This saved run has no Hugging Face architecture adapter. Use Portable package or Checkpoint.")
        case .package:
            let package=try WorkflowHistoryReader.stageTrainedExport(history,root:workspace,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
            defer {package.remove()}
            guard package.reference==history,package.runID==reference.runID,package.requestIdentity==reference.requestIdentity,
                  package.modelManifestSHA256==reference.modelManifestSHA256,
                  package.files.first(where:{$0.path=="checkpoint.pt"})?.sha256==reference.checkpointSHA256 else {throw WorkflowArchiveError("Trained export does not match the captured model reference.")}
            publication=try WorkflowExportPublication.stageExclusive(package,to:destination)
            inventory=package.files+[.init(path:"export-manifest.json",bytes:package.manifestBytes.count,sha256:WorkflowArchiveCodec.hash(package.manifestBytes))]
        case .checkpoint:
            let parent=WorkflowCheckpointReference(projectID:reference.projectID,directoryID:reference.directoryID,
                runID:reference.runID,requestIdentity:reference.requestIdentity,modelManifestSHA256:reference.modelManifestSHA256)
            let snapshot=try WorkflowHistoryReader.stageCheckpoint(parent,root:workspace,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
            defer {snapshot.remove()}
            guard let file=snapshot.files.first(where:{$0.path=="checkpoint.pt"}),file.sha256==reference.checkpointSHA256,
                  file.bytes>0,file.bytes<=100_000_000 else {throw WorkflowArchiveError("The captured training checkpoint is missing or differs from the selected model.")}
            let bytes=try DiagnosticFileIO.read(root:snapshot.directory,parts:["checkpoint.pt"],limit:file.bytes)
            guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==reference.checkpointSHA256 else {throw WorkflowArchiveError("Captured checkpoint bytes changed before saving.")}
            // Copy the verified checkpoint as data. Never invoke a pickle/PyTorch loader.
            publication=try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:reference.checkpointSHA256,to:destination)
            inventory=[.init(path:destination.lastPathComponent,bytes:file.bytes,sha256:file.sha256)]
        }
        do {
            try verify()
            return .init(receipt:.init(reference:reference,format:format,destination:publication.destination,files:inventory),publication:publication)
        } catch {publication.close();throw error}
    }
}

/// Explicit Save model action; graph opening, source selection and History reads never save.
@MainActor final class WorkflowSaveModelSession:ObservableObject {
    @Published private(set) var isBusy=false
    @Published private(set) var savedURL:URL?
    @Published private(set) var receipt:WorkflowSaveModelReceipt?
    @Published private(set) var statusMessage="Choose a new destination, then Save model."
    private let reference:WorkflowFrozenModelReference
    private let format:WorkflowSaveModelConfiguration.Format
    private let root:URL
    private let accountingRoot:URL?
    private let storageLimitBytes:Int64
    private let reservation:WorkflowExecutionReservation
    private let isCurrent:()->Bool
    private let onSaved:(WorkflowSaveModelReceipt)->Void
    private let onFailed:()->Void
    private let io:any WorkflowSaveModelIO
    private var active=true
    private var task:Task<Void,Never>?
    init(reference:WorkflowFrozenModelReference,format:WorkflowSaveModelConfiguration.Format,root:URL,
         accountingRoot:URL?=nil,storageLimitBytes:Int64,reservation:WorkflowExecutionReservation,
         io:any WorkflowSaveModelIO=WorkflowSaveModelWriter(),isCurrent:@escaping()->Bool,
         onSaved:@escaping(WorkflowSaveModelReceipt)->Void={_ in},onFailed:@escaping()->Void={}) {
        self.reference=reference;self.format=format;self.root=root;self.accountingRoot=accountingRoot
        self.storageLimitBytes=storageLimitBytes;self.reservation=reservation;self.io=io;self.isCurrent=isCurrent;self.onSaved=onSaved;self.onFailed=onFailed
    }
    @discardableResult func save(using access:WorkflowModelDestinationAccess)->Bool {
        begin(destination:access.destination,checkAccess:{try access.check()},closeAccess:{access.close()})
    }
    /// Direct callers hold/authorize this exact parent URL, just as the existing export surface.
    @discardableResult func save(to destination:URL,access:URL?=nil)->Bool {
        guard let access else {statusMessage="Saving requires an explicit local destination grant.";return false}
        let started=access.startAccessingSecurityScopedResource()
        do {
            let held=try WorkflowModelDestinationAccess(destination:destination,accessURL:access,revision:0,scopeStarted:started)
            return save(using:held)
        } catch {if started {access.stopAccessingSecurityScopedResource()};statusMessage=error.localizedDescription;return false}
    }
    private func begin(destination:URL,checkAccess:@escaping()throws->Void,closeAccess:@escaping()->Void)->Bool {
        guard active,isCurrent(),!isBusy else {closeAccess();return false}
        guard let token=reservation.acquire() else {closeAccess();statusMessage="Wait for the current workspace operation.";return false}
        isBusy=true;savedURL=nil;receipt=nil;statusMessage="Verifying captured model and preparing save…"
        task=Task { [self] in
            var staged:WorkflowSaveModelPrepared?
            defer {staged?.close();closeAccess();reservation.release(token);task=nil;isBusy=false}
            do {
                try Task.checkCancellation();try checkAccess()
                let prepared=try await io.stage(reference:reference,format:format,root:root,destination:destination,
                    accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
                staged=prepared
                guard active,isCurrent(),prepared.receipt.reference==reference,prepared.receipt.format==format,
                      prepared.receipt.destination.standardizedFileURL.path==destination.standardizedFileURL.path else {throw WorkflowArchiveError("Save model belongs to an earlier source, destination or workflow selection.")}
                try Task.checkCancellation();try checkAccess()
                // No actor suspension between current-authority checks and the atomic rename.
                try prepared.commit()
                var saved=prepared.receipt;saved.savedAt=Date();receipt=saved;savedURL=saved.destination
                statusMessage=format == .package ? "Model package saved · already-prepared inputs required.":"Training checkpoint saved."
                onSaved(saved)
            } catch {
                statusMessage=error is CancellationError ? "Save model cancelled.":"Could not save model: "+error.localizedDescription
                onFailed()
            }
        }
        return true
    }
    func cancel() {task?.cancel();if isBusy {statusMessage="Cancelling model save…"}}
    func invalidate() {active=false;cancel()}
}
