import Foundation
import Darwin
import KwipMLCore

/// Reconciles recorded work only after the kernel proves that every protocol
/// owner/worker has released it. Never launches work or rewrites run artifacts.
enum WorkflowExperimentRecoveryCoordinator {
    enum Availability:Equatable {
        case inUse,recoverable,terminal,unavailable(String)
    }
    private struct Capture {
        let recovery:WorkflowGroupRecovery
        let original:Data
        let context:WorkflowExperimentHistoryReader.CapturedContext
        let canonicalRoot:URL
        let revocations:[(String,String,String)]
    }
    /// Observes only. No capability is retained after a status is returned.
    static func probe(directory:URL,workspaceDirectory:URL,checkCancellation:()throws->Void = {try Task.checkCancellation()})throws->Availability {
        do {
            let captured=try capture(directory:directory,workspaceDirectory:workspaceDirectory,checkCancellation:checkCancellation)
            defer {captured.recovery.close()}
            try checkCancellation()
            return captured.context.record.isTerminal ? .terminal:.recoverable
        } catch is CancellationError {throw CancellationError()}
        catch WorkflowFileLeaseError.busy {return .inUse}
        catch {return .unavailable(error.localizedDescription)}
    }
    private static func capture(directory:URL,workspaceDirectory:URL,checkCancellation:()throws->Void)throws->Capture {
        try checkCancellation()
        let groupID=directory.lastPathComponent
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:groupID)
        do {
            try recovery.validateDirectory(directory)
            let canonicalRoot=directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            guard let path=realpath(workspaceDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
            defer {free(path)}
            guard String(cString:path) == canonicalRoot.path,
                  directory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-experiments",
                  let original=try recovery.readGroupRecord() else {throw WorkflowFileLeaseError.unsafePath}
            let context=try WorkflowExperimentHistoryReader.captureContext(directory:directory)
            let record=context.record
            guard try JSONDecoder().decode(WorkflowExperimentGroupRecord.self,from:original) == record,
                  record.groupID == recovery.metadata.groupID,
                  directory.deletingLastPathComponent().lastPathComponent == record.projectID else {throw WorkflowFileLeaseError.unsafePath}
            var revocations:[(String,String,String)]=[]
            // Validate every deterministic phase slot before publishing any revocation.
            for variant in record.variants {
                try checkCancellation()
                guard let runID=variant.runID else {continue}
                var preparationClaimed=false
                for phase in [WorkflowLaunchIntent.Phase.preparation,.training] {
                    guard let slot=try recovery.readPhaseSlot(runID:runID,phase:phase.rawValue) else {continue}
                    let intent=try WorkflowLaunchIntent.decode(slot)
                    try WorkflowLaunchIntent.validate(intent,groupDirectory:directory)
                    guard intent.projectID == record.projectID,intent.groupID == record.groupID,
                          intent.generation == recovery.metadata.generation,intent.ownershipSHA256 == recovery.ownershipSHA256,
                          intent.ordinal == variant.ordinal,intent.runID == runID,intent.directoryID == variant.directoryID,
                          intent.requestIdentity == variant.requestIdentity,intent.phase == phase,
                          phase != .training || preparationClaimed else {throw WorkflowFileLeaseError.unsafePath}
                    let launch=try recovery.readLaunch(nonce:intent.nonce)
                    guard launch == nil || launch == slot else {throw WorkflowFileLeaseError.unsafePath}
                    let hash=WorkflowPhaseArtifacts.hash(slot)
                    if let raw=try recovery.readLaunchClaim(nonce:intent.nonce) {
                        guard let object=try JSONSerialization.jsonObject(with:raw) as? [String:Any],
                              Set(object.keys) == ["schemaVersion","nonce","intentSHA256","state"] else {throw WorkflowFileLeaseError.unsafePath}
                        let claim=try JSONDecoder().decode(WorkflowLaunchClaim.self,from:raw)
                        guard claim.schemaVersion == 1,claim.nonce == intent.nonce,claim.intentSHA256 == hash,
                              ["claimed","revoked"].contains(claim.state),launch != nil || claim.state == "revoked" else {throw WorkflowFileLeaseError.unsafePath}
                        if phase == .preparation {preparationClaimed=claim.state == "claimed"}
                    } else {revocations.append((runID,phase.rawValue,hash))}
                }
            }
            try checkCancellation()
            guard try recovery.readGroupRecord() == original else {throw WorkflowFileLeaseError.unsafePath}
            return Capture(recovery:recovery,original:original,context:context,canonicalRoot:canonicalRoot,revocations:revocations)
        } catch {recovery.close();throw error}
    }
    static func recover(directory:URL,workspaceDirectory:URL,cleanupOwnedTemporaries:Bool = true,checkCancellation:()throws->Void = {try Task.checkCancellation()})throws->WorkflowExperimentGroupRecord {
        let captured=try capture(directory:directory,workspaceDirectory:workspaceDirectory,checkCancellation:checkCancellation)
        let recovery=captured.recovery,context=captured.context,original=captured.original
        let canonicalRoot=captured.canonicalRoot,revocations=captured.revocations,record=context.record
        defer {recovery.close()}
        var notes:[WorkflowExperimentRecoveryReceipt.Variant]=[]
        var next=record
        next.stopRequested=record.stopRequested || !record.isTerminal
        for index in next.variants.indices {
            try checkCancellation()
            switch next.variants[index].status {
            case .pending:
                next.variants[index].status = .cancelled
                notes.append(.init(ordinal:index+1,priorStatus:.pending,status:.cancelled,reason:.pendingCancelled,
                    detail:"This variant had not started. Recovery cancelled it; no model was launched."))
            case .running:
                do {
                    let verified=try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:index+1,
                        workspaceDirectory:canonicalRoot,policy:.recoveryRunning)
                    guard let status=WorkflowExperimentVariantStatus(rawValue:verified.run.outcome.status.rawValue),
                          [.completed,.failed,.cancelled].contains(status) else {throw WorkflowFileLeaseError.unsafePath}
                    next.variants[index].status=status
                    next.variants[index].outcomeSHA256=verified.outcomeSHA256
                    notes.append(.init(ordinal:index+1,priorStatus:.running,status:status,reason:.verifiedOutcomeRecovered,
                        detail:"Independently verified the saved terminal outcome, variant identity, captured inputs and required artifacts."))
                } catch is CancellationError {throw CancellationError()}
                catch {
                    next.variants[index].status = .interrupted
                    next.variants[index].outcomeSHA256=nil
                    notes.append(.init(ordinal:index+1,priorStatus:.running,status:.interrupted,reason:.terminalEvidenceUnavailable,
                        detail:WorkflowExperimentRecoveryReceipt.boundedDetail(error.localizedDescription)))
                }
            default:break
            }
        }
        if !notes.isEmpty {
            next.recoveryReceipt = .init(schemaVersion:1,timestampUnixSeconds:Date().timeIntervalSince1970,
                priorRevision:record.revision,variants:notes)
        }
        try checkCancellation()
        guard try recovery.readGroupRecord() == original else {throw WorkflowFileLeaseError.unsafePath}
        for (runID,phase,hash) in revocations {
            try checkCancellation()
            try recovery.revokePhase(runID:runID,phase:phase,intentSHA256:hash)
        }
        try checkCancellation()
        let committed = try WorkflowExperimentGroupStore.commitRecovery(expected:original,proposed:next,directory:directory,recovery:recovery)
        // Reconciliation is already durable. Optional cleanup refusal must retain
        // evidence without relabeling that successful recovery as a failure.
        if cleanupOwnedTemporaries { _ = try? recovery.cleanupPublicationTemporaries() }
        return committed
    }
}
