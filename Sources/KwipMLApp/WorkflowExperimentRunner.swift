import Foundation
import Combine
import Darwin
import KwipMLCore

/// Serial, stop-on-first-failure execution. The host supplies workspace reservation
/// and private-binding authority; pending runs never mutate the canvas.
@MainActor final class WorkflowExperimentRunner:ObservableObject {
    let group:WorkflowExperimentGroupStore
    @Published private(set) var activeSession:WorkflowExecutionSession?
    @Published private(set) var outcomes:[WorkflowExecutionOutcome]=[]
    @Published private(set) var isRunning=false
    @Published private(set) var error:String?
    private let requests:[WorkflowExecutionRequest]
    private let owner:WorkflowAuthoringStore
    private let groupOwner:WorkflowGroupOwner
    private let workspaceDirectory:URL
    private let reservation:WorkflowExecutionReservation
    private let storageAdmission: WorkflowExperimentStorageAdmission?
    private var admissionTask: Task<Void, Never>?
    private let launch:WorkflowWorkerLauncher
    private let canContinue:()->Bool
    private let onTerminal:()->Void
    private var started=false,finished=false,stopRequested=false
    private var retained:WorkflowExperimentRunner?
    #if KWIPML_RECOVERY_UAT
    private var recoveryUATToken: UUID?
    #endif

    init(requests:[WorkflowExecutionRequest],owner:WorkflowAuthoringStore,group:WorkflowExperimentGroupStore,groupOwner:WorkflowGroupOwner,
         workspaceDirectory:URL,reservation:WorkflowExecutionReservation,
         storageAdmission: WorkflowExperimentStorageAdmission? = nil,
         launch:@escaping WorkflowWorkerLauncher=WorkflowExecutionSession.launchWorker,
         canContinue:@escaping ()->Bool={true},onTerminal:@escaping ()->Void)throws {
        try groupOwner.validateDirectory(group.directory)
        let projectDirectory=group.directory.deletingLastPathComponent()
        let experimentsDirectory=projectDirectory.deletingLastPathComponent()
        let canonicalWorkspace=experimentsDirectory.deletingLastPathComponent()
        guard let supplied=realpath(workspaceDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(supplied)}
        guard String(cString:supplied) == canonicalWorkspace.path,
              experimentsDirectory.lastPathComponent == "workflow-experiments",
              projectDirectory.lastPathComponent == group.record.projectID else {throw WorkflowFileLeaseError.unsafePath}
        guard groupOwner.metadata.groupID == group.record.groupID,
              requests.count == group.record.variants.count,!group.record.stopRequested,
              group.record.variants.allSatisfy({$0.status == .pending}),
              zip(requests,group.record.variants).allSatisfy({request,variant in
                  request.requestIdentity == variant.requestIdentity && request.experiment?.ordinal == variant.ordinal
                  && request.experiment?.groupID == group.record.groupID && request.isCurrent(in:owner)
              }) else {throw DataFlowContractError("Start a new group with its complete, current pending requests.")}
        if let storageAdmission {
            guard storageAdmission.projectID == group.record.projectID, storageAdmission.groupID == group.record.groupID else {
                throw WorkflowStorageAdmission.Failure.invalidScope
            }
        }
        self.storageAdmission = storageAdmission
        self.requests=requests;self.owner=owner;self.group=group;self.workspaceDirectory=canonicalWorkspace
        self.groupOwner=groupOwner
        self.reservation=reservation;self.launch=launch;self.canContinue=canContinue;self.onTerminal=onTerminal
    }
    @discardableResult func start()->Bool {
        guard !started,!finished,!stopRequested else {return false}
        started=true;isRunning=true;retained=self
        advance()
        return error == nil
    }
    func cancel() {
        guard !finished else {return}
        #if KWIPML_RECOVERY_UAT
        if let pause = recoveryUATToken {
            recoveryUATToken = nil
            RecoveryUATGate.shared.cancel(pause)
            return
        }
        #endif
        stopRequested=true
        do {try group.cancelPending()} catch {self.error=error.localizedDescription}
        if let admissionTask { admissionTask.cancel() }
        else if let activeSession {activeSession.cancel()} else {finish()}
    }
    private func advance() {
        guard isRunning,!finished,activeSession == nil,admissionTask == nil else {return}
        do {try groupOwner.check()} catch {fail(error.localizedDescription);return}
        guard !stopRequested,!group.record.stopRequested else {finish();return}
        guard let variant=group.record.variants.first(where:{$0.status == .pending}) else {finish();return}
        let request=requests[variant.ordinal-1]
        guard canContinue(),request.isCurrent(in:owner),!owner.isBusy,!owner.hasPendingInspectorEdits else {
            fail("Workflow or private source authority changed. Remaining experiment runs were stopped.");return
        }
        guard let inputs=request.experimentInputs else {fail("Experiment inputs are unavailable.");return}
        let directoryID=UUID()
        if let storageAdmission {
            admissionTask = Task { [self] in
                do {
                    let counts = try inputs.sources.values.map { try WorkflowCapturedSourceArchive.declaredByteCount($0) }
                    let hasCollections = inputs.sources.values.contains { $0.isFileCollection }
                    let admission = Task.detached(priority: .userInitiated) {
                        try Task.checkCancellation()
                        let peak = try WorkflowExecutionStorageAdmission.peakBytes(sourceByteCounts: counts, checkpointFiles: [], hasFileCollections: hasCollections)
                        try storageAdmission.admitVariant(directoryID: directoryID.uuidString, runPeakBytes: peak)
                    }
                    try await withTaskCancellationHandler(operation: { try await admission.value }, onCancel: { admission.cancel() })
                    try Task.checkCancellation()
                    admissionTask = nil
                    guard !stopRequested, !group.record.stopRequested else { finish(); return }
                    try groupOwner.check()
                    guard canContinue(), request.isCurrent(in: owner), !owner.isBusy, !owner.hasPendingInspectorEdits else {
                        throw DataFlowContractError("Workflow changed during storage admission. Remaining variants were stopped.")
                    }
                    beginVariant(request: request, inputs: inputs, ordinal: variant.ordinal, directoryID: directoryID)
                } catch {
                    admissionTask = nil
                    if stopRequested { finish() } else { fail(error.localizedDescription) }
                }
            }
        } else { beginVariant(request: request, inputs: inputs, ordinal: variant.ordinal, directoryID: directoryID) }
    }
    private func beginVariant(request: WorkflowExecutionRequest, inputs: WorkflowExperimentInputs, ordinal: Int, directoryID: UUID) {
        let directory=workspaceDirectory.appendingPathComponent("workflow-runs/\(owner.projectID)/\(directoryID.uuidString)")
        let session=WorkflowExecutionSession(request:request,owner:owner,sources:inputs.sources,directory:directory,
            reservation:reservation,groupOwner:groupOwner,launch:launch,onTerminal:{[weak self] outcome in self?.received(outcome,ordinal:ordinal,directoryID:directoryID)})
        do {try group.begin(ordinal:ordinal,runID:session.runID,directoryID:directoryID)}
        catch {fail(error.localizedDescription);return}
        activeSession=session
        if !session.start(),activeSession === session {
            activeSession=nil
            do {try group.finish(ordinal:ordinal,runID:session.runID,directoryID:directoryID,
                requestIdentity:request.requestIdentity,status:.failed)} catch {self.error=error.localizedDescription}
            fail(session.statusMessage)
        }
    }
    private func received(_ outcome:WorkflowExecutionOutcome,ordinal:Int,directoryID:UUID) {
        guard !finished,let activeSession,activeSession.runID == outcome.runID,
              requests[ordinal-1].requestIdentity == outcome.request.requestIdentity else {return}
        self.activeSession=nil;outcomes.append(outcome)
        do {
            guard outcome.recordError == nil else {throw DataFlowContractError(outcome.recordError!)}
            let bytes=try DiagnosticFileIO.read(root:outcome.directory,parts:["outcome.json"],limit:256_000)
            let status:WorkflowExperimentVariantStatus
            switch outcome.status {
            case .completed:status = .completed
            case .failed:status = .failed
            case .cancelled:status = .cancelled
            default:throw DataFlowContractError("Worker callback was not terminal.")
            }
            try group.finish(ordinal:ordinal,runID:outcome.runID,directoryID:directoryID,
                requestIdentity:outcome.request.requestIdentity,status:status,outcomeSHA256:WorkflowPhaseArtifacts.hash(bytes))
        } catch {fail("Group result could not be recorded: "+error.localizedDescription);return}
        guard outcome.status == .completed,!stopRequested else {
            if outcome.status == .failed {error=outcome.error ?? "An experiment variant failed."}
            stopRequested=true
            do {try group.cancelPending()} catch {self.error=error.localizedDescription}
            finish();return
        }
        // Yield between variants so a queued cancel or project switch wins before launch.
        #if KWIPML_RECOVERY_UAT
        if group.record.variants.contains(where: { $0.status == .pending }),
           let pause = RecoveryUATGate.shared.hold(.betweenVariants,
            identity: .init(groupID: group.record.groupID, runID: outcome.runID.uuidString, ordinal: ordinal, nonce: nil),
            resume: { [weak self] in
                self?.recoveryUATToken = nil
                Task { [weak self] in self?.advance() }
            }, cancel: { [weak self] in
                self?.recoveryUATToken = nil
                self?.cancel()
            }) {
            recoveryUATToken = pause
            return
        }
        #endif
        Task {[weak self] in self?.advance()}
    }
    private func fail(_ message:String) {
        error=message;stopRequested=true
        do {try group.cancelPending()} catch {self.error=message+" Group stop could not be persisted: "+error.localizedDescription}
        finish()
    }
    private func finish() {
        guard !finished,activeSession == nil,admissionTask == nil else {return}
        finished=true;isRunning=false
        groupOwner.close()
        storageAdmission?.close()
        retained=nil;onTerminal()
    }
}
