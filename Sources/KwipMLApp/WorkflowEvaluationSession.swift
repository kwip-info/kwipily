import Foundation
import Combine
import CoreFoundation
import KwipMLCore

enum WorkflowEvaluationStatus:String,Codable {case completed,failed,cancelled}
struct WorkflowEvaluationOutcome:Codable {
    var schemaVersion:Int=1
    var kind:String="frozenEvaluationOutcome"
    let evaluationID:String
    let projectID:String
    let requestIdentity:String
    let status:WorkflowEvaluationStatus
    let startedAt:Date
    let finishedAt:Date
    let capture:WorkflowArchiveFile?
    let manifestSHA256:String?
    let artifacts:[WorkflowArchiveFile]
    let error:String?
    var failureCode: WorkflowFailureCode? = nil
}
struct WorkflowEvaluationSessionResult {
    let outcome:WorkflowEvaluationOutcome
    let directory:URL
    let recordError:String?
    let exposureError:String?
}
enum WorkflowEvaluationPhase:String {case idle,capturing,starting,evaluating,cancelling,verifying,completed,failed,cancelled}

/// Keeps both workspace reservation and file ownership until real process exit,
/// independent artifact verification, and immutable terminal publication.
@MainActor final class WorkflowEvaluationSession:ObservableObject {
    let evaluationID: UUID
    let request:WorkflowEvaluationRequest
    @Published private(set) var phase:WorkflowEvaluationPhase = .idle
    @Published private(set) var statusMessage="Ready"
    @Published private(set) var completedCount=0
    @Published private(set) var totalCount=0
    private var completionRecorded=false
    private var completedDirectory:URL?
    private(set) var verifiedResult:WorkflowEvaluationArtifacts.Verified?
    var predictionTableArtifact:TableViewQuerySession.Artifact? {guard phase == .completed,completionRecorded,let completedDirectory else {return nil};return verifiedResult?.predictionTable?.artifact(directory:completedDirectory)}
    private(set) var workerInput:WorkflowEvaluationWorkerInput?
    private(set) var ownershipSHA256:String?
    private let owner:WorkflowAuthoringStore
    private let bindings:WorkflowPrivateFrozenModelBindings
    private let workspace:URL
    private let exposureRoot:URL?
    private let reservation:WorkflowExecutionReservation
    private let launch:WorkflowWorkerLauncher
    private let onTerminal:(WorkflowEvaluationSessionResult)->Void
    private var authority:WorkflowEvaluationOwner?
    private var storageReservation: WorkflowStorageAdmission.Reservation?
    private var reservationID:UUID?
    private var retainedSession:WorkflowEvaluationSession?
    private var work:Task<Void,Never>?
    private var terminalRecordingTask:Task<Void,Never>?
    private var worker:WorkflowWorkerHandle?
    private var startedAt:Date?
    private var accepted=false,finished=false,cancelled=false,exited=false,ready=false,workerStarted=false
    private var failureCode: WorkflowFailureCode?
    private var protocolError:String?
    private var completion:(hash:String,count:Int)?

    private let historyPolicy:RunHistoryPolicy
    init(request:WorkflowEvaluationRequest,owner:WorkflowAuthoringStore,bindings:WorkflowPrivateFrozenModelBindings,
         workspace:URL,exposureRoot:URL? = nil,reservation:WorkflowExecutionReservation,
         launch:@escaping WorkflowWorkerLauncher=WorkflowExecutionSession.launchWorker,
         onTerminal:@escaping(WorkflowEvaluationSessionResult)->Void) {
        self.historyPolicy=request.historyPolicy
        self.evaluationID = request.storageReservation?.evaluationID ?? UUID()
        self.exposureRoot=exposureRoot;self.request=request;self.owner=owner;self.bindings=bindings;self.workspace=workspace
        self.reservation=reservation;self.launch=launch;self.onTerminal=onTerminal
    }
    @discardableResult func start()->Bool {
        guard !accepted,!finished,request.isCurrent(in:owner,bindings:bindings),let token=reservation.acquire() else {return false}
        accepted=true;reservationID=token;retainedSession=self;startedAt=Date();owner.setExecutionBusy(true)
        phase = .capturing;statusMessage="Capturing frozen evaluation inputs…"
        work=Task {
            do {
                try owner.captureHistoryDecision(id:evaluationID.uuidString,policy:historyPolicy)
                let workerRequest=WorkflowEvaluationWorkerRequest(schemaVersion:request.executionBudget == nil ? 1 : 2,evaluationID:evaluationID.uuidString,projectID:request.document.projectID,
                    requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshot.receiptSHA256,settings:request.plan.settings,executionBudget:request.executionBudget,retainComponentOutputs:request.historyPolicy.componentOutputs ? nil:false)
                storageReservation = try request.storageReservation?.take(workspace: workspace,
                    projectID: request.document.projectID, evaluationID: evaluationID)
                let authority=try WorkflowEvaluationOwnership.create(workspace:workspace,request:workerRequest)
                self.authority=authority;ownershipSHA256=authority.ownershipSHA256
                let captured=request,id=evaluationID,directory=authority.directory, storageToken=storageReservation
                let capture=Task.detached(priority:.userInitiated) {
                    if let storageToken {
                        let metadata = authority.metadata
                        try storageToken.bind(to: .init(scope: .init(namespace: .workflowEvaluations,
                            components: [metadata.projectID, metadata.evaluationID]),
                            directoryIdentity: metadata.directoryIdentity, ownerLockIdentity: metadata.ownerLockIdentity,
                            activityLockIdentity: metadata.activityLockIdentity))
                    }
                    let input=try WorkflowEvaluationWorkerInput.publish(request:captured,evaluationID:id,directory:directory,owner:authority)
                    // Staging can be lengthy: recheck runtime after all copies, before launch.
                    try captured.verifyCapturedInputs()
                    return input
                }
                let input=try await withTaskCancellationHandler(operation:{try await capture.value},onCancel:{capture.cancel()})
                workerInput=input
                try Task.checkCancellation();try checkScope()
                var budgetArguments: [String] = []
                if let window = request.budgetWindow {
                    let remaining = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                    budgetArguments = ["--kwip-launch-budget-sha256", try authority.publishLaunchBudget(remainingSeconds: remaining)]
                }
                try authority.prepareLaunch()
                phase = .starting;statusMessage="Starting frozen evaluation…"
                let args=request.runtime.argumentPrefix+["--kwip-evaluate-supervise",directory.path,try input.request.requestSHA256(),authority.ownershipSHA256]+budgetArguments
                work=nil
                let handle=try launch(request.runtime.executableURL,args,request.runtime.workingDirectoryURL,
                    {[weak self] in self?.consume($0)}, {[weak self] in self?.didExit($0,stderr:$1)})
                if !finished && !exited {worker=handle;if cancelled || protocolError != nil {handle.cancel()}}
            } catch {
                if case ExecutionBudgetError.wallTimeLimitExceeded = error { failureCode = .wallTimeLimitExceeded }
                finish(cancelled ? .cancelled:.failed,error:error.localizedDescription)
            }
        }
        return true
    }
    func cancel() {
        guard accepted,!finished,!cancelled else {return}
        cancelled=true;phase = .cancelling;statusMessage="Cancelling; waiting for evaluation to stop…"
        work?.cancel();worker?.cancel()
    }
    private func checkScope()throws {
        guard request.matchesExecutionScope(in:owner,bindings:bindings) else {throw WorkflowArchiveError("Workflow or frozen selection changed during evaluation. Capture again.")}
        try authority?.check()
    }
    private func reject(_ message:String) {
        guard protocolError==nil else {return}
        protocolError=message;statusMessage=message;worker?.cancel()
    }
    private func consume(_ line:String) {
        guard !finished,!exited,!cancelled,protocolError==nil else {return}
        do {
            guard line.utf8.count<=1_000_000,let object=try JSONSerialization.jsonObject(with:Data(line.utf8)) as? [String:Any],let event=object["event"] as? String else {throw WorkflowArchiveError("Malformed evaluation worker event.")}
            func keys(_ expected:Set<String>)throws {guard Set(object.keys)==expected else {throw WorkflowArchiveError("Unexpected evaluation event fields.")}}
            func count(_ key:String)throws->Int {
                guard let n=object[key] as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,n.doubleValue.rounded()==n.doubleValue,n.doubleValue>=0,n.doubleValue<=100_000 else {throw WorkflowArchiveError("Invalid evaluation row count.")}
                return n.intValue
            }
            if event=="error" {
                try keys(["event","message"])
                throw WorkflowArchiveError(String((object["message"] as? String ?? "Evaluation worker failed.").prefix(4096)))
            }
            guard object["evaluationID"] as? String==evaluationID.uuidString else {throw WorkflowArchiveError("Evaluation event belongs to another run.")}
            if event=="evaluationReady" {
                try keys(["event","evaluationID","requestSHA256","ownershipSHA256"])
                guard !ready,let input=workerInput,object["requestSHA256"] as? String == (try input.request.requestSHA256()),object["ownershipSHA256"] as? String==ownershipSHA256 else {throw WorkflowArchiveError("Evaluation readiness does not match captured authority.")}
                try checkScope();ready=true;return
            }
            guard ready,completion==nil,object["requestIdentity"] as? String==request.requestIdentity else {throw WorkflowArchiveError("Evaluation output preceded readiness or followed completion.")}
            switch event {
            case "evaluationStarted":
                try keys(["event","evaluationID","requestIdentity","totalCount"])
                guard !workerStarted else {throw WorkflowArchiveError("Duplicate evaluation start.")}
                totalCount=try count("totalCount");guard totalCount>0 else {throw WorkflowArchiveError("Evaluation partition is empty.")}
                workerStarted=true;phase = .evaluating;statusMessage="Evaluating archived \(request.plan.settings.partitionRole) rows…"
            case "evaluationProgress":
                try keys(["event","evaluationID","requestIdentity","completedCount","totalCount"])
                let n=try count("completedCount")
                guard workerStarted,try count("totalCount")==totalCount,n>=completedCount,n<=totalCount else {throw WorkflowArchiveError("Evaluation progress is inconsistent.")}
                completedCount=n
            case "evaluationCompleted":
                try keys(["event","evaluationID","requestIdentity","manifestSHA256","evaluatedCount"])
                let n=try count("evaluatedCount")
                guard workerStarted,n==totalCount,completedCount==totalCount,let hash=object["manifestSHA256"] as? String,WorkflowArchiveCodec.isHash(hash) else {throw WorkflowArchiveError("Evaluation completion is inconsistent.")}
                completion=(hash,n);statusMessage="Waiting for evaluation worker to exit…"
            default:throw WorkflowArchiveError("Unsupported evaluation event.")
            }
        } catch {reject(error.localizedDescription)}
    }
    private func didExit(_ code:Int32,stderr:String) {
        guard !finished,!exited else {return}
        exited=true;worker=nil
        if cancelled {finish(.cancelled,error:"Evaluation cancelled.");return}
        if code == 124, let policy = request.executionBudget {
            failureCode = .wallTimeLimitExceeded
            finish(.failed, error: ExecutionBudgetError.wallTimeLimitExceeded(seconds: policy.maximumRunSeconds).localizedDescription); return
        }
        if code == 125 {
            failureCode = .resourceBudgetFailed
            finish(.failed, error: stderr.isEmpty ? "The worker could not maintain its resource budget. Inspect the run log before retrying." : String(stderr.prefix(4096))); return
        }
        guard code==0,protocolError==nil,ready,let completion,let input=workerInput,let authority else {
            finish(.failed,error:protocolError ?? String((stderr.isEmpty ? "Evaluation exited without verified completion (status \(code)).":stderr).prefix(4096)));return
        }
        phase = .verifying;statusMessage="Verifying complete evaluation results…"
        work=Task {
            do {
                try checkScope()
                let verify=Task.detached(priority:.userInitiated) {try WorkflowEvaluationArtifacts.verify(request:input.request,snapshot:input.snapshot,root:authority.directory)}
                let value=try await withTaskCancellationHandler(operation:{try await verify.value},onCancel:{verify.cancel()})
                try Task.checkCancellation();try checkScope()
                guard value.manifestSHA256==completion.hash,value.summary.evaluatedCount==completion.count else {throw WorkflowArchiveError("Saved evaluation differs from observed completion.")}
                verifiedResult=value
                finish(.completed,error:nil,value:value)
            } catch {finish(cancelled ? .cancelled:.failed,error:error.localizedDescription)}
            work=nil
        }
    }
    private func finish(_ status:WorkflowEvaluationStatus,error:String?,value:WorkflowEvaluationArtifacts.Verified?=nil) {
        guard !finished else {return}
        finished=true
        let outcome=WorkflowEvaluationOutcome(evaluationID:evaluationID.uuidString,projectID:request.document.projectID,
            requestIdentity:request.requestIdentity,status:status,startedAt:startedAt ?? Date(),finishedAt:Date(),
            capture:workerInput?.capture,manifestSHA256:value?.manifestSHA256,artifacts:value?.files ?? [],error:error.map{String($0.prefix(4096))},failureCode:status == .failed ? failureCode : nil)
        var recordError:String?
        let directory=authority?.directory ?? workspace.appendingPathComponent("workflow-evaluations").appendingPathComponent(request.document.projectID).appendingPathComponent(evaluationID.uuidString)
        do {
            guard let authority else {throw WorkflowArchiveError("Evaluation directory could not be allocated.")}
            try authority.check()
            try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(outcome),root:directory,parts:["outcome.json"],limit:256_000)
        } catch {recordError=error.localizedDescription}
        if status == .completed,recordError==nil,let value {
            // The immutable outcome is the completion commit point. Keep ownership and
            // reservation until recording finishes; never publish raw metrics from here.
            phase = .verifying;statusMessage="Recording evaluation history…"
            let root=workspace, exposureRoot=self.exposureRoot
            terminalRecordingTask=Task {
                let recording=Task.detached(priority:.userInitiated) {()->String? in
                    do {try WorkflowEvaluationDisclosure.recordEvaluation(value,root:root,exposureRoot:exposureRoot);return nil}
                    catch {return error.localizedDescription}
                }
                let exposureError=await recording.value
                publishTerminal(outcome,directory:directory,recordError:nil,exposureError:exposureError)
                terminalRecordingTask=nil
            }
        } else {publishTerminal(outcome,directory:directory,recordError:recordError,exposureError:nil)}
    }
    private func publishTerminal(_ outcome:WorkflowEvaluationOutcome,directory:URL,recordError:String?,exposureError:String?) {
        let status=outcome.status
        completionRecorded=status == .completed && recordError == nil && exposureError == nil
        completedDirectory=completionRecorded ? directory:nil
        phase=recordError == nil ? (status == .completed ? .completed:status == .cancelled ? .cancelled:.failed):.failed
        statusMessage=recordError.map{"Could not save evaluation outcome: \($0)"}
            ?? exposureError.map{"Evaluation saved; results remain undisclosed because exposure tracking is unavailable: \($0)"}
            ?? outcome.error ?? "Evaluation complete; open Results to inspect the saved outcome."
        if request.storageReservation != nil { request.removeSnapshot() }
        authority?.close();authority=nil
        storageReservation?.close();storageReservation=nil
        if let token=reservationID {reservationID=nil;owner.setExecutionBusy(false);reservation.release(token)}
        onTerminal(.init(outcome:outcome,directory:directory,recordError:recordError,exposureError:exposureError))
        retainedSession=nil
    }
}
