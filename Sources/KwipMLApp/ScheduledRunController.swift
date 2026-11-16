import Foundation
import AppKit
import Combine
@preconcurrency import KwipMLCore

/// Admission changes execution availability, never the editable document or viewport.
@MainActor enum ScheduledRunAdmission {
    private final class Weak {weak var value:WorkspaceStore?;init(_ value:WorkspaceStore){self.value=value}}
    private static var workspaces:[Weak]=[]
    private static var owners:[String:UUID]=[:]
    static var hasOwner:Bool {!owners.isEmpty}
    static func register(_ workspace:WorkspaceStore){workspaces.removeAll{$0.value==nil};workspaces.append(Weak(workspace))}
    private static func key(_ workspace:WorkspaceStore)->String {workspace.executionAdmissionKey}
    static func isBlocked(_ workspace:WorkspaceStore)->Bool {if let owner=owners[key(workspace)] {return owner != workspace.scheduledExecutionID};return false}
    static func isBusy(directory:URL)->Bool {
        let key=directory.standardizedFileURL.resolvingSymlinksInPath().path
        return owners[key] != nil || workspaces.contains {weak in
            guard let w=weak.value,self.key(w)==key else{return false}
            return w.isLocallyBusy || w.workflowSession?.hasActiveOperation == true
        }
    }
    static func acquire(_ id:UUID,directory:URL)->Bool {
        guard !isBusy(directory:directory) else{return false}
        owners[directory.standardizedFileURL.resolvingSymlinksInPath().path]=id;notify();return true
    }
    static func release(_ id:UUID,directory:URL) {
        let key=directory.standardizedFileURL.resolvingSymlinksInPath().path
        guard owners[key]==id else{return};owners[key]=nil;notify()
    }
    private static func notify(){for w in workspaces {w.value?.objectWillChange.send()}}
}
struct ScheduleVersionChoice:Identifiable,Equatable,Sendable {
    let id:String;let number:Int;let hash:String;let label:String;let date:Date
}
struct ScheduleTargetChoice:Identifiable,Equatable,Sendable {
    let id:String;let title:String;let address:WorkflowNodeAddress;let kind:String
}
struct ScheduleVersionDetails:Sendable {
    let document:WorkflowDocument
    let targets:[ScheduleTargetChoice]
    let effects:[String]
}

@MainActor final class ScheduledRunController:ObservableObject {
    private static var controllers:[String:ScheduledRunController]=[:]
    static func shared(workspace:WorkspaceStore)->ScheduledRunController {
        let key=workspace.executionAdmissionKey
        if let controller=controllers[key] {if controller.workspace==nil {controller.workspace=workspace};return controller}
        let controller=ScheduledRunController(workspace:workspace);controllers[key]=controller;return controller
    }
    @Published private(set) var state=ScheduledRunState()
    @Published private(set) var error:String?
    @Published private(set) var projects:[ProjectRecord]=[]
    @Published private(set) var activeJobID:String?
    @Published private(set) var phase=""
    private weak var workspace:WorkspaceStore?
    private let root:URL
    private var ledger:ScheduledRunLedger?
    private var timer:Timer?
    private var checking=false
    private var sleeping=false
    private var sleepObservers:[NSObjectProtocol]=[]
    private var activeWorkspace:WorkspaceStore?
    private var activeTask:Task<Void,Never>?
    private var cancelled=false
    var configureJobWorkspace:((WorkspaceStore)->Void)?
    init(workspace:WorkspaceStore,startTimer:Bool=true) {
        self.workspace=workspace;root=workspace.applicationDirectory.standardizedFileURL.resolvingSymlinksInPath()
        Task { [weak self,root] in
            do {
                let ledger=try await Task.detached(priority:.utility){try ScheduledRunLedger(root:root)}.value
                guard let self else{await ledger.close();return};self.ledger=ledger;try await self.refresh()
                if startTimer {
                    let center=NSWorkspace.shared.notificationCenter
                    self.sleepObservers=[
                        center.addObserver(forName:NSWorkspace.willSleepNotification,object:nil,queue:.main) {[weak self] _ in Task {@MainActor in self?.sleeping=true;await self?.ledger?.markUnavailable()}},
                        center.addObserver(forName:NSWorkspace.didWakeNotification,object:nil,queue:.main) {[weak self] _ in Task {@MainActor in await self?.ledger?.markUnavailable();self?.sleeping=false;await self?.tick()}}
                    ]
                    self.timer=Timer.scheduledTimer(withTimeInterval:1,repeats:true) {[weak self] _ in Task {@MainActor in await self?.tick()}}
                }
            } catch {self?.error="Scheduling unavailable; no jobs will start. "+error.localizedDescription}
        }
    }
    func shutdown()async {guard activeJobID==nil else{return};timer?.invalidate();timer=nil;for observer in sleepObservers {NSWorkspace.shared.notificationCenter.removeObserver(observer)};sleepObservers=[];await ledger?.close();ledger=nil}
    func ready()async throws {for _ in 0..<200 {if ledger != nil{return};if let error {throw WorkflowArchiveError(error)};try await Task.sleep(for:.milliseconds(10))};throw WorkflowArchiveError("Scheduling is still loading.")}
    func refresh()async throws {
        guard let ledger else{return};let next=try await ledger.snapshot();if next != state {state=next}
    }
    func loadProjects()async {
        let root=root
        do {projects=try await Task.detached(priority:.utility){try ProjectLibrary(directory:root.appendingPathComponent("projects")).listSources().filter {record in
            guard case .workflow = try? ProjectLibrary(directory:root.appendingPathComponent("projects")).loadSource(id:record.id) else{return false};return true
        }}.value}catch{self.error=error.localizedDescription}
    }
    func versions(projectID:String)async throws->[ScheduleVersionChoice] {
        let root=root
        return try await Task.detached(priority:.userInitiated){try CanvasVersionJournal(root:root,projectID:projectID).versions.map {.init(id:$0.id,number:$0.number,hash:$0.hash,label:$0.label,date:$0.savedAt)}}.value
    }
    func details(projectID:String,versionID:String)async throws->ScheduleVersionDetails {
        let root=root
        return try await Task.detached(priority:.userInitiated) {try ScheduleSavedSource.details(root:root,projectID:projectID,versionID:versionID)}.value
    }
    func save(_ proposed:ScheduleRule)async throws {
        try await ready()
        let root=root
        let details=try await Task.detached(priority:.userInitiated) {try ScheduleSavedSource.validate(proposed,root:root)}.value
        var rule=proposed;rule.hasEffects = !details.effects.isEmpty
        rule.outputAccessHash=rule.hasEffects && rule.allowEffects ? try await outputAccessHash(projectID:rule.projectID):nil
        try await ledger?.save(rule);try await refresh();error=nil
    }
    func outputAccessHash(projectID:String)async throws->String {
        let root=root
        return try await Task.detached(priority:.utility) {try ScheduleSavedSource.outputAccessHash(root:root,projectID:projectID)}.value
    }
    /// Lookup only: MCP must never start a scheduler service to save a draft.
    static func existing(workspace:WorkspaceStore)->ScheduledRunController? {controllers[workspace.executionAdmissionKey]}
    func applyDisabled(_ proposal:DisabledScheduleProposal,authorize:@escaping @Sendable ()throws->Void)async throws->ScheduleRule {
        try await ready()
        guard let ledger else {throw WorkflowArchiveError("Scheduling is unavailable.")}
        let result=try await ledger.saveDisabled(proposal,authorize:authorize)
        try await refresh();return result
    }
    func applyDisabled(_ proposal:DisabledScheduleProposal?,request:DisabledScheduleRequest,authorize:@escaping @Sendable (Set<String>)throws->Void)async throws->ScheduleRule {
        try await ready()
        guard let ledger else {throw MCPSettings.Refusal(code:"schedule_unavailable")}
        let saved=try await ledger.saveDisabled(proposal,request:request,authorize:authorize)
        try await refresh();return saved
    }
    func setEnabled(_ enabled:Bool)async {do{try await ready();try await ledger?.setEnabled(enabled);try await refresh();error=nil}catch{self.error=error.localizedDescription}}
    func remove(_ id:String)async {do{try await ledger?.remove(id);try await refresh()}catch{self.error=error.localizedDescription}}
    func limits(days:Int,maximum:Int)async throws {try await ledger?.limits(days:days,maximum:maximum);try await refresh()}
    func tick(now:Date=Date())async {
        guard !checking,!sleeping,let ledger else{return};checking=true;defer{checking=false}
        do {
            let job=try await ledger.tick(now:now,busy:ScheduledRunAdmission.isBusy(directory:root))
            try await refresh()
            if let job {
                let token=UUID()
                guard ScheduledRunAdmission.acquire(token,directory:root) else {try await ledger.finish(job.id,state:.queued,message:"Waiting for workspace admission.");try await refresh();return}
                activeJobID=job.id;cancelled=false
                activeTask=Task {[self] in await execute(job,token:token)}
            }
        }catch{self.error=error.localizedDescription}
    }
    func runNow(_ id:String)async {
        let token=UUID()
        guard activeJobID==nil,ScheduledRunAdmission.acquire(token,directory:root) else{error="Workspace busy. Wait for the active operation.";return}
        do {try await ready();guard let ledger else{throw WorkflowArchiveError("Scheduling unavailable.")};let job=try await ledger.claimNow(id);activeJobID=job.id;cancelled=false;try await refresh();activeTask=Task {[self] in await execute(job,token:token)}}
        catch {ScheduledRunAdmission.release(token,directory:root);self.error=error.localizedDescription}
    }
    func runOwned(_ request:OwnedScheduleRequest,authorize:@escaping @Sendable (Set<String>)throws->Void)async throws->ScheduledJob {
        try await ready();guard let ledger else {throw MCPSettings.Refusal(code:"schedule_unavailable")}
        if let replay=try await ledger.replayOwned(request,authorize:authorize) {return replay}
        let token=UUID()
        guard activeJobID==nil,ScheduledRunAdmission.acquire(token,directory:root) else {throw MCPSettings.Refusal(code:"schedule_workspace_busy")}
        var claimed:ScheduledJob?
        do {
            let result=try await ledger.claimOwned(request,authorize:authorize)
            guard result.execute else {ScheduledRunAdmission.release(token,directory:root);return result.job}
            claimed=result.job;activeJobID=result.job.id;cancelled=false
            try await refresh()
            activeTask=Task {[self] in await execute(result.job,token:token,delegatedAuthorization:authorize)}
            return result.job
        } catch {
            if let claimed {try? await ledger.finish(claimed.id,state:.blocked,message:"Delegated dispatch could not start. The request will not be retried automatically.")}
            activeJobID=nil;ScheduledRunAdmission.release(token,directory:root);throw error
        }
    }
    func cancelOwned(_ id:String,client:String,project:String,authorize:@escaping @Sendable (Set<String>)throws->Void)async throws->(ScheduledJob,String) {
        try await ready();guard let ledger else {throw MCPSettings.Refusal(code:"schedule_unavailable")}
        let receipt=try await ledger.requestOwnedCancellation(id,client:client,project:project,authorize:authorize)
        if !receipt.job.state.active {return (receipt.job,"alreadyTerminal")}
        guard activeJobID==id else {
            let latest=try await ledger.ownedJob(id,client:client,project:project,authorize:authorize)
            return (latest.job,latest.job.state.active ? "awaitingRecovery":"alreadyTerminal")
        }
        cancel(id)
        // Requested is not joined or rolled back. The normal execute path keeps
        // admission and worker ownership through cooperative termination.
        return (receipt.job,"requested")
    }
    func cancel(_ id:String) {guard activeJobID==id else{return};cancelled=true;activeTask?.cancel();activeWorkspace?.workflowSession?.cancel()}
    func executeWebhook(_ rule:ScheduleRule,jobID:String,payload:Data?,tableNodeID:String?,eventSchema:[String:String]=[:])async throws->ScheduledJob {
        try await ready();let token=UUID()
        guard activeJobID==nil,ScheduledRunAdmission.acquire(token,directory:root) else{throw WebhookRefusal(503,"workspace_busy")}
        do {
            guard let ledger else{throw WebhookRefusal(503,"job_authority_unavailable")}
            let job=try await ledger.claimExternal(rule,jobID:jobID)
            activeJobID=job.id;cancelled=false;try await refresh()
            await execute(job,token:token,eventPayload:payload,eventTable:tableNodeID,eventSchema:eventSchema)
            guard let result=state.jobs.first(where:{$0.id==jobID}) else{throw WebhookRefusal(503,"job_record_unavailable")}
            return result
        }catch{if activeJobID==nil{ScheduledRunAdmission.release(token,directory:root)};throw error}
    }
    private func execute(_ job:ScheduledJob,token:UUID,eventPayload:Data?=nil,eventTable:String?=nil,eventSchema:[String:String]=[:],delegatedAuthorization:(@Sendable (Set<String>)throws->Void)?=nil)async {
        guard let ledger,let template=workspace else {ScheduledRunAdmission.release(token,directory:root);activeJobID=nil;return}
        var retained:WorkspaceStore?
        let staging=root.appendingPathComponent("scheduled-runs/staging/"+job.id)
        defer {
            retained?.workflowSession?.sourceProvider.releaseScheduledSources()
            retained?.workflowSession?.retire();retained?.workflowSession=nil
            activeWorkspace=nil;activeJobID=nil;activeTask=nil;phase=""
            ScheduledRunAdmission.release(token,directory:root)
            // Only the disposable source copy, never run History or user projects.
            Task.detached(priority:.utility) {try? FileManager.default.removeItem(at:staging)}
        }
        var recordsBefore=Set<String>()
        do {
            let details=try await details(projectID:job.rule.projectID,versionID:job.rule.versionID)
            let version=try await versions(projectID:job.rule.projectID).first{$0.id==job.rule.versionID}
            guard version?.hash==job.rule.versionHash,details.document.drafts.isEmpty else{throw WorkflowArchiveError("Pinned saved canvas failed verification.")}
            guard details.effects.isEmpty || job.rule.allowEffects else{throw WorkflowArchiveError("Unattended output destinations have not been authorized for this schedule.")}
            if !details.effects.isEmpty {
                let currentAccess=try await outputAccessHash(projectID:job.rule.projectID)
                guard job.rule.outputAccessHash==currentAccess else{throw WorkflowArchiveError("Output access or a connection changed since this schedule was approved. Review its destinations and Apply the rule again before another run.")}
            }
            let delegatedComponents=MCPComponentAccess.identifiers(details.document)
            try delegatedAuthorization?(delegatedComponents)
            if delegatedAuthorization != nil,cancelled {throw CancellationError()}
            var document=details.document
            if let eventTable,!eventSchema.isEmpty {
                guard let i=document.nodes.firstIndex(where:{$0.address.domain == .data && $0.address.localID==eventTable}),case .dataOperator(DataFlowSourceOperators.tableInput,var config)=document.nodes[i].payload,(try? TableInputConfiguration.format(of:config)) == .json else{throw WebhookRefusal(422,"json_table_input_required")}
                var options=try TableInputConfiguration.options(in:config)
                options["schema"] = .object(eventSchema.mapValues(JSONValue.string));options["schema_overrides"] = .object([:])
                config["options"] = .object(options)
                document.nodes[i].payload = .dataOperator(operatorID:DataFlowSourceOperators.tableInput,config:config)
                try WorkflowCodec.validate(document)
            }
            try await Task.detached(priority:.userInitiated) {
                try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
                _ = try ProjectLibrary(directory:staging).saveSource(.workflow(document),id:document.projectID)
            }.value
            try Task.checkCancellation()
            let w=WorkspaceStore(rootOverride:template.projectRoot,storageOverride:root,resourcesOverride:template.runtime?.kind == .bundled ? Bundle.main.resourceURL:nil,detectCapabilities:false)
            w.scheduledExecutionID=token;retained=w;activeWorkspace=w
            let owner=try WorkflowAuthoringStore(library:ProjectLibrary(directory:staging),projectID:job.rule.projectID,modelRegistry:w.registry,dataRegistry:w.workflowDataRegistry())
            let host=try WorkflowWorkspaceSession(workspace:w,owner:owner);w.workflowSession=host;w.activeProjectID=owner.projectID
            if let eventPayload,let eventTable {
                let address=WorkflowNodeAddress(domain:.data,localID:eventTable)
                guard document.nodes.contains(where:{if $0.address==address,case .dataOperator(DataFlowSourceOperators.tableInput,let c)=$0.payload{return (try? TableInputConfiguration.format(of:c)) == .json};return false}) else{throw WebhookRefusal(422,"json_table_input_required")}
                let file=staging.appendingPathComponent("webhook-event.json")
                try await Task.detached(priority:.utility){try DiagnosticFileIO.writeImmutable(eventPayload,root:staging,parts:["webhook-event.json"],limit:1_048_576)}.value
                guard host.bindings.setPath(file.path,for:address,lease:owner.lease(),expectedRevision:host.bindings.revision) else{throw WorkflowArchiveError(host.bindings.error ?? "Event input could not be bound.")}
            }
            configureJobWorkspace?(w)
            phase="Verifying runtime"
            let verified=await withCheckedContinuation {c in w.refreshCapabilities {c.resume(returning:$0)}}
            guard verified else{throw WorkflowArchiveError(w.capabilityStatus)}
            let policy=try w.executionPolicy(for:owner.projectID)
            let encoded:[String:JSONValue] = ["budget":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(policy.budget)),"storageLimitBytes":policy.storageLimitBytes.map(JSONValue.integer) ?? .null,"preparationCacheLimitBytes":.integer(Int64(policy.preparationCacheLimitBytes))]
            let existing=try await Task.detached {try DatabaseOutputLedger(root:self.root,projectID:job.rule.projectID).load()}.value
            recordsBefore=Set(existing.map(\.id))
            // Refuse pre-existing unresolved outcomes before invoking any flow.
            guard !existing.contains(where:{!$0.acknowledged && ["unknown","intent"].contains($0.receipt["status"]?.stringValue ?? "")}) else {throw WorkflowArchiveError("An output destination has an unresolved earlier receipt. Review it before enabling this schedule.")}
            try await ledger.finish(job.id,state:.running,message:"Preparing saved workflow.",capture:owner.runHistoryPolicy.componentOutputs ? "Project capture enabled":"Optional output capture off",policy:encoded)
            try await refresh();try Task.checkCancellation()
            try delegatedAuthorization?(delegatedComponents)
            if delegatedAuthorization != nil,cancelled {throw CancellationError()}
            // Preparation above awaits runtime, source and ledger work. Native
            // destination consent can change during that interval. Recheck on
            // the main actor immediately before host admission, with no await
            // between this check and run; reuse the same authority as save.
            if !details.effects.isEmpty {
                let currentAccess=try ScheduleSavedSource.outputAccessHash(root:root,projectID:job.rule.projectID)
                guard job.rule.allowEffects,job.rule.outputAccessHash==currentAccess else {throw WorkflowArchiveError("Output access changed during preparation. Review and approve the schedule again.")}
            }
            let accepted:Bool
            if let target=job.rule.target,let node=document.nodes.first(where:{$0.address==target}) {
                switch node.payload {case .evaluateTest,.generateText:_ = owner.select(.init(nodeAddresses:[target]),lease:owner.lease());accepted=host.run(useSelection:true);case .train,.estimator:accepted=host.run(train:target,useSelection:false);default:accepted=false}
            }else{accepted=host.run(useSelection:false)}
            guard accepted else{throw WorkflowArchiveError(host.statusMessage)}
            let identity=host.operationIdentity
            while host.hasActiveOperation {
                if let delegatedAuthorization {
                    do {try delegatedAuthorization(delegatedComponents)}
                    catch {cancelled=true;host.cancel()}
                }
                if Task.isCancelled {host.cancel()}
                let next=host.isFetchingAPISources ? "Fetching API sources":host.statusMessage
                if phase != next {phase=String(next.prefix(500))}
                // Cancellation is cooperative; retain admission until children have joined.
                await Self.joinPause()
            }
            guard host.operationIdentity==identity else{throw WorkflowArchiveError("Job operation identity changed.")}
            let records=try await Task.detached {try DatabaseOutputLedger(root:self.root,projectID:job.rule.projectID).load()}.value.filter{!recordsBefore.contains($0.id)}
            let uncertain=records.contains{!$0.acknowledged && ["unknown","intent"].contains($0.receipt["status"]?.stringValue ?? "")}
            let executionPhase=host.currentExecution?.phase ?? host.currentDataPreparation?.phase ?? host.currentBackendFit?.phase
            let succeeded=job.rule.target.map{host.terminalStatus(for:$0)=="Completed"} ?? (executionPhase == .completed)
            let result:ScheduledJob.State=uncertain ? .needsReview:cancelled ? .cancelled:succeeded ? .completed:.failed
            let trainingOutcome=host.outcomes.first {$0.runID==host.currentExecution?.runID}
            if let problem=trainingOutcome?.recordError {throw WorkflowArchiveError("Run completed but required History could not be recorded: "+problem)}
            let runID=host.currentDataPreparation?.historyOperationID ?? trainingOutcome?.directory.lastPathComponent ?? host.currentBackendFit?.historyOperationID
            let activity=host.currentDataPreparation?.messages ?? host.currentExecution?.console.entries.map(\.message) ?? host.currentBackendFit?.messages ?? []
            let previous=try await ledger.snapshot().jobs.first{$0.id==job.id}
            try await ledger.finish(job.id,state:result,message:uncertain ? "Output delivery needs review. The schedule is paused; inspect receipts and the destination.":host.statusMessage,runID:runID,capture:previous?.captureDescription ?? "",policy:previous?.effectivePolicy ?? [:],activity:owner.runHistoryPolicy.metricsAndDiagnostics ? activity:[],receipts:records.map(\.id))
            try await refresh()
        }catch {
            retained?.workflowSession?.cancel()
            while retained?.workflowSession?.hasActiveOperation==true {await Self.joinPause()}
            let records=(try? await Task.detached {try DatabaseOutputLedger(root:self.root,projectID:job.rule.projectID).load()}.value) ?? []
            let unresolved=records.filter{!$0.acknowledged && ["unknown","intent"].contains($0.receipt["status"]?.stringValue ?? "")}
            do {try await ledger.finish(job.id,state:unresolved.isEmpty ? (cancelled ? .cancelled:.blocked):.needsReview,message:error.localizedDescription,receipts:unresolved.map(\.id));try await refresh()}catch{self.error="Job finalization failed; its durable claim will require recovery review. "+error.localizedDescription}
        }
    }
    private static func joinPause()async {await withCheckedContinuation {c in DispatchQueue.global(qos:.utility).asyncAfter(deadline:.now()+0.2) {c.resume()}}}
    func preparation(for job:ScheduledJob)async->(WorkflowDataPreparationHistory,WorkflowDataPreparationHistory.Entry)? {
        guard let runID=job.runID else{return nil};let history=WorkflowDataPreparationHistory(root:root,projectID:job.rule.projectID)
        await history.reloadAndWait();guard let entry=history.entries.first(where:{$0.id==runID}) else{return nil};return (history,entry)
    }
    func openCanvasCopy(_ job:ScheduledJob)async throws {
        let details=try await details(projectID:job.rule.projectID,versionID:job.rule.versionID),root=root
        let record=try await Task.detached(priority:.userInitiated) {
            var copy=details.document;copy.projectID=UUID().uuidString;copy.name=String((copy.name+" · scheduled v\(job.rule.versionNumber)").prefix(80));copy.guidedBuild=nil
            return try ProjectLibrary(directory:root.appendingPathComponent("projects")).saveSource(.workflow(copy),id:copy.projectID)
        }.value
        _ = try ProjectWindowRegistry.shared.open(record)
    }
    func openProjectHistory(_ job:ScheduledJob) {
        if let record=projects.first(where:{$0.id==job.rule.projectID}) {do {_ = try ProjectWindowRegistry.shared.open(record,section:"History")}catch{self.error=error.localizedDescription}}
    }
    func receipts(for job:ScheduledJob)async throws->[DatabaseOutputLedger.Record] {
        let root=root
        return try await Task.detached(priority:.utility){try DatabaseOutputLedger(root:root,projectID:job.rule.projectID).load().filter{job.receipts.contains($0.id) || job.state == .needsReview && job.receipts.isEmpty && !$0.acknowledged && ["intent","unknown"].contains($0.receipt["status"]?.stringValue ?? "")}}.value
    }
    func review(_ job:ScheduledJob)async throws {
        // The explicit user action acknowledges existing receipts, never resends.
        guard activeJobID==nil else{throw WorkflowArchiveError("Wait for the active job before reviewing an outcome.")}
        let selected=try await receipts(for:job).map(\.id)
        let root=root
        try await Task.detached(priority:.utility) {
            let ledger=DatabaseOutputLedger(root:root,projectID:job.rule.projectID)
            for record in try ledger.load() where selected.contains(record.id) && !record.acknowledged && ["intent","unknown"].contains(record.receipt["status"]?.stringValue ?? "") {try ledger.acknowledge(record.id)}
        }.value
        try await ledger?.review(job.id);try await refresh()
    }
}
