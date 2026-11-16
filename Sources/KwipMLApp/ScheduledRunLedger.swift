import Foundation
import Darwin
@preconcurrency import KwipMLCore

/// Calendar occurrences, not timer intervals. Repeated local times run once; gaps
/// advance to the next valid local time. No polling loop allocates a missed backlog.
struct ScheduleTiming:Codable,Equatable,Sendable {
    enum Kind:String,Codable,CaseIterable,Sendable {case once,interval,daily,weekly}
    var kind:Kind = .daily
    var timeZone:String = TimeZone.current.identifier
    var anchor:Date = Date().addingTimeInterval(60)
    var intervalMinutes:Int = 15
    var hour:Int = 8
    var minute:Int = 30
    var weekday:Int = 2
    func validate()throws {
        guard TimeZone(identifier:timeZone) != nil,anchor.timeIntervalSince1970.isFinite,
              (1...10080).contains(intervalMinutes),(0...23).contains(hour),(0...59).contains(minute),(1...7).contains(weekday) else {throw WorkflowArchiveError("Choose a valid time zone, time and interval of 1–10,080 minutes.")}
    }
    func next(after date:Date)throws->Date? {
        try validate()
        switch kind {
        case .once:return anchor>date ? anchor:nil
        case .interval:
            let seconds=Double(intervalMinutes)*60
            return anchor>date ? anchor:anchor.addingTimeInterval((floor(date.timeIntervalSince(anchor)/seconds)+1)*seconds)
        case .daily,.weekly:
            var calendar=Calendar(identifier:.gregorian);calendar.timeZone=TimeZone(identifier:timeZone)!
            var components=DateComponents();components.hour=hour;components.minute=minute;components.second=0
            if kind == .weekly {components.weekday=weekday}
            return calendar.nextDate(after:date,matching:components,matchingPolicy:.nextTime,repeatedTimePolicy:.first,direction:.forward)
        }
    }
}
struct ScheduleRule:Codable,Equatable,Identifiable,Sendable {
    enum Busy:String,Codable,CaseIterable,Sendable {case skip,latest}
    enum Missed:String,Codable,CaseIterable,Sendable {case skip,latest}
    var id=UUID().uuidString
    var revision:UInt64=0
    var projectID:String
    var projectName:String
    var versionID:String
    var versionNumber:Int
    var versionHash:String
    var name:String
    var timing:ScheduleTiming
    var target:WorkflowNodeAddress? = nil
    var enabled=false
    var busy:Busy = .skip
    var missed:Missed = .skip
    var hasEffects=false
    var allowEffects=false
    var outputAccessHash:String? = nil
    var nextDue:Date? = nil
    func validate()throws {
        try timing.validate()
        guard outputAccessHash == nil || WorkflowArchiveCodec.isHash(outputAccessHash!),revision<UInt64.max-4,[id,projectID,versionID].allSatisfy({UUID(uuidString:$0)?.uuidString==$0}),
              WorkflowArchiveCodec.isHash(versionHash),(1...512).contains(versionNumber),
              !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=120,projectName.count<=200,
              !enabled || !hasEffects || allowEffects else {throw WorkflowArchiveError("Choose a saved workflow version and authorize its configured outputs before enabling the schedule.")}
    }
}
struct ScheduledJob:Codable,Equatable,Identifiable,Sendable {
    enum State:String,Codable,CaseIterable,Sendable {
        case queued,claimed,running,completed,failed,cancelled,skipped,missed,blocked,needsReview
        var active:Bool {[.queued,.claimed,.running].contains(self)}
        var title:String {switch self {case .needsReview:return "Needs review";case .claimed:return "Preparing";default:return rawValue.capitalized}}
    }
    var id=UUID().uuidString
    var occurrence:String
    var rule:ScheduleRule
    var dueAt:Date
    var startedAt:Date? = nil
    var finishedAt:Date? = nil
    var state:State
    var message:String = ""
    var runID:String? = nil
    var phase:String = ""
    var reviewed=false
    var trigger:String = "Scheduled"
    var captureDescription:String = ""
    var effectivePolicy:[String:JSONValue] = [:]
    var activity:[String] = []
    var receipts:[String] = []
    var duration:TimeInterval? {guard let startedAt,let finishedAt else{return nil};return max(0,finishedAt.timeIntervalSince(startedAt))}
}
struct DisabledScheduleRequest:Sendable {
    let client:String,project:String,id:String,digest:String
    var key:String {WorkflowArchiveCodec.hash(Data((client+"\0"+id).utf8))}
}
struct DisabledScheduleReceipt:Codable,Equatable,Sendable {
    let client:String,project:String,digest:String,components:Set<String>,rule:ScheduleRule
}
struct OwnedScheduleRequest:Sendable {
    let client:String,project:String,id:String,digest:String,ruleID:String,versionHash:String
    let revision:UInt64
    var key:String {WorkflowArchiveCodec.hash(Data((client+"\0"+id).utf8))}
}
struct OwnedScheduleReceipt:Codable,Equatable,Sendable {
    let client:String,project:String,digest:String,components:Set<String>
    var job:ScheduledJob
    var cancellationRequested=false
}
struct OwnedScheduleClaim:Sendable {let job:ScheduledJob;let execute:Bool}
struct ScheduledRunState:Codable,Equatable,Sendable {
    var disabledRequests:[String:DisabledScheduleReceipt]? = nil
    var ownedRequests:[String:OwnedScheduleReceipt]? = nil
    var schemaVersion=1
    var enabled=false
    var rules:[ScheduleRule]=[]
    var jobs:[ScheduledJob]=[]
    var lastObservedAt:Date? = nil
    var retentionDays=30
    var maximumJobs=1000
}

/// One local owner. Writes are atomic and precede dispatch; corrupt or exhausted
/// evidence fails closed. Only this actor performs schedule metadata file I/O.
actor ScheduledRunLedger {
    private let root:URL
    private var lease:WorkflowFileLease?
    private var state:ScheduledRunState
    private var closed=false
    private var lastSeenAt:Date?
    private var unavailableSinceLastTick=false
    private static let parts=["scheduled-runs","state.json"]
    /// Observes durable metadata without ownership, recovery, writes or service startup.
    /// The existing family decoder and validator remain the only ledger authority.
    static func readOnlySnapshot(root:URL)throws->ScheduledRunState? {
        guard let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:Self.parts,limit:8_000_000) else {return nil}
        let value=try JSONDecoder().decode(ScheduledRunState.self,from:bytes)
        try Self.validate(value)
        return value
    }
    init(root:URL)throws {
        let canonical=root.standardizedFileURL.resolvingSymlinksInPath()
        self.root=canonical
        let folder=canonical.appendingPathComponent("scheduled-runs")
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        var info=stat()
        guard lstat(folder.path,&info)==0,info.st_mode & S_IFMT == S_IFDIR,let pointer=realpath(folder.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        let resolved=URL(fileURLWithPath:String(cString:pointer));free(pointer)
        lease=try WorkflowFileLease(directory:resolved,name:"owner.lock",mode:.exclusive,create:true)
        var value:ScheduledRunState
        if let bytes=try DiagnosticFileIO.readIfPresent(root:root,parts:Self.parts,limit:8_000_000) {value=try JSONDecoder().decode(ScheduledRunState.self,from:bytes)}else {value=ScheduledRunState()}
        try Self.validate(value)
        for i in value.jobs.indices where value.jobs[i].state == .queued && value.jobs[i].rule.missed == .skip {
            value.jobs[i].state = .missed;value.jobs[i].finishedAt=Date();value.jobs[i].message="App closed before this queued occurrence dispatched."
        }
        for i in value.jobs.indices where [.claimed,.running].contains(value.jobs[i].state) {
            let effects=value.jobs[i].rule.hasEffects
            value.jobs[i].state=effects ? .needsReview:.failed;value.jobs[i].finishedAt=Date()
            value.jobs[i].message=effects ? "Interrupted job. Check output destinations and receipts before enabling this schedule again.":"Interrupted job; it was not automatically restarted."
            if effects,let index=value.rules.firstIndex(where:{$0.id==value.jobs[i].rule.id}) {value.rules[index].enabled=false;value.rules[index].revision+=1}
        }
        for key in Array(value.ownedRequests?.keys ?? Dictionary<String,OwnedScheduleReceipt>().keys) {
            if let id=value.ownedRequests?[key]?.job.id,let job=value.jobs.first(where:{$0.id==id}) {value.ownedRequests?[key]?.job=job}
        }
        state=value;lastSeenAt=value.lastObservedAt
        try DiagnosticFileIO.writeAtomic(try WorkflowArchiveCodec.encode(value),root:root,parts:Self.parts)
    }
    func close(){lease?.close();lease=nil;closed=true}
    func snapshot()throws->ScheduledRunState {guard !closed else{throw WorkflowFileLeaseError.closed};return state}
    private static func validate(_ value:ScheduledRunState)throws {
        guard value.schemaVersion==1,value.rules.count<=128,value.jobs.count<=10000,(1...3650).contains(value.retentionDays),(10...10000).contains(value.maximumJobs),Set(value.rules.map(\.id)).count==value.rules.count,Set(value.jobs.map(\.id)).count==value.jobs.count,Set(value.jobs.map(\.occurrence)).count==value.jobs.count else {throw WorkflowArchiveError("Scheduled-run metadata is invalid or exceeds its bound. Existing evidence is preserved.")}
        guard (value.disabledRequests?.count ?? 0)<=256 else {throw WorkflowArchiveError("Schedule request receipt limit reached.")}
        for (key,receipt) in value.disabledRequests ?? [:] {
            guard WorkflowArchiveCodec.isHash(key),!receipt.client.isEmpty,receipt.client.count<=256,receipt.project==receipt.rule.projectID,
                  WorkflowArchiveCodec.isHash(receipt.digest),!receipt.rule.enabled,!receipt.rule.allowEffects,
                  receipt.rule.outputAccessHash==nil,receipt.components.count<=512,receipt.components.allSatisfy({$0.count<=256}) else {throw WorkflowArchiveError("Schedule request evidence is malformed.")}
            try receipt.rule.validate()
        }
        guard (value.ownedRequests?.count ?? 0)<=256,Set((value.ownedRequests ?? [:]).values.map{ $0.job.id }).count==(value.ownedRequests?.count ?? 0) else {throw WorkflowArchiveError("Owned schedule request limit reached.")}
        for (key,receipt) in value.ownedRequests ?? [:] {
            guard WorkflowArchiveCodec.isHash(key),WorkflowArchiveCodec.isHash(receipt.digest),!receipt.client.isEmpty,receipt.client.count<=256,
                  receipt.project==receipt.job.rule.projectID,UUID(uuidString:receipt.job.id) != nil,receipt.components.count<=512,
                  receipt.components.allSatisfy({$0.count<=256}),receipt.job.activity.count<=200,receipt.job.activity.allSatisfy({$0.count<=1000}),receipt.job.message.count<=4096,receipt.job.phase.count<=100,
                  receipt.job.receipts.count<=128,receipt.job.receipts.allSatisfy({UUID(uuidString:$0) != nil}),!receipt.job.rule.enabled,receipt.job.trigger=="MCP one-shot" else {throw WorkflowArchiveError("Owned schedule evidence is malformed.")}
            try receipt.job.rule.validate()
        }
        for r in value.rules {try r.validate()}
        for j in value.jobs {try j.rule.validate();guard UUID(uuidString:j.id) != nil,j.occurrence.count<=200,j.message.count<=4096,j.phase.count<=100,j.activity.count<=200,j.activity.allSatisfy({$0.count<=1000}),j.receipts.count<=128,j.receipts.allSatisfy({UUID(uuidString:$0) != nil}) else {throw WorkflowArchiveError("Job evidence is malformed.")}}
    }
    private func commit(_ value:ScheduledRunState)throws {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        try Self.validate(value)
        let bytes=try WorkflowArchiveCodec.encode(value)
        guard bytes.count<=8_000_000 else {throw WorkflowArchiveError("Job metadata reached its 8 MB bound. Review History limits; unresolved evidence is preserved.")}
        try DiagnosticFileIO.writeAtomic(bytes,root:root,parts:Self.parts);state=value
    }
    private func prune(_ value:inout ScheduledRunState,now:Date) {
        let cutoff=now.addingTimeInterval(-Double(value.retentionDays)*86400)
        value.jobs.removeAll { !$0.state.active && !($0.state == .needsReview && !$0.reviewed) && ($0.finishedAt ?? $0.dueAt)<cutoff }
        if value.jobs.count>value.maximumJobs {
            var remove=value.jobs.count-value.maximumJobs
            value.jobs=value.jobs.reversed().filter {j in
                if remove>0,!j.state.active,!(j.state == .needsReview && !j.reviewed) {remove-=1;return false};return true
            }.reversed()
        }
    }
    private func append(_ job:ScheduledJob,to value:inout ScheduledRunState,now:Date)throws {
        prune(&value,now:now)
        if value.jobs.count>=value.maximumJobs {
            if let index=value.jobs.lastIndex(where:{!$0.state.active && !($0.state == .needsReview && !$0.reviewed)}) {value.jobs.remove(at:index)}
            else {throw WorkflowArchiveError("Job limit is occupied by active or unresolved jobs. Review them before admitting new work.")}
        }
        value.jobs.insert(job,at:0)
    }
    func saveDisabled(_ proposal:DisabledScheduleProposal,authorize:@Sendable ()throws->Void)throws->ScheduleRule {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        try authorize()
        try proposal.revalidate(root:root)
        guard state.rules.first(where:{$0.id==proposal.rule.id})==proposal.previous else {throw WorkflowArchiveError("Schedule changed in another view.")}
        try save(proposal.rule)
        guard let saved=state.rules.first(where:{$0.id==proposal.rule.id}),!saved.enabled,!saved.allowEffects else {throw WorkflowArchiveError("Disabled schedule receipt unavailable.")}
        return saved
    }
    func saveDisabled(_ proposal:DisabledScheduleProposal?,request:DisabledScheduleRequest,authorize:@Sendable (Set<String>)throws->Void)throws->ScheduleRule {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        guard !request.client.isEmpty,request.client.count<=256,!request.client.contains("\0"),UUID(uuidString:request.id) != nil,WorkflowArchiveCodec.isHash(request.digest) else {throw MCPSettings.Refusal(code:"schedule_invalid_request")}
        if let receipt=state.disabledRequests?[request.key] {
            guard receipt.client==request.client,receipt.project==request.project,receipt.digest==request.digest else {throw MCPSettings.Refusal(code:"schedule_request_conflict")}
            try authorize(receipt.components)
            return receipt.rule
        }
        guard let proposal,proposal.rule.projectID==request.project else {throw MCPSettings.Refusal(code:"schedule_proposal_unavailable")}
        try authorize(proposal.components)
        try proposal.revalidate(root:root)
        guard state.rules.first(where:{$0.id==proposal.rule.id})==proposal.previous else {throw MCPSettings.Refusal(code:"schedule_stale")}
        guard (state.disabledRequests?.count ?? 0)<256 else {throw MCPSettings.Refusal(code:"schedule_request_limit")}
        var next=try saving(proposal.rule,now:Date())
        guard let saved=next.rules.first(where:{$0.id==proposal.rule.id}) else {throw MCPSettings.Refusal(code:"schedule_apply_failed")}
        if next.disabledRequests==nil {next.disabledRequests=[:]}
        next.disabledRequests?[request.key] = .init(client:request.client,project:request.project,digest:request.digest,components:proposal.components,rule:saved)
        try commit(next)
        return saved
    }
    func save(_ rule:ScheduleRule,now:Date=Date())throws {try commit(saving(rule,now:now))}
    private func saving(_ rule:ScheduleRule,now:Date)throws->ScheduledRunState {
        try rule.validate();var next=state;var rule=rule
        guard !rule.enabled || !next.jobs.contains(where:{$0.rule.id==rule.id && $0.state == .needsReview && !$0.reviewed}) else {throw WorkflowArchiveError("Review this schedule's uncertain job before enabling it again.")}
        guard !next.jobs.contains(where:{$0.rule.id==rule.id && [.claimed,.running].contains($0.state)}) else {throw WorkflowArchiveError("Wait for or cancel this schedule's active job before editing it.")}
        rule.nextDue=rule.enabled ? try rule.timing.next(after:now):nil
        guard !rule.enabled || rule.nextDue != nil else {throw WorkflowArchiveError("Choose a future date for a one-time schedule.")}
        if let i=next.rules.firstIndex(where:{$0.id==rule.id}) {
            guard next.rules[i].revision==rule.revision,rule.revision<UInt64.max else {throw WorkflowArchiveError("This schedule changed in another view. Revert to its latest saved settings before applying.")}
            rule.revision+=1;next.rules[i]=rule
        }else {guard next.rules.count<128,rule.revision==0 else{throw WorkflowArchiveError("Schedule limit reached or invalid new rule revision.")};rule.revision=1;next.rules.append(rule)}
        for i in next.jobs.indices where next.jobs[i].rule.id==rule.id && next.jobs[i].state == .queued {next.jobs[i].state = .skipped;next.jobs[i].finishedAt=now;next.jobs[i].message="Schedule changed before dispatch."}
        return next
    }
    func remove(_ id:String)throws {
        guard !state.jobs.contains(where:{$0.rule.id==id && [.claimed,.running].contains($0.state)}) else {throw WorkflowArchiveError("Cancel the active job before removing its rule.")}
        var next=state;next.rules.removeAll{$0.id==id}
        for i in next.jobs.indices where next.jobs[i].rule.id==id && next.jobs[i].state == .queued {next.jobs[i].state = .skipped;next.jobs[i].finishedAt=Date();next.jobs[i].message="Schedule removed; recorded jobs preserved."}
        try commit(next)
    }
    func setEnabled(_ enabled:Bool,now:Date=Date())throws {
        var next=state;next.enabled=enabled;if !enabled || next.lastObservedAt==nil {next.lastObservedAt=now}
        if !enabled {for i in next.jobs.indices where next.jobs[i].state == .queued {next.jobs[i].state = .skipped;next.jobs[i].finishedAt=now;next.jobs[i].message="Workspace scheduling paused."}}
        try commit(next);lastSeenAt=next.lastObservedAt
    }
    func limits(days:Int,maximum:Int,now:Date=Date())throws {
        var next=state;next.retentionDays=days;next.maximumJobs=maximum;try Self.validate(next);prune(&next,now:now);try commit(next)
    }
    func markUnavailable(){unavailableSinceLastTick=true}
    func tick(now:Date,busy:Bool)throws->ScheduledJob? {
        guard state.enabled else{return nil}
        var next=state
        let unavailable=unavailableSinceLastTick || (lastSeenAt.map{now.timeIntervalSince($0)>15} ?? false)
        unavailableSinceLastTick=false
        lastSeenAt=now
        if unavailable {for i in next.jobs.indices where next.jobs[i].state == .queued && next.jobs[i].rule.missed == .skip {next.jobs[i].state = .missed;next.jobs[i].finishedAt=now;next.jobs[i].message="Queued occurrence skipped while the app or Mac was unavailable."}}
        let occupied=busy || next.jobs.contains{[.claimed,.running].contains($0.state)}
        for i in next.rules.indices where next.rules[i].enabled {
            guard let due=next.rules[i].nextDue,due<=now else{continue}
            let rule=next.rules[i],occurrence=rule.id+"/"+String(due.timeIntervalSince1970)
            next.rules[i].nextDue=try rule.timing.next(after:now)
            if next.rules[i].nextDue==nil {next.rules[i].enabled=false;next.rules[i].revision+=1}
            guard !next.jobs.contains(where:{$0.occurrence==occurrence}) else{continue}
            var job=ScheduledJob(occurrence:occurrence,rule:rule,dueAt:due,state:.queued)
            if unavailable && rule.missed == .skip {job.state = .missed;job.message="Missed while the app or Mac was unavailable.";job.finishedAt=now}
            else if occupied && rule.busy == .skip {job.state = .skipped;job.message="Workspace busy; no request sent.";job.finishedAt=now}
            else {
                for j in next.jobs.indices where next.jobs[j].rule.id==rule.id && next.jobs[j].state == .queued {next.jobs[j].state = .skipped;next.jobs[j].finishedAt=now;next.jobs[j].message="Replaced by the latest occurrence."}
            }
            try append(job,to:&next,now:now)
        }
        next.lastObservedAt=now
        var claim:ScheduledJob?
        if !occupied,let index=next.jobs.lastIndex(where:{$0.state == .queued}) {
            // A consumed one-time rule is disabled but its already-queued occurrence is valid.
            next.jobs[index].state = .claimed;next.jobs[index].startedAt=now;claim=next.jobs[index]
        }
        // No periodic disk writes or UI publication while schedules are idle.
        if claim != nil || next.rules != state.rules || next.jobs != state.jobs {try commit(next)}
        return claim
    }
    func claimExternal(_ rule:ScheduleRule,jobID:String,now:Date=Date())throws->ScheduledJob {
        try rule.validate()
        guard UUID(uuidString:jobID) != nil,!state.jobs.contains(where:{$0.id==jobID || [.claimed,.running].contains($0.state)}) else{throw WorkflowArchiveError("External job is already claimed or the workspace is busy.")}
        var next=state
        var job=ScheduledJob(id:jobID,occurrence:rule.id+"/webhook/"+jobID,rule:rule,dueAt:now,startedAt:now,state:.claimed)
        job.trigger="Webhook";try append(job,to:&next,now:now);try commit(next);return job
    }
    func replayOwned(_ request:OwnedScheduleRequest,authorize:@Sendable (Set<String>)throws->Void)throws->ScheduledJob? {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        guard let receipt=state.ownedRequests?[request.key] else {return nil}
        // Authorization precedes receipt identity diagnostics and any returned outcome.
        try authorize(receipt.components)
        guard receipt.client==request.client,receipt.project==request.project,receipt.digest==request.digest else {throw MCPSettings.Refusal(code:"schedule_request_conflict")}
        return receipt.job
    }
    func claimOwned(_ request:OwnedScheduleRequest,authorize:@Sendable (Set<String>)throws->Void,now:Date=Date())throws->OwnedScheduleClaim {
        guard !request.client.isEmpty,request.client.count<=256,!request.client.contains("\0"),UUID(uuidString:request.id) != nil,WorkflowArchiveCodec.isHash(request.digest) else {throw MCPSettings.Refusal(code:"schedule_invalid_request")}
        if let replay=try replayOwned(request,authorize:authorize) {return .init(job:replay,execute:false)}
        guard var rule=state.rules.first(where:{$0.id==request.ruleID && $0.projectID==request.project}),!rule.enabled,
              rule.revision==request.revision,rule.versionHash==request.versionHash else {throw MCPSettings.Refusal(code:"schedule_stale_or_unavailable")}
        let details=try ScheduleSavedSource.validate(rule,root:root),components=MCPComponentAccess.identifiers(details.document)
        try authorize(components)
        guard details.document.drafts.isEmpty,!state.jobs.contains(where:{[.claimed,.running].contains($0.state)}),
              !state.jobs.contains(where:{$0.rule.id==rule.id && $0.state == .needsReview && !$0.reviewed}) else {throw MCPSettings.Refusal(code:"schedule_busy_or_review_required")}
        if !details.effects.isEmpty {
            let currentHash=try ScheduleSavedSource.outputAccessHash(root:root,projectID:rule.projectID)
            guard rule.allowEffects,rule.outputAccessHash==currentHash else {throw MCPSettings.Refusal(code:"schedule_native_output_approval_required")}
        }
        guard (state.ownedRequests?.count ?? 0)<256 else {throw MCPSettings.Refusal(code:"schedule_request_limit")}
        rule.hasEffects = !details.effects.isEmpty
        var next=state,job=ScheduledJob(occurrence:rule.id+"/delegated/"+request.key,rule:rule,dueAt:now,startedAt:now,state:.claimed)
        job.trigger="MCP one-shot"
        try append(job,to:&next,now:now)
        if next.ownedRequests==nil {next.ownedRequests=[:]}
        next.ownedRequests?[request.key] = .init(client:request.client,project:request.project,digest:request.digest,components:components,job:job)
        try commit(next);return .init(job:job,execute:true)
    }
    func ownedJob(_ id:String,client:String,project:String,authorize:@Sendable (Set<String>)throws->Void)throws->OwnedScheduleReceipt {
        guard !closed else {throw WorkflowFileLeaseError.closed}
        guard let receipt=state.ownedRequests?.values.first(where:{$0.job.id==id && $0.client==client && $0.project==project}) else {throw MCPSettings.Refusal(code:"schedule_owned_job_unavailable")}
        try authorize(receipt.components);return receipt
    }
    func requestOwnedCancellation(_ id:String,client:String,project:String,authorize:@Sendable (Set<String>)throws->Void)throws->OwnedScheduleReceipt {
        var receipt=try ownedJob(id,client:client,project:project,authorize:authorize)
        if !receipt.job.state.active || receipt.cancellationRequested {return receipt}
        guard let key=state.ownedRequests?.first(where:{$0.value.job.id==id && $0.value.client==client})?.key else {throw MCPSettings.Refusal(code:"schedule_owned_job_unavailable")}
        receipt.cancellationRequested=true;var next=state;next.ownedRequests?[key]=receipt;try commit(next);return receipt
    }
    func claimNow(_ ruleID:String,now:Date=Date())throws->ScheduledJob {
        guard !state.jobs.contains(where:{[.claimed,.running].contains($0.state)}),let rule=state.rules.first(where:{$0.id==ruleID}),!state.jobs.contains(where:{$0.rule.id==ruleID && $0.state == .needsReview && !$0.reviewed}) else {throw WorkflowArchiveError("Wait for the active job or review the uncertain outcome before running again.")}
        guard !rule.hasEffects || rule.allowEffects else {throw WorkflowArchiveError("Authorize this schedule's output destinations first.")}
        var next=state
        var job=ScheduledJob(occurrence:rule.id+"/manual/"+UUID().uuidString,rule:rule,dueAt:now,startedAt:now,state:.claimed)
        job.trigger="Run now";try append(job,to:&next,now:now);try commit(next);return job
    }
    func finish(_ id:String,state result:ScheduledJob.State,message:String,runID:String?=nil,capture:String="",policy:[String:JSONValue]=[:],activity:[String]=[],receipts:[String]=[],now:Date=Date())throws {
        var next=state;guard let i=next.jobs.firstIndex(where:{$0.id==id}),next.jobs[i].state.active else{throw WorkflowArchiveError("Job is no longer owned by this operation.")}
        next.jobs[i].state=result;next.jobs[i].message=String(message.prefix(4096));next.jobs[i].phase=result.title
        next.jobs[i].runID=runID;next.jobs[i].captureDescription=capture;next.jobs[i].effectivePolicy=policy
        next.jobs[i].activity=activity.suffix(200).map{String($0.prefix(1000))};next.jobs[i].receipts=receipts
        if !result.active {next.jobs[i].finishedAt=now}
        for key in Array(next.ownedRequests?.keys ?? Dictionary<String,OwnedScheduleReceipt>().keys) where next.ownedRequests?[key]?.job.id==id {next.ownedRequests?[key]?.job=next.jobs[i]}
        if result == .needsReview,let r=next.rules.firstIndex(where:{$0.id==next.jobs[i].rule.id}) {next.rules[r].enabled=false;next.rules[r].nextDue=nil;next.rules[r].revision+=1}
        try commit(next)
    }
    func review(_ id:String)throws {
        var next=state;guard let i=next.jobs.firstIndex(where:{$0.id==id}),next.jobs[i].state == .needsReview else {throw WorkflowArchiveError("No uncertain job to review.")}
        next.jobs[i].reviewed=true;try commit(next)
    }
}
