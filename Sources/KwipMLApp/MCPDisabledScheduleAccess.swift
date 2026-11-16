import Foundation
import KwipMLCore

/// Metadata-only proposals. This never creates a scheduler, window, destination
/// grant or execution. Durable deduplication is committed by the native ledger.
@MainActor final class MCPDisabledScheduleAccess {
    private struct Pending {let client:String,project:String;let proposal:DisabledScheduleProposal;let expires:Date}
    private var pending:[String:Pending]=[:]
    var now:()->Date = Date.init
    func invalidate(){pending.removeAll()}
    private func fail(_ code:String)throws->Never {throw MCPSettings.Refusal(code:code)}
    func preview(client:String,project:String,root:URL,values:[String:JSONValue],authorize:(Set<String>)throws->Void)async throws->[String:JSONValue] {
        let rule=try Self.rule(values,project:project)
        let proposal:DisabledScheduleProposal
        do {proposal=try await Task.detached {try DisabledScheduleProposal.preview(rule,root:root)}.value}
        catch {try fail("schedule_invalid_or_stale")}
        try authorize(proposal.components)
        pending=pending.filter{$0.value.expires>now()}
        guard pending.count<64 else {try fail("schedule_proposal_limit")}
        let token=UUID().uuidString
        pending[token] = .init(client:client,project:project,proposal:proposal,expires:now().addingTimeInterval(120))
        var result=Self.receipt(rule:proposal.rule,applied:false)
        result["proposalID"] = .string(token);result["expiresInSeconds"]=120
        result["configuredOutputCount"] = .integer(Int64(proposal.effects.count))
        return result
    }
    func apply(client:String,project:String,token:String,requestID:String,digest:String,controller:ScheduledRunController,
               authorize:@escaping @Sendable (Set<String>)throws->Void)async throws->[String:JSONValue] {
        let entry=pending[token]
        let proposal=(entry?.client==client && entry?.project==project && (entry?.expires ?? .distantPast)>now()) ? entry?.proposal:nil
        do {
            let rule=try await controller.applyDisabled(proposal,request:.init(client:client,project:project,id:requestID,digest:digest),authorize:authorize)
            if proposal != nil {pending[token]=nil}
            return Self.receipt(rule:rule,applied:true)
        } catch let error as MCPSettings.Refusal {throw error}
        catch {try fail("schedule_apply_failed_or_stale")}
    }
    static func receipt(rule:ScheduleRule,applied:Bool)->[String:JSONValue] {
        ["projectID":.string(rule.projectID),"ruleID":.string(rule.id),"ruleRevision":.integer(Int64(rule.revision)),
         "versionID":.string(rule.versionID),"versionNumber":.integer(Int64(rule.versionNumber)),"versionHash":.string(rule.versionHash),
         "enabled":false,"outputConsent":false,"hasConfiguredOutputs":.bool(rule.hasEffects),"applied":.bool(applied),
         "requiresOwnerReviewBeforeExecution":true,"executionStarted":false,
         "evidence":.string(applied ? "Disabled schedule write receipt, not current readiness. Identical request replay returns this original receipt even after later native edits. No output access was granted.":"Disabled metadata proposal only. Configured output count is descriptive, not authorization or runtime readiness.")]
    }
    static func rule(_ v:[String:JSONValue],project:String)throws->ScheduleRule {
        func invalid()throws->Never {throw MCPSettings.Refusal(code:"schedule_invalid_arguments")}
        func text(_ key:String)throws->String {guard let s=v[key]?.stringValue,!s.isEmpty,s.count<=512 else {try invalid()};return s}
        func integer(_ key:String)throws->Int {guard case .integer(let n)=v[key],n>=0,n<=Int64(Int.max) else {try invalid()};return Int(n)}
        let required:Set<String>=["versionID","versionNumber","versionHash","name","timing","enabled","target","busy","missed"]
        guard required.isSubset(of:Set(v.keys)),Set(v.keys).isSubset(of:required.union(["ruleID","ruleRevision"])),v["enabled"] == .bool(false),
              (v["ruleID"]==nil)==(v["ruleRevision"]==nil),case .object(let t)=v["timing"],
              let kindText=t["kind"]?.stringValue,let kind=ScheduleTiming.Kind(rawValue:kindText),let zone=t["timeZone"]?.stringValue else {try invalid()}
        let extra:Set<String>
        switch kind {case .once:extra=["anchor"];case .interval:extra=["anchor","intervalMinutes"];case .daily:extra=["hour","minute"];case .weekly:extra=["hour","minute","weekday"]}
        guard Set(t.keys)==extra.union(["kind","timeZone"]) else {try invalid()}
        var timing=ScheduleTiming(kind:kind,timeZone:zone)
        for key in ["hour","minute","weekday","intervalMinutes"] where extra.contains(key) {
            guard case .integer(let n)=t[key],n>=0,n<=10080 else {try invalid()}
            switch key {case "hour":timing.hour=Int(n);case "minute":timing.minute=Int(n);case "weekday":timing.weekday=Int(n);default:timing.intervalMinutes=Int(n)}
        }
        if extra.contains("anchor") {guard let s=t["anchor"]?.stringValue,let date=ISO8601DateFormatter().date(from:s) else {try invalid()};timing.anchor=date}
        let target:WorkflowNodeAddress?
        if v["target"] == .null {target=nil} else {
            guard case .object(let a)=v["target"],Set(a.keys)==["domain","localID"] else {try invalid()}
            target=try JSONDecoder().decode(WorkflowNodeAddress.self,from:JSONEncoder().encode(JSONValue.object(a)))
        }
        guard let busy=ScheduleRule.Busy(rawValue:try text("busy")),let missed=ScheduleRule.Missed(rawValue:try text("missed")) else {try invalid()}
        var rule=ScheduleRule(projectID:project,projectName:"",versionID:try text("versionID"),versionNumber:try integer("versionNumber"),versionHash:try text("versionHash"),name:try text("name"),timing:timing,target:target,busy:busy,missed:missed)
        if v["ruleID"] != nil {rule.id=try text("ruleID");rule.revision=UInt64(try integer("ruleRevision"))}
        try rule.validate();return rule
    }
}

extension MCPDisabledScheduleAccess {
    static var inputSchema:JSONValue {
        let text:JSONValue = .object(["type":.string("string")])
        let integer:JSONValue = .object(["type":.string("integer"),"minimum":0])
        func choice(_ values:[String])->JSONValue {.object(["type":.string("string"),"enum":.array(values.map(JSONValue.string))])}
        return .object(["type":.string("object"),"additionalProperties":false,"required":.array(["versionID","versionNumber","versionHash","name","timing","enabled","target","busy","missed"].map(JSONValue.string)),"properties":.object([
            "ruleID":text,"ruleRevision":integer,"versionID":text,"versionNumber":integer,"versionHash":text,"name":text,
            "enabled":.object(["type":.string("boolean"),"const":false]),"busy":choice(["skip","latest"]),"missed":choice(["skip","latest"]),
            "target":.object(["type":.array([.string("object"),.string("null")]),"additionalProperties":false,"properties":.object(["domain":text,"localID":text]),"required":.array([.string("domain"),.string("localID")])]),
            "timing":.object(["type":.string("object"),"additionalProperties":false,"properties":.object([
                "kind":choice(ScheduleTiming.Kind.allCases.map(\.rawValue)),"timeZone":text,"anchor":.object(["type":.string("string"),"description":.string("ISO8601 date with timezone, e.g. 2026-12-01T09:00:00Z")]),"intervalMinutes":integer,"hour":integer,"minute":integer,"weekday":integer]),
                "required":.array([.string("kind"),.string("timeZone")]),"description":.string("Exactly kind,timeZone plus once:anchor; interval:anchor,intervalMinutes; daily:hour,minute; weekly:hour,minute,weekday. Native timing ranges apply; weekday1=Sunday.")])]),
            "description":.string("Create: omit ruleID and ruleRevision. Edit: supply both from automation_read schedules; only disabled rules may be edited. Pin version identity from automation_read versions. target:null for data-only flows, otherwise exact saved Train/Prebuilt models/Evaluate/Generate address. No credential, output permission, enablement or raw destination fields accepted.")])
    }
}
