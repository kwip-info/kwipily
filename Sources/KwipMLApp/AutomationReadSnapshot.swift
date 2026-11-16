import Foundation
import KwipMLCore

/// Project-scoped, immutable metadata projection. No executing controller is constructed.
/// Recorded enabled/running state is not an observation of service readiness or live execution.
enum AutomationReadSnapshot {
    enum Family:String,CaseIterable,Sendable {case schedules,versions,jobs,endpoints,deliveries}
    struct TargetChoice:Encodable,Sendable {let address:WorkflowNodeAddress;let kind:String}
    struct Record:Encodable,Sendable {
        let id:String
        let projectID:String
        var versionID:String
        var versionNumber:Int
        var versionHash:String
        var target:WorkflowNodeAddress?
        var targetChoices:[TargetChoice]?
        var revision:UInt64?
        var state:String?
        var enabled:Bool?
        var timing:ScheduleTiming?
        var busyPolicy:String?
        var missedPolicy:String?
        var hasEffects:Bool?
        var reviewed:Bool?
        var savedAt:Date?
        var dueAt:Date?
        var receivedAt:Date?
        var nextDue:Date?
        var ruleID:String?
        var endpointID:String?
        var startedAt:Date?
        var finishedAt:Date?
        var runID:String?
        var jobID:String?
        var authentication:String?
        var duplicates:Int?
        var httpStatus:Int?
    }
    struct Item:Sendable {let record:Record;let components:Set<String>}
    struct Snapshot:Sendable {
        let family:Family
        let items:[Item]
        let sourceStatus:String
        let observedAt:Date
        let omittedUnverifiableRecords:Bool
    }
    static func load(root:URL,projectID:String,family:Family)async throws->Snapshot {
        try await Task.detached(priority:.utility){try read(root:root,projectID:projectID,family:family)}.value
    }
    private static func read(root:URL,projectID:String,family:Family)throws->Snapshot {
        guard UUID(uuidString:projectID)?.uuidString==projectID else {throw MCPSettings.Refusal(code:"automation_invalid_arguments")}
        let observed=Date()
        let journal=try CanvasVersionJournal(root:root,projectID:projectID)
        let versions=Dictionary(uniqueKeysWithValues:journal.versions.map{($0.id,$0)})
        var identities:[String:Set<String>]=[:],items:[Item]=[],omitted=false,status="recorded"
        func components(_ id:String)throws->Set<String> {
            if let cached=identities[id]{return cached}
            let values=MCPComponentAccess.identifiers(try journal.document(version:id));identities[id]=values;return values
        }
        func item(_ rule:ScheduleRule,id:String,modify:(inout Record)->Void)throws {
            guard rule.projectID==projectID else{return}
            guard let version=versions[rule.versionID],version.hash==rule.versionHash,version.number==rule.versionNumber else{omitted=true;return}
            var record=Record(id:id,projectID:projectID,versionID:version.id,versionNumber:version.number,versionHash:version.hash,target:rule.target)
            record.ruleID=rule.id
            modify(&record)
            items.append(.init(record:record,components:try components(version.id)))
        }
        switch family {
        case .versions:
            for v in journal.versions {
                var record=Record(id:v.id,projectID:projectID,versionID:v.id,versionNumber:v.number,versionHash:v.hash)
                record.savedAt=v.savedAt
                let details=try ScheduleSavedSource.details(root:root,projectID:projectID,versionID:v.id)
                record.targetChoices=details.targets.map{.init(address:$0.address,kind:$0.kind)}
                // The same verified saved document supplies choices and permission scope.
                items.append(.init(record:record,components:MCPComponentAccess.identifiers(details.document)))
            }
            if items.isEmpty {status="noRecordedVersions"}
        case .schedules,.jobs:
            if let state=try ScheduledRunLedger.readOnlySnapshot(root:root) {
                if family == .schedules {
                    for rule in state.rules where rule.projectID==projectID {
                        try item(rule,id:rule.id){r in r.revision=rule.revision;r.enabled=rule.enabled;r.timing=rule.timing;r.busyPolicy=rule.busy.rawValue;r.missedPolicy=rule.missed.rawValue;r.hasEffects=rule.hasEffects;r.nextDue=rule.nextDue}
                    }
                } else {
                    for job in state.jobs where job.rule.projectID==projectID {
                        try item(job.rule,id:job.id){r in r.state=job.state.rawValue;r.reviewed=job.reviewed;r.dueAt=job.dueAt;r.startedAt=job.startedAt;r.finishedAt=job.finishedAt;r.runID=canonicalID(job.runID)}
                    }
                }
            } else {status="notRecorded"}
        case .endpoints,.deliveries:
            if let state=try WebhookLedger.readOnlySnapshot(root:root) {
                if family == .endpoints {
                    for endpoint in state.endpoints where endpoint.rule.projectID==projectID {
                        try item(endpoint.rule,id:endpoint.id){r in r.revision=endpoint.revision;r.enabled=endpoint.enabled;r.authentication=endpoint.authentication.rawValue;r.busyPolicy=endpoint.busy.rawValue;r.hasEffects=endpoint.rule.hasEffects}
                    }
                } else {
                    for delivery in state.deliveries where delivery.endpoint.rule.projectID==projectID {
                        try item(delivery.endpoint.rule,id:delivery.id){r in r.state=delivery.state.rawValue;r.reviewed=delivery.reviewed;r.receivedAt=delivery.receivedAt;r.endpointID=delivery.endpoint.id;r.finishedAt=delivery.finishedAt;r.jobID=canonicalID(delivery.jobID);r.duplicates=delivery.duplicates;r.httpStatus=delivery.httpStatus}
                    }
                }
            } else {status="notRecorded"}
        }
        return .init(family:family,items:items.sorted{$0.record.id<$1.record.id},sourceStatus:status,observedAt:observed,omittedUnverifiableRecords:omitted)
    }
    private static func canonicalID(_ value:String?)->String? {guard let value,UUID(uuidString:value)?.uuidString==value else{return nil};return value}
}
