import Foundation
import KwipMLCore

/// Shared saved-source validation for native schedule editing and disabled MCP drafts.
/// Reading a proposal never constructs a controller, opens a window or acquires a ledger.
enum ScheduleSavedSource {
    static func details(root:URL,projectID:String,versionID:String)throws->ScheduleVersionDetails {
        let document=try CanvasVersionJournal(root:root,projectID:projectID).document(version:versionID)
        let targets=document.nodes.compactMap {node->ScheduleTargetChoice? in
            let fallback:String,kind:String
            switch node.payload {case .train:fallback="Train";kind="train";case .estimator:fallback="Fit model";kind="estimator";case .evaluateTest:fallback="Evaluate";kind="evaluate";case .generateText:fallback="Continue a prompt";kind="generate";default:return nil}
            return .init(id:node.address.domain.rawValue+"/"+node.address.localID,title:node.name.isEmpty ? fallback:node.name,address:node.address,kind:kind)
        }
        let effects=document.nodes.compactMap {node->String? in
            switch node.payload {case .saveModel:return (node.name.isEmpty ? "Save model":node.name)+" · model export"
            case .dataOperator(let id,let c) where id=="kwip.data.outputDelimited":return (node.name.isEmpty ? "Table Output":node.name)+" · "+(c["destination"]?.stringValue ?? "file")
            default:return nil}
        }
        return .init(document:document,targets:targets,effects:effects)
    }
    static func validate(_ rule:ScheduleRule,root:URL)throws->ScheduleVersionDetails {
        let details=try details(root:root,projectID:rule.projectID,versionID:rule.versionID)
        guard let version=try CanvasVersionJournal(root:root,projectID:rule.projectID).versions.first(where:{$0.id==rule.versionID}),version.hash==rule.versionHash,version.number==rule.versionNumber else {throw WorkflowArchiveError("Saved canvas identity changed. Choose the version again.")}
        if !details.targets.isEmpty {guard let target=rule.target,details.targets.contains(where:{$0.address==target}) else {throw WorkflowArchiveError("Choose the saved workflow's run target.")}}
        return details
    }
    static func outputAccessHash(root:URL,projectID:String)throws->String {
        let files=[["workflow-output-destinations",projectID,"bindings.json"],["workflow-model-destinations",projectID,"bindings.json"],["database-output-bindings",projectID,"bindings.json"],["database-connections","profiles.json"]]
        var pins:[String:String]=[:]
        for parts in files {let data=try DiagnosticFileIO.readIfPresent(root:root,parts:parts,limit:1_000_000);pins[parts.joined(separator:"/")]=data.map(WorkflowArchiveCodec.hash) ?? "absent"}
        return WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(pins))
    }
}

struct DisabledScheduleProposal {
    let rule:ScheduleRule
    let effects:[String]
    let components:Set<String>
    let previous:ScheduleRule?
    fileprivate let outputAccessHash:String
    static func preview(_ proposed:ScheduleRule,root:URL)throws->Self {
        guard !proposed.enabled,!proposed.allowEffects,proposed.outputAccessHash==nil,proposed.nextDue==nil else {throw WorkflowArchiveError("This action saves disabled schedules without output authorization only.")}
        let details=try ScheduleSavedSource.validate(proposed,root:root)
        guard !details.targets.isEmpty || proposed.target==nil else {throw WorkflowArchiveError("Data-only workflows have no control target.")}
        guard details.document.drafts.isEmpty else {throw WorkflowArchiveError("Save a canvas without unfinished drafts before scheduling.")}
        let previous=try ScheduledRunLedger.readOnlySnapshot(root:root)?.rules.first(where:{$0.id==proposed.id})
        guard previous?.projectID==nil || previous?.projectID==proposed.projectID,previous?.enabled != true,
              previous?.revision ?? 0 == proposed.revision else {throw WorkflowArchiveError("The disabled schedule changed. Preview its current settings again.")}
        var rule=proposed;rule.projectName=details.document.name;rule.hasEffects = !details.effects.isEmpty
        try rule.validate()
        return .init(rule:rule,effects:details.effects,components:MCPComponentAccess.identifiers(details.document),previous:previous,outputAccessHash:try ScheduleSavedSource.outputAccessHash(root:root,projectID:rule.projectID))
    }
    func revalidate(root:URL)throws {
        let current=try Self.preview(rule,root:root)
        guard current.previous==previous,current.outputAccessHash==outputAccessHash,current.components==components else {throw WorkflowArchiveError("Schedule source or destination authority changed. Preview again.")}
    }
}
