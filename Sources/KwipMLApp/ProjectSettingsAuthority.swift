import Foundation
import KwipMLCore

/// Local settings are not portable graph edits. Both native forms and automation
/// use this authority; permission grants remain the caller's responsibility.
enum ProjectSettingsGroup:String,CaseIterable {case execution,trainingCapture,derivativeCapture,historyCapture}
enum ProjectSettingsChange {
    case execution(ProjectExecutionOverrides)
    case trainingCapture(TrainingSampleCapturePolicy?)
    case derivativeCapture(DerivativeCapturePolicy?)
    case historyCapture(RunHistoryPolicy)
    case reset(ProjectSettingsGroup)
}
struct ProjectSettingsSnapshot {
    let projectID:String
    let lease:WorkflowEditorLease?
    let documentRevision:UInt64?
    fileprivate let allowsLegacy:Bool
    fileprivate let workspaceIdentity:URL
    let global:ExecutionPreferences
    let overrides:ProjectExecutionOverrides
    let history:RunHistoryPolicy
    let effective:ResolvedExecutionPreferences
    let globalRevision:String
    let projectRevision:String
    let historyRevision:String
    let revision:String
}
struct ProjectSettingsProposal {
    fileprivate let baseline:ProjectSettingsSnapshot
    fileprivate let overrides:ProjectExecutionOverrides?
    fileprivate let history:RunHistoryPolicy?
    var projectID:String {baseline.projectID}
    var baseRevision:String {baseline.revision}
    var proposedOverrides:ProjectExecutionOverrides {overrides ?? baseline.overrides}
    var proposedHistory:RunHistoryPolicy {history ?? baseline.history}
    var proposedEffective:ResolvedExecutionPreferences {get throws {try baseline.global.resolve(project:proposedOverrides)}}
}
struct ProjectSettingsFailure:LocalizedError {
    let code:String
    var errorDescription:String? {
        switch code {
        case "settings_stale":return "Settings changed. Revert or reload before applying these choices."
        case "settings_busy":return "Finish the active operation and pending edits before changing project settings."
        default:return "Reopen this project's settings before changing its choices."
        }
    }
}

extension WorkspaceStore {
    func projectSettingsSnapshot(projectID:String,allowLegacy:Bool=false)throws->ProjectSettingsSnapshot {
        guard activeProjectID==projectID else {throw ProjectSettingsFailure(code:"project_unavailable")}
        let host=workflowSession
        if let host {
            guard host.owner.projectID==projectID,host.owner.isCurrent(host.owner.lease()) else {throw ProjectSettingsFailure(code:"project_unavailable")}
        } else if !allowLegacy {throw ProjectSettingsFailure(code:"project_unavailable")}
        let global=try appPreferenceStore.load().execution
        let project=try projectExecutionPreferenceStore.load(projectID:projectID)
        let history=try RunHistoryPolicyStore(root:workspaceDirectory,projectID:projectID).load()
        let globalHash=try Self.settingsHash(global),projectHash=try Self.settingsHash(project),historyHash=try Self.settingsHash(history)
        return .init(projectID:projectID,lease:host?.owner.lease(),documentRevision:host?.owner.document.revision,allowsLegacy:allowLegacy,workspaceIdentity:workspaceDirectory.standardizedFileURL,
            global:global,overrides:project,history:history,effective:try global.resolve(project:project),
            globalRevision:globalHash,projectRevision:projectHash,historyRevision:historyHash,
            revision:WorkflowArchiveCodec.hash(Data((projectID+":"+globalHash+":"+projectHash+":"+historyHash).utf8)))
    }
    private static func settingsHash<T:Encodable>(_ value:T)throws->String {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        return WorkflowArchiveCodec.hash(try encoder.encode(value))
    }
    private func checkedSettings(_ baseline:ProjectSettingsSnapshot)throws->ProjectSettingsSnapshot {
        let current=try projectSettingsSnapshot(projectID:baseline.projectID,allowLegacy:baseline.allowsLegacy)
        guard current.workspaceIdentity==baseline.workspaceIdentity,current.lease==baseline.lease,current.documentRevision==baseline.documentRevision,current.revision==baseline.revision else {
            throw ProjectSettingsFailure(code:"settings_stale")
        }
        guard !isBusy,workflowSession?.hasActiveOperation != true,workflowSession?.owner.isBusy != true,
              workflowSession?.owner.hasPendingInspectorEdits != true,workflowSession?.owner.document.drafts.isEmpty != false else {
            throw ProjectSettingsFailure(code:"settings_busy")
        }
        return current
    }
    func previewProjectSettings(_ change:ProjectSettingsChange,baseline:ProjectSettingsSnapshot)throws->ProjectSettingsProposal {
        let current=try checkedSettings(baseline)
        var next:ProjectExecutionOverrides?,history:RunHistoryPolicy?
        switch change {
        case .execution(let value):
            next=try ProjectExecutionOverrides(maximumRunSeconds:value.maximumRunSeconds,cpuThreads:value.cpuThreads,memory:value.memory,
                sampleCapture:current.overrides.sampleCapture,derivativeCapture:current.overrides.derivativeCapture,workloadPolicy:value.workloadPolicy)
        case .trainingCapture(let value): next=try current.overrides.replacingSampleCapture(value)
        case .derivativeCapture(let value): next=try current.overrides.replacingDerivativeCapture(value)
        case .historyCapture(let value):
            guard current.lease != nil else {throw ProjectSettingsFailure(code:"project_unavailable")}
            history=value
        case .reset(let group):
            switch group {
            case .execution:next=try current.overrides.resettingResources()
            case .trainingCapture:next=try current.overrides.replacingSampleCapture(nil)
            case .derivativeCapture:next=try current.overrides.replacingDerivativeCapture(nil)
            case .historyCapture:
                guard current.lease != nil else {throw ProjectSettingsFailure(code:"project_unavailable")}
                history = .all
            }
        }
        // Validate the complete serialized value now, including capture invariants.
        if let next {_ = try JSONDecoder().decode(ProjectExecutionOverrides.self,from:JSONEncoder().encode(next))}
        return .init(baseline:current,overrides:next,history:history)
    }
    @discardableResult func applyProjectSettings(_ proposal:ProjectSettingsProposal)throws->ProjectSettingsSnapshot {
        _ = try checkedSettings(proposal.baseline)
        // No suspension between compare and atomic save; one proposal touches one store.
        if let next=proposal.overrides {
            try projectExecutionPreferenceStore.save(next,projectID:proposal.baseline.projectID)
            objectWillChange.send()
        } else if let history=proposal.history,let owner=workflowSession?.owner {
            try owner.setRunHistoryPolicy(history)
        } else {throw ProjectSettingsFailure(code:"project_unavailable")}
        appPreferencesError=nil
        return try projectSettingsSnapshot(projectID:proposal.baseline.projectID,allowLegacy:proposal.baseline.allowsLegacy)
    }
    @discardableResult func applyProjectSettings(_ change:ProjectSettingsChange,baseline:ProjectSettingsSnapshot?)->Bool {
        do {
            guard let baseline else {throw ProjectSettingsFailure(code:"settings_stale")}
            _ = try applyProjectSettings(previewProjectSettings(change,baseline:baseline));return true
        } catch {appPreferencesError=error.localizedDescription;return false}
    }
}
