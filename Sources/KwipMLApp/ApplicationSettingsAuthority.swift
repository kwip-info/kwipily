import Foundation
import KwipMLCore

/// Ordinary application preferences, not permission, credential or cleanup actions.
enum ApplicationSettingsGroup:String,CaseIterable {
    case execution,storage,trainingCapture,derivativeCapture,appearance,canvas,feedback
    var effectTiming:String {
        switch self {
        case .execution,.trainingCapture,.derivativeCapture:
            return "Captured at each operation admission. Already accepted operations keep their policy; later automatic consumers and queued operations may use new defaults. Required model, replay and connected outputs remain governed by their contracts."
        case .storage:return "Used by future storage/cache admissions. Existing files are not deleted and accepted reservations are not rewritten."
        case .appearance:return "Applies to open and future app windows. macOS accessibility preferences can override panel transparency."
        case .canvas:return "Zoom sensitivity applies to open and future canvases; restoration preference applies at next launch. Existing layout and authored positions are unchanged."
        case .feedback:return "Applies immediately, including future outcomes of active runs. macOS notification permission and actual delivery remain separate and unconfirmed."
        }
    }
}
struct ApplicationCanvasPreferences:Equatable {
    let zoomSensitivity:Double
    let restoreAtLaunch:Bool
}
struct ApplicationCanvasPreferencesDraft {var zoomSensitivity:Double;var restoreAtLaunch:Bool}
enum ApplicationSettingsChange {
    case execution(ExecutionPreferences),storage(ExecutionPreferences)
    case trainingCapture(TrainingSampleCapturePolicy),derivativeCapture(DerivativeCapturePolicy)
    case appearance(AppearancePreferences),canvas(ApplicationCanvasPreferences),feedback(RunFeedbackPreferences)
    case reset(ApplicationSettingsGroup)
    var group:ApplicationSettingsGroup {
        switch self {case .execution:return .execution;case .storage:return .storage;case .trainingCapture:return .trainingCapture
        case .derivativeCapture:return .derivativeCapture;case .appearance:return .appearance;case .canvas:return .canvas
        case .feedback:return .feedback;case .reset(let group):return group}
    }
}
struct ApplicationSettingsSnapshot {
    /// Internal typed authority only. MCP must project an explicit safe group schema.
    let preferences:AppPreferences
    let revision:String
    let feedbackDeliveryStatus:String
    fileprivate let identity:URL
}
struct ApplicationSettingsProposal {
    fileprivate let baseline:ApplicationSettingsSnapshot
    let proposedPreferences:AppPreferences
    let group:ApplicationSettingsGroup
    var baseRevision:String {baseline.revision}
}

extension WorkspaceStore {
    func applicationSettingsSnapshot()throws->ApplicationSettingsSnapshot {
        let value=try appPreferenceStore.load()
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        return .init(preferences:value,revision:WorkflowArchiveCodec.hash(try encoder.encode(value)),
            feedbackDeliveryStatus:RunFeedbackController.shared.deliveryMessage,identity:appPreferenceStore.fileURL.standardizedFileURL)
    }
    private func checkApplicationSettings(_ baseline:ApplicationSettingsSnapshot)throws->ApplicationSettingsSnapshot {
        let current=try applicationSettingsSnapshot()
        guard current.identity==baseline.identity,current.revision==baseline.revision else {throw ProjectSettingsFailure(code:"settings_stale")}
        return current
    }
    func previewApplicationSettings(_ change:ApplicationSettingsChange,baseline:ApplicationSettingsSnapshot)throws->ApplicationSettingsProposal {
        let current=try checkApplicationSettings(baseline)
        var next=current.preferences
        switch change {
        case .execution(let value):
            next.execution=try .init(maximumRunSeconds:value.maximumRunSeconds,cpuThreads:value.cpuThreads,maximumResidentBytes:value.maximumResidentBytes,
                managedStorageLimitBytes:next.execution.managedStorageLimitBytes,sampleCapture:next.execution.sampleCapture,
                preparationCacheLimitBytes:next.execution.preparationCacheLimitBytes,derivativeCapture:next.execution.derivativeCapture,workloadPolicy:value.workloadPolicy)
        case .storage(let value):
            next.execution=try .init(maximumRunSeconds:next.execution.maximumRunSeconds,cpuThreads:next.execution.cpuThreads,maximumResidentBytes:next.execution.maximumResidentBytes,
                managedStorageLimitBytes:value.managedStorageLimitBytes,sampleCapture:next.execution.sampleCapture,
                preparationCacheLimitBytes:value.preparationCacheLimitBytes,derivativeCapture:next.execution.derivativeCapture,workloadPolicy:next.execution.workloadPolicy)
        case .trainingCapture(let value):next.execution=try next.execution.replacingSampleCapture(value)
        case .derivativeCapture(let value):next.execution=try next.execution.replacingDerivativeCapture(value)
        case .appearance(let value):next.appearance=value
        case .canvas(let value):next.canvasZoomSensitivity=value.zoomSensitivity;next.restoreCanvasSessionAtLaunch=value.restoreAtLaunch
        case .feedback(let value):next.runFeedback=value
        case .reset(let group):
            switch group {
            case .execution:return try previewApplicationSettings(.execution(.factory),baseline:current)
            case .storage:return try previewApplicationSettings(.storage(.factory),baseline:current)
            case .trainingCapture:return try previewApplicationSettings(.trainingCapture(.factory),baseline:current)
            case .derivativeCapture:return try previewApplicationSettings(.derivativeCapture(.factory),baseline:current)
            case .appearance:next.appearance = .factory
            case .canvas:next.canvasZoomSensitivity=nil;next.restoreCanvasSessionAtLaunch=nil
            case .feedback:next.runFeedback=nil
            }
        }
        _ = try AppPreferencesStore.validated(next)
        return .init(baseline:current,proposedPreferences:next,group:change.group)
    }
    /// Future admissions use newly saved execution/capture defaults. Already
    /// accepted operations keep their immutable policy; later automatic consumers
    /// are separate admissions. Presentation and feedback take effect immediately.
    @discardableResult func applyApplicationSettings(_ proposal:ApplicationSettingsProposal)throws->ApplicationSettingsSnapshot {
        _ = try checkApplicationSettings(proposal.baseline)
        try appPreferenceStore.save(proposal.proposedPreferences)
        appPreferences=proposal.proposedPreferences;appPreferencesError=nil
        if proposal.group == .feedback {RunFeedbackController.shared.configure(proposal.proposedPreferences.effectiveRunFeedback)}
        return try applicationSettingsSnapshot()
    }
    @discardableResult func applyApplicationSettings(_ change:ApplicationSettingsChange,baseline:ApplicationSettingsSnapshot?)->Bool {
        do {
            guard let baseline else {throw ProjectSettingsFailure(code:"settings_stale")}
            _ = try applyApplicationSettings(previewApplicationSettings(change,baseline:baseline));return true
        } catch {appPreferencesError=error.localizedDescription;return false}
    }
    @discardableResult func changeAppearance(_ update:(inout AppearancePreferences)->Void)->Bool {
        do {
            let baseline=try applicationSettingsSnapshot();var next=baseline.preferences.appearance;update(&next)
            return applyApplicationSettings(.appearance(next),baseline:baseline)
        } catch {appPreferencesError=error.localizedDescription;return false}
    }
    @discardableResult func changeCanvasPreferences(_ update:(inout ApplicationCanvasPreferencesDraft)->Void)->Bool {
        do {
            let baseline=try applicationSettingsSnapshot()
            var next=ApplicationCanvasPreferencesDraft(zoomSensitivity:baseline.preferences.effectiveCanvasZoomSensitivity,restoreAtLaunch:baseline.preferences.restoreCanvasSessionAtLaunch != false)
            update(&next)
            return applyApplicationSettings(.canvas(.init(zoomSensitivity:next.zoomSensitivity,restoreAtLaunch:next.restoreAtLaunch)),baseline:baseline)
        } catch {appPreferencesError=error.localizedDescription;return false}
    }
    /// Immediate native controls patch a fresh typed group; deferred forms pass
    /// their saved baseline to applyApplicationSettings instead.
    @discardableResult func changeApplicationSettings(_ change:ApplicationSettingsChange)->Bool {
        do {return applyApplicationSettings(change,baseline:try applicationSettingsSnapshot())}
        catch {appPreferencesError=error.localizedDescription;return false}
    }
}
