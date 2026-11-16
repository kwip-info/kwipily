import Foundation
import KwipMLCore

extension WorkspaceStore {
    var projectExecutionPreferenceStore: ProjectExecutionPreferencesStore {
        .init(workspace: workspaceDirectory)
    }
    /// Capture once at action acceptance. Nil explicitly means application defaults.
    /// Corrupt persisted limits refuse the action rather than quietly removing limits.
    func executionPolicy(for projectID: String?, capabilityMaximumSeconds: Int? = nil) throws -> ResolvedExecutionPreferences {
        let global = try appPreferenceStore.load().execution
        let project = try projectID.map { try projectExecutionPreferenceStore.load(projectID: $0) }
        return try global.resolve(project: project, capabilityMaximumSeconds: capabilityMaximumSeconds)
    }
    @discardableResult func setExecutionPreferences(_ preferences: ExecutionPreferences) -> Bool {
        do {
            var next = try appPreferenceStore.load()
            next.execution = try preferences.replacingSampleCapture(next.execution.sampleCapture).replacingDerivativeCapture(next.execution.derivativeCapture)
            try appPreferenceStore.save(next)
            appPreferences = next; appPreferencesError = nil
            return true
        } catch { appPreferencesError = "Execution defaults were not saved. \(error.localizedDescription)"; return false }
    }
    @discardableResult func resetExecutionPreferences() -> Bool {
        do {
            appPreferences = try appPreferenceStore.resetExecution()
            appPreferencesError = nil; return true
        } catch { appPreferencesError = "Execution defaults were not reset. \(error.localizedDescription)"; return false }
    }
    @discardableResult func setProjectExecutionOverrides(_ overrides: ProjectExecutionOverrides, projectID: String) -> Bool {
        do {
            let current = try projectExecutionPreferenceStore.load(projectID: projectID)
            try projectExecutionPreferenceStore.save(overrides.replacingSampleCapture(current.sampleCapture).replacingDerivativeCapture(current.derivativeCapture), projectID: projectID)
            objectWillChange.send(); appPreferencesError = nil
            return true
        } catch { appPreferencesError = "Project execution settings were not saved. \(error.localizedDescription)"; return false }
    }
    @discardableResult func resetProjectExecutionOverrides(projectID: String) -> Bool {
        do {
            _ = try projectExecutionPreferenceStore.reset(projectID: projectID)
            objectWillChange.send(); appPreferencesError = nil
            return true
        } catch { appPreferencesError = "Project execution settings were not reset. \(error.localizedDescription)"; return false }
    }
    /// Capture settings patch only their own preference group. Resource form drafts may be older.
    @discardableResult func setSampleCapturePreferences(_ policy: TrainingSampleCapturePolicy) -> Bool {
        do {
            var next = try appPreferenceStore.load()
            next.execution = try next.execution.replacingSampleCapture(policy)
            try appPreferenceStore.save(next)
            appPreferences = next; appPreferencesError = nil
            return true
        } catch { appPreferencesError = "Sample capture defaults were not saved. \(error.localizedDescription)"; return false }
    }
    @discardableResult func resetSampleCapturePreferences() -> Bool {
        setSampleCapturePreferences(.factory)
    }
    @discardableResult func setProjectSampleCapture(_ policy: TrainingSampleCapturePolicy?, projectID: String) -> Bool {
        do {
            let current = try projectExecutionPreferenceStore.load(projectID: projectID)
            try projectExecutionPreferenceStore.save(current.replacingSampleCapture(policy), projectID: projectID)
            objectWillChange.send(); appPreferencesError = nil
            return true
        } catch { appPreferencesError = "Project sample capture settings were not saved. \(error.localizedDescription)"; return false }
    }
    @discardableResult func resetProjectSampleCapture(projectID: String) -> Bool {
        setProjectSampleCapture(nil, projectID: projectID)
    }

}
