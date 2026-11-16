import Foundation
import KwipMLCore

extension WorkspaceStore {
    /// Patch only this permission in the latest saved policy; settings drafts may be older.
    @discardableResult func setDerivativeCapturePreferences(_ policy: DerivativeCapturePolicy) -> Bool {
        do {
            var next = try appPreferenceStore.load()
            next.execution = try next.execution.replacingDerivativeCapture(policy)
            try appPreferenceStore.save(next)
            appPreferences = next; appPreferencesError = nil
            return true
        } catch {
            appPreferencesError = "Derivative capture default was not saved. \(error.localizedDescription)"
            return false
        }
    }

    @discardableResult func resetDerivativeCapturePreferences() -> Bool {
        setDerivativeCapturePreferences(.factory)
    }

    /// Nil inherits; explicit false remains a project override. A stale settings window cannot edit another project.
    @discardableResult func setProjectDerivativeCapture(_ policy: DerivativeCapturePolicy?, projectID: String) -> Bool {
        guard activeProjectID == projectID else {
            appPreferencesError = "Reopen the active project's settings before changing derivative capture."
            return false
        }
        do {
            let current = try projectExecutionPreferenceStore.load(projectID: projectID)
            try projectExecutionPreferenceStore.save(current.replacingDerivativeCapture(policy), projectID: projectID)
            objectWillChange.send(); appPreferencesError = nil
            return true
        } catch {
            appPreferencesError = "Project derivative capture choice was not saved. \(error.localizedDescription)"
            return false
        }
    }

    @discardableResult func resetProjectDerivativeCapture(projectID: String) -> Bool {
        setProjectDerivativeCapture(nil, projectID: projectID)
    }
}
