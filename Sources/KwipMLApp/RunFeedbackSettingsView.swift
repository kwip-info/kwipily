import SwiftUI
import KwipMLCore

struct RunFeedbackSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject private var feedback = RunFeedbackController.shared
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionLabel(title: "Run feedback")
            Text(feedback.summary).font(.caption).foregroundStyle(MLTheme.muted)
            Toggle("Show run status in the macOS menu bar", isOn: field(\.showStatusIcon))
            Toggle("Notify when a run completes", isOn: field(\.notifyCompleted, requestsPermission: true))
            Toggle("Notify when a run fails", isOn: field(\.notifyFailed, requestsPermission: true))
            Text("Applies to new training runs, experiment groups and backend fits. Cancelled runs never send a completion notification. Notifications contain no project names, file paths or data.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            Text(feedback.deliveryMessage).font(.caption).foregroundStyle(MLTheme.muted)
            Button("Reset run feedback") { _ = store.setRunFeedback(.factory) }
        }.onAppear { feedback.refreshAuthorization() }
    }
    private func field(_ path: WritableKeyPath<RunFeedbackPreferences, Bool>, requestsPermission: Bool = false) -> Binding<Bool> {
        runFeedbackBinding(store: store, path: path, requestsPermission: requestsPermission)
    }

}

extension WorkspaceStore {
    @discardableResult func setRunFeedback(_ preferences: RunFeedbackPreferences) -> Bool {
        changeApplicationSettings(.feedback(preferences))
    }
}

/// Run-menu controls and Settings share the same persistence and explicit permission boundary.
struct RunFeedbackMenu: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject private var feedback = RunFeedbackController.shared
    var body: some View {
        Menu("Run feedback") {
            Text(feedback.summary)
            Divider()
            Toggle("Show menu-bar status", isOn: runFeedbackBinding(store: store, path: \.showStatusIcon))
            Toggle("Notify on completion", isOn: runFeedbackBinding(store: store, path: \.notifyCompleted, requestsPermission: true))
            Toggle("Notify on failure", isOn: runFeedbackBinding(store: store, path: \.notifyFailed, requestsPermission: true))
        }
    }
}

func runFeedbackBinding(store: WorkspaceStore, path: WritableKeyPath<RunFeedbackPreferences, Bool>, requestsPermission: Bool = false) -> Binding<Bool> {
    Binding(get: { store.appPreferences.effectiveRunFeedback[keyPath: path] }, set: { value in
        do {
            var next = try store.appPreferenceStore.load().effectiveRunFeedback
            next[keyPath: path] = value
            if store.setRunFeedback(next), requestsPermission, value { RunFeedbackController.shared.requestAuthorization() }
        } catch {
            store.appPreferencesError = "Run feedback settings were not saved. " + error.localizedDescription
        }
    })
}
