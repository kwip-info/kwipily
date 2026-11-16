import SwiftUI
import KwipMLCore

/// A separate permission and inheritance group; other sample captures do not grant it.
struct DerivativeCaptureSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    var projectID: String? = nil
    @State private var policy = DerivativeCapturePolicy.factory
    @State private var baseline = DerivativeCapturePolicy.factory
    @State private var inherit = false
    @State private var baselineInherit = false
    @State private var settingsBaseline: ProjectSettingsSnapshot?
    @State private var applicationBaseline: ApplicationSettingsSnapshot?
    @State private var loaded = false
    @State private var error: String?
    @State private var notice: String?
    private var current: Bool { projectID == nil || projectID == store.activeProjectID }
    private var dirty: Bool { inherit != baselineInherit || (!inherit && policy != baseline) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Selected derivative values")
            Text("A derivative measures how changing an input or learned setting changes a selected result. Enable this separate permission before requesting those values in a new run.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            VStack(alignment: .leading, spacing: 10) {
                if projectID != nil {
                    FlatChoiceControl(title: "Derivative capture inheritance", selection: inherit ? "inherit" : "override", options: [
                        .init(id: "inherit", title: "Derivatives · inherit app default"),
                        .init(id: "override", title: "Derivatives · project choice")]) { inherit = $0 == "inherit" }
                }
                FlatChoiceControl(title: "Selected derivative values", selection: policy.values ? "capture" : "omit", options: [
                    .init(id: "capture", title: "Derivative values · allow capture"),
                    .init(id: "omit", title: "Derivative values · do not capture")]) { policy = .init(values: $0 == "capture") }
                    .disabled(inherit)
                HStack {
                    Button("Apply derivative choice", action: apply).disabled(!dirty)
                    Button("Revert") { load(); notice = nil }.disabled(!dirty && error == nil)
                    Spacer()
                    Button(projectID == nil ? "Reset derivative default" : "Reset derivative inheritance", action: reset)
                }.buttonStyle(FlatButtonStyle())
            }.disabled(!loaded || !current)
            if !loaded && current { Button("Reload derivative settings", action: load).buttonStyle(FlatButtonStyle()) }
            if !current { Text("Reopen the active project’s settings to change derivative capture.").font(.caption).foregroundStyle(MLTheme.rose) }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
            if let notice { Text(notice).font(.caption).foregroundStyle(MLTheme.teal) }
            if let effective = try? store.executionPolicy(for: projectID) {
                Text("Saved: selected derivative values \(effective.derivativeCapture.values ? "allowed" : "not captured") · \(effective.derivativeCaptureSource == .project ? "project choice" : "app default")")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            Text("This permission does not change activation values, prediction rows or gradient summaries. It does not request derivatives by itself or remove earlier saved values. Active runs keep their accepted choice.")
                .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
        }.padding(12).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
        .frame(maxWidth: 600, alignment: .leading)
        .onAppear(perform: load)
        .onChange(of: projectID) { _, _ in load() }
        .onChange(of: store.appPreferences.execution) { _, _ in if !dirty { load() } }
    }
    private func load() {
        guard current else { loaded = false; return }
        do {
            applicationBaseline = projectID == nil ? try store.applicationSettingsSnapshot() : nil
            settingsBaseline = try projectID.map { try store.projectSettingsSnapshot(projectID: $0, allowLegacy: true) }
            let global = try store.appPreferenceStore.load().execution
            let project = try projectID.map { try store.projectExecutionPreferenceStore.load(projectID: $0) }
            inherit = projectID != nil && project?.derivativeCapture == nil
            policy = project?.derivativeCapture ?? global.derivativeCapture
            baseline = policy; baselineInherit = inherit; loaded = true; error = nil
        } catch { loaded = false; self.error = error.localizedDescription }
    }
    private func apply() {
        guard current, loaded else { return }
        let saved: Bool
        if projectID != nil { saved = store.applyProjectSettings(.derivativeCapture(inherit ? nil : policy), baseline: settingsBaseline) }
        else { saved = store.applyApplicationSettings(.derivativeCapture(policy),baseline:applicationBaseline) }
        if saved { load(); notice = "Saved for future runs. Other capture choices are unchanged." }
        else { error = store.appPreferencesError }
    }
    private func reset() {
        guard current, loaded else { return }
        let saved: Bool
        if projectID != nil { saved = store.applyProjectSettings(.reset(.derivativeCapture), baseline: settingsBaseline) }
        else { saved = store.applyApplicationSettings(.reset(.derivativeCapture),baseline:applicationBaseline) }
        if saved { load(); notice = "Derivative capture reset. Other capture choices and resource limits are unchanged." }
        else { error = store.appPreferencesError }
    }
}
