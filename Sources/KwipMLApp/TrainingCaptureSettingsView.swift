import SwiftUI
import KwipMLCore

struct TrainingCaptureSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    var projectID: String? = nil
    @State private var policy = TrainingSampleCapturePolicy.factory
    @State private var baseline = TrainingSampleCapturePolicy.factory
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
            SectionLabel(title: "Training capture")
            Text("Choose what new training runs retain. Activation values remain sampled; held-out prediction capture includes the complete Validation table for connected data tools. Active runs keep their accepted choices.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            VStack(alignment: .leading, spacing: 10) {
                if projectID != nil {
                    FlatChoiceControl(title: "Capture inheritance", selection: inherit ? "inherit" : "override", options: [
                        .init(id: "inherit", title: "Capture · inherit app defaults"),
                        .init(id: "override", title: "Capture · project choices")]) { inherit = $0 == "inherit" }
                }
                choice("Activation sample values", enabled: policy.activationValues) {
                    policy=policy.changing(activationValues:$0)
                }
                choice("Tensor heatmaps",enabled:policy.tensorHeatmaps) {policy=policy.changing(tensorHeatmaps:$0)}
                if policy.tensorHeatmaps {
                    FlatChoiceControl(title:"Post-run heatmap playback",selection:String(policy.heatmapFrameLimit),options:[
                        .init(id:"0",title:"Playback · off"),.init(id:"6",title:"Playback · up to 6 frames"),
                        .init(id:"12",title:"Playback · up to 12 frames"),.init(id:"24",title:"Playback · up to 24 frames")]) {
                            policy=policy.changing(heatmapFrameLimit:Int($0) ?? 0)
                        }.disabled(inherit)
                    Text("Samples existing validation passes. Bounded summaries only; no extra training forwards or full tensor history. Actual epochs and coverage appear in the viewer.")
                        .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
                }
                choice("Held-out prediction rows", enabled: policy.predictionRows) {
                    policy=policy.changing(predictionRows:$0)
                }
                choice("Operation timing and memory", enabled: policy.performanceMeasurements) {
                    policy=policy.changing(performanceMeasurements:$0)
                }
                Text("Optional: one final validation pass, up to 8 rows and 128 operations. CPU host wall time or synchronized MPS host wall time; process resident-memory samples. This is not training throughput, GPU kernel time, peak memory or per-layer allocation. MPS synchronization adds observation overhead.")
                    .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Button("Apply capture choices", action: apply).disabled(!dirty)
                    Button("Revert") { load(); notice = nil }.disabled(!dirty && error == nil)
                    Spacer()
                    Button(projectID == nil ? "Reset capture defaults" : "Reset capture inheritance", action: reset)
                }.buttonStyle(FlatButtonStyle())
            }.disabled(!loaded || !current)
            if !loaded && current { Button("Reload capture settings", action: load).buttonStyle(FlatButtonStyle()) }
            if !current { Text("Reopen the active project’s settings to change capture choices.").font(.caption).foregroundStyle(MLTheme.rose) }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
            if let notice { Text(notice).font(.caption).foregroundStyle(MLTheme.teal) }
            if let effective = try? store.executionPolicy(for: projectID) {
                Text("Saved: activation values \(effective.sampleCapture.activationValues ? "captured" : "omitted") · heatmaps \(effective.sampleCapture.tensorHeatmaps ? "captured" : "omitted") · playback up to \(effective.sampleCapture.heatmapFrameLimit) frames · prediction rows \(effective.sampleCapture.predictionRows ? "captured" : "omitted") · timing/memory \(effective.sampleCapture.performanceMeasurements ? "captured" : "omitted") · \(effective.sampleCaptureSource == .project ? "project choices" : "app defaults")")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            Text("Metrics, tensor shapes and types remain available. Replay still retains prepared data, fitted state and checkpoints. Explicit source inspection and inference previews may show data; Run from retains activation caches. These controls do not erase earlier results or redact arbitrary error logs. Gradient summaries are configured in Train’s Run settings.")
                .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
            DerivativeCaptureSettingsView(store: store, projectID: projectID)
        }.frame(maxWidth: 600, alignment: .leading)
        .onAppear(perform: load)
        .onChange(of: projectID) { _, _ in load() }
        .onChange(of: store.appPreferences.execution) { _, _ in if !dirty { load() } }
    }
    private func choice(_ title: String, enabled: Bool, update: @escaping (Bool) -> Void) -> some View {
        FlatChoiceControl(title: title, selection: enabled ? "capture" : "omit", options: [
            .init(id: "capture", title: "\(title) · capture"),
            .init(id: "omit", title: "\(title) · omit")]) { update($0 == "capture") }
            .disabled(inherit)
    }
    private func load() {
        guard current else { loaded = false; return }
        do {
            applicationBaseline = projectID == nil ? try store.applicationSettingsSnapshot() : nil
            settingsBaseline = try projectID.map { try store.projectSettingsSnapshot(projectID: $0, allowLegacy: true) }
            let global = try store.appPreferenceStore.load().execution
            let project = try projectID.map { try store.projectExecutionPreferenceStore.load(projectID: $0) }
            inherit = projectID != nil && project?.sampleCapture == nil
            policy = project?.sampleCapture ?? global.sampleCapture
            baseline = policy; baselineInherit = inherit; loaded = true; error = nil
        } catch { loaded = false; self.error = error.localizedDescription }
    }
    private func apply() {
        guard current, loaded else { return }
        let saved: Bool
        if projectID != nil { saved = store.applyProjectSettings(.trainingCapture(inherit ? nil : policy), baseline: settingsBaseline) }
        else { saved = store.applyApplicationSettings(.trainingCapture(policy),baseline:applicationBaseline) }
        if saved { load(); notice = "Saved for future training runs." }
        else { error = store.appPreferencesError }
    }
    private func reset() {
        guard current, loaded else { return }
        let saved: Bool
        if projectID != nil { saved = store.applyProjectSettings(.reset(.trainingCapture), baseline: settingsBaseline) }
        else { saved = store.applyApplicationSettings(.reset(.trainingCapture),baseline:applicationBaseline) }
        if saved { load(); notice = "Capture reset. Resource limits are unchanged." }
        else { error = store.appPreferencesError }
    }
}
