import SwiftUI
import AppKit
struct ExperimentsView: View {
    @ObservedObject var store: WorkspaceStore
    private var a: ExperimentRecord? { store.experiments.first { $0.id == store.comparisonA } }
    private var b: ExperimentRecord? { store.experiments.first { $0.id == store.comparisonB } }
    var body: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Experiments").font(.system(size: 28, weight: .semibold))
                Text("Compare immutable runs. Reuse a recipe, change one variable, and run again.").foregroundStyle(MLTheme.muted)
                if store.experiments.isEmpty {
                    ContentUnavailableView("No recorded experiments yet", systemImage: "chart.xyaxis.line", description: Text("Run this project to record its settings, times and outcome. Earlier POC artifacts remain available through Code."))
                } else {
                    VStack(alignment: .leading, spacing: 10) {
                        FlatChoiceControl(title: "Experiment A", selection: store.comparisonA ?? "", options: options) { store.comparisonA = $0 }
                        FlatChoiceControl(title: "Experiment B", selection: store.comparisonB ?? "", options: options) { store.comparisonB = $0 }
                    }
                    if let a, let b {
                        if a.id == b.id { Text("Choose two different experiments.").foregroundStyle(.orange) }
                        else {
                            Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 10) {
                                row("", "A", "B")
                                row("Architecture", "Reference A", sameArchitecture(a,b) ? "Same as A" : "Different from A")
                                row("Source identity", a.sourceHash.map { String($0.prefix(12)) } ?? "Unavailable", b.sourceHash.map { String($0.prefix(12)) } ?? "Unavailable")
                                row("Outcome", a.status, b.status)
                                row("Finished", a.finishedAt.formatted(), b.finishedAt.formatted())
                                row("Elapsed seconds", String(format:"%.2f",a.finishedAt.timeIntervalSince(a.startedAt)), String(format:"%.2f",b.finishedAt.timeIntervalSince(b.startedAt)))
                                row("Epochs completed", "\(a.epoch)", "\(b.epoch)")
                                row("Batch size", "\(a.settings.batchSize)", "\(b.settings.batchSize)")
                                row("Sample limit", "\(a.settings.sampleLimit)", "\(b.settings.sampleLimit)")
                                row("Model seed", "\(a.settings.seed)", "\(b.settings.seed)")
                                row("Split seed", "\(a.settings.splitSeed)", "\(b.settings.splitSeed)")
                                row("Validation %", "\(a.settings.validationPercent)", "\(b.settings.validationPercent)")
                                row("Learning rate", "\(a.settings.learningRate)", "\(b.settings.learningRate)")
                                row("Train / validation rows", counts(a), counts(b))
                                row("Validation loss", metric(a.validationLoss), metric(b.validationLoss))
                                row("Accuracy", metric(a.accuracy), metric(b.accuracy))
                                row("Device", a.device.uppercased(), b.device.uppercased())
                                row("Split identity", a.splitHash.map { String($0.prefix(12)) } ?? "Unavailable", b.splitHash.map { String($0.prefix(12)) } ?? "Unavailable")
                            }.font(.system(size: 12)).padding(14).background(MLTheme.panel)
                            Text(comparisonNote(a,b)).font(.system(size:12)).foregroundStyle(MLTheme.muted)
                        }
                    }
                    Divider(); SectionLabel(title: "History · most recent first")
                    ForEach(store.experiments) { run in
                        VStack(alignment: .leading, spacing: 8) {
                            Text(run.label).font(.system(size:14,weight:.medium))
                            Text("Model seed \(run.settings.seed) · split seed \(run.settings.splitSeed) · batch \(run.settings.batchSize) · validation \(run.settings.validationPercent)%").font(.caption).foregroundStyle(MLTheme.muted)
                            if let error = run.error { Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
                            if let events = run.events, !events.isEmpty {
                                DisclosureGroup("Recorded events") { ForEach(Array(events.enumerated()), id: \.offset) { _, event in Text(event).font(.system(size: 11, design: .monospaced)).textSelection(.enabled) } }
                            }
                            RecordedDiagnosticView(store: store, record: run)
                            HStack {
                                Button("Use settings") { store.useExperimentSettings(run) }.disabled(store.isBusy)
                                Button("Artifacts") { NSWorkspace.shared.open(URL(fileURLWithPath: run.directory)) }
                                Text(store.semanticSignature(run.graph) == store.semanticSignature(store.graph) ? "Matches current configuration" : "Earlier configuration").font(.caption).foregroundStyle(MLTheme.muted)
                            }.buttonStyle(FlatButtonStyle())
                            Divider()
                        }
                    }
                }
            }.padding(24)
        }
    }
    private func sameArchitecture(_ a: ExperimentRecord, _ b: ExperimentRecord) -> Bool {
        var first = a.graph, second = b.graph; first.execution = nil; second.execution = nil
        return store.semanticSignature(first) == store.semanticSignature(second)
    }
    private var options: [FlatChoiceOption] { store.experiments.map { FlatChoiceOption(id: $0.id, title: $0.label) } }
    private func row(_ label:String,_ first:String,_ second:String)->some View { GridRow { Text(label).foregroundStyle(MLTheme.muted);Text(first).textSelection(.enabled);Text(second).textSelection(.enabled) } }
    private func metric(_ value:Double?)->String { value.map { String(format:"%.5f",$0) } ?? "—" }
    private func counts(_ value:ExperimentRecord)->String { "\(value.trainingSamples.map(String.init) ?? "—") / \(value.validationSamples.map(String.init) ?? "—")" }
    private func comparisonNote(_ a:ExperimentRecord,_ b:ExperimentRecord)->String {
        if a.status != "completed" || b.status != "completed" { return "One or both runs are incomplete. Partial measurements are shown for diagnosis, not as final model scores." }
        if a.splitHash == nil || a.splitHash != b.splitHash { return "These runs use different or unrecorded held-out rows. Score differences measure split sensitivity as well as model performance; repeat multiple seeds before drawing conclusions." }
        return "These runs share the recorded split identity. Compare the full configuration above; matching rows alone do not establish statistical significance."
    }
}
