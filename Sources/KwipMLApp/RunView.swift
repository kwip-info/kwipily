import SwiftUI
import Charts

struct RunView: View {
    @ObservedObject var store: WorkspaceStore
    private var chartMetrics: [EpochMetric] { MetricSampling.sample(store.metrics) }
    private var isChartSampled: Bool { store.metrics.count > 250 || (store.metrics.last?.epoch ?? 0) > store.metrics.count }
    var body: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Experiment").font(.system(size: 26, weight: .semibold))
                        Text(store.isRunning ? "Training your graph · real PyTorch computation" : store.lastCompleted ? "Completed · held-out evaluation" : store.wasCancelled ? "Cancelled · partial metrics only" : store.runError != nil ? "Failed · no completed result" : "Run a valid flow to begin").font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    if store.resultsStale { Label("Historical result", systemImage: "clock.arrow.circlepath").foregroundStyle(.orange).font(.caption) }
                    Button("Compare runs") { store.activeSection = "Experiments" }.buttonStyle(FlatButtonStyle())
                    if store.outputDirectory != nil { Button("Reveal artifacts") { store.revealArtifacts() } }
                }
                if store.isRunning { ProgressView(value: Double(store.metrics.last?.epoch ?? 0), total: Double(store.epochCount)) }
                HStack(spacing: 40) {
                    metric("Epoch", "\(store.metrics.last?.epoch ?? 0) / \(store.epochCount)")
                    metric("Validation loss", store.metrics.last.map { String(format: "%.4f", $0.validationLoss) } ?? "—")
                    metric("Held-out accuracy", store.metrics.last?.accuracy.map { String(format: "%.1f%%", $0 * 100) } ?? "—")
                }
                Divider()
                CheckpointView(store: store)
                Divider()
                GradientCapturePanel(capture: store.gradientCapture)
                Divider()
                SectionLabel(title: "Loss over training")
                if store.metrics.isEmpty {
                    ContentUnavailableView("No run yet", systemImage: "chart.xyaxis.line", description: Text("The chart will show measured training and validation loss."))
                        .frame(height: 200)
                } else {
                    Chart {
                        ForEach(chartMetrics) { point in
                            LineMark(x: .value("Epoch", point.epoch), y: .value("Loss", point.trainLoss)).foregroundStyle(by: .value("Series", "Training"))
                            LineMark(x: .value("Epoch", point.epoch), y: .value("Loss", point.validationLoss)).foregroundStyle(by: .value("Series", "Validation"))
                        }
                    }.chartForegroundStyleScale(["Training": MLTheme.muted, "Validation": MLTheme.rose]).capturedEpochAxis(EpochChartAxis(epochs:store.metrics.map(\.epoch))).chartXAxisLabel("Epoch").chartYAxisLabel("Loss score").frame(height: 200)
                    .accessibilityLabel("Measured training and validation loss")
                    if isChartSampled { Text(store.lastCompleted ? "Chart sampled · full metrics in export" : "Chart sampled · partial live observations").font(.system(size:10)).foregroundStyle(MLTheme.muted) }
                }
                if !store.predictions.isEmpty {
                    Divider()
                    SectionLabel(title: "Held-out predictions")
                    Chart(Array(store.predictions.enumerated()), id: \.offset) { _, point in
                        if let x = point["x"] as? Double, let y = point["y"] as? Double, let label = point["predicted"] as? Int {
                            PointMark(x: .value("Feature 1", x), y: .value("Feature 2", y)).foregroundStyle(label == 0 ? MLTheme.teal : MLTheme.rose).symbolSize(18)
                        }
                    }.frame(height: 180).accessibilityLabel("Held-out predictions scatter plot")
                    Text("Color shows predicted class. Accuracy above is measured against held-out labels.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                Divider()
                RunDiagnosticPanel(metrics: store.metrics, error: store.runError,
                    sampling: (store.metrics.last?.epoch ?? 0) > store.metrics.count ? .sampled : .unspecified)
                Divider()
                SectionLabel(title: "Run events")
                ForEach(Array(store.eventLog.enumerated()), id: \.offset) { _, message in Text(message).font(.system(size: 11, design: .monospaced)).foregroundStyle(MLTheme.muted) }
                if let error = store.runError { Text(error).font(.system(size: 12)).foregroundStyle(.orange).textSelection(.enabled) }
                Text("Each completed run saves an immutable graph, model.py, checkpoint.pt, metrics and a provenance manifest. " + (store.runtime?.kind == .bundled ? "Bundled local runtime · development-signed candidate." : "Development runtime · local checkout.")).font(.caption).foregroundStyle(MLTheme.muted)
            }.padding(24)
        }
    }
    func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 7) { SectionLabel(title: title); Text(value).font(.system(size: 23, weight: .medium, design: .monospaced)) }
    }
}
