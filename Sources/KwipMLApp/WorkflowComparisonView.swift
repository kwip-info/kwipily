import SwiftUI
import Charts
import KwipMLCore

struct WorkflowComparisonView: View {
    let baseline: VerifiedWorkflowOutcome
    let candidate: VerifiedWorkflowOutcome
    var body: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Run comparison").font(.headline)
                Text("A is the pinned baseline. B is the selected verified run. These are captured snapshots; current sources and runtime have not been revalidated.").font(.caption).foregroundStyle(MLTheme.muted)
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
                    row("", "A · Baseline", "B · Selected")
                    row("Run", baseline.outcome.runID, candidate.outcome.runID)
                    row("Finished", baseline.outcome.finishedAt, candidate.outcome.finishedAt)
                    row("Epoch", epoch(baseline), epoch(candidate))
                    row("Training / held-out rows", counts(baseline), counts(candidate))
                    row("Training loss", number(baseline.modelContent?.metrics.last?.trainLoss), number(candidate.modelContent?.metrics.last?.trainLoss))
                    row("Validation loss", number(baseline.modelContent?.metrics.last?.validationLoss), number(candidate.modelContent?.metrics.last?.validationLoss))
                    row("Accuracy", number(baseline.modelContent?.metrics.last?.accuracy), number(candidate.modelContent?.metrics.last?.accuracy))
                }.font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                Text(WorkflowPartitionEvidence.comparison(baseline.modelContent?.partitionEvidence, candidate.modelContent?.partitionEvidence).message)
                    .font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                Chart {
                    ForEach(MetricSampling.sample(baseline.modelContent?.metrics ?? [])) { point in
                        LineMark(x: .value("Epoch", point.epoch), y: .value("Validation loss", point.validationLoss))
                            .foregroundStyle(by: .value("Run", "A · Baseline"))
                    }
                    ForEach(MetricSampling.sample(candidate.modelContent?.metrics ?? [])) { point in
                        LineMark(x: .value("Epoch", point.epoch), y: .value("Validation loss", point.validationLoss))
                            .foregroundStyle(by: .value("Run", "B · Selected"))
                    }
                }.chartForegroundStyleScale(["A · Baseline": MLTheme.muted, "B · Selected": MLTheme.rose])
                    .capturedEpochAxis(EpochChartAxis(epochs:((baseline.modelContent?.metrics ?? [])+(candidate.modelContent?.metrics ?? [])).map(\.epoch)))
                    .chartXAxisLabel("Epoch").chartYAxisLabel("Validation loss score").frame(height: 180)
                    .accessibilityLabel("Verified baseline and selected run validation loss")
                Text("Chart may be sampled from loaded observations. When measured for a classification task, accuracy is a fraction from 0 to 1. Scores at different epochs or on different held-out rows are not a controlled model comparison. Matching settings alone do not establish identical rows or statistical significance.").font(.caption).foregroundStyle(MLTheme.muted)
                configuration
            }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    @ViewBuilder private var configuration: some View {
        switch Result(catching: { try WorkflowComparisonFields.differences(WorkflowComparisonFields.configuration(baseline), WorkflowComparisonFields.configuration(candidate)) }) {
        case .success(let fields):
            Text("Configuration changes · \(fields.count)").font(.headline)
            Text("Paths identify exact captured fields, including data-component seeds and row budgets. Model identity excludes labels and layout; a changed identity requires inspecting Saved model source. Source identities describe captured bytes, not live files.").font(.caption).foregroundStyle(MLTheme.muted)
            if fields.isEmpty { Text("No differences in the compared configuration fields.").font(.caption) }
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(fields) { field in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(field.path).foregroundStyle(MLTheme.rose)
                        Text("A  " + field.baseline)
                        Text("B  " + field.candidate)
                    }.font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                    Divider()
                }
            }
        case .failure(let error):
            Text("Configuration comparison unavailable: \(error.localizedDescription)").foregroundStyle(MLTheme.rose)
        }
    }
    private func counts(_ value: VerifiedWorkflowOutcome) -> String {
        guard let evidence = value.modelContent?.partitionEvidence else { return "Not recorded" }
        return "\(evidence.trainingCount) / \(evidence.validationCount)"
    }
    private func epoch(_ value: VerifiedWorkflowOutcome) -> String { value.modelContent?.metrics.last.map { String($0.epoch) } ?? "Not recorded" }
    private func number(_ value: Double?) -> String { value.map { String(format: "%.6g", $0) } ?? "Not recorded" }
    private func row(_ title: String, _ a: String, _ b: String) -> some View {
        GridRow { Text(title).foregroundStyle(MLTheme.muted); Text(a); Text(b) }
    }
}
