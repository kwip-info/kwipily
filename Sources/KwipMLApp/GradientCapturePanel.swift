import SwiftUI
import KwipMLCore

struct GradientCapturePanel: View {
    @ObservedObject var capture: GradientCaptureStore
    var body: some View {
        GradientObservationView(message: capture.message, runID: capture.runID,
                                inventory: capture.inventory, summaries: capture.summaries)
    }
}

/// Displays bounded, already validated observations without owning execution or
/// reading artifacts. Callers remain responsible for verification and provenance.
struct GradientObservationView: View {
    let message: String
    let runID: String?
    let inventory: GradientCaptureInventory?
    let summaries: [GradientEpochSummary]
    @State private var selectedEpoch: String = ""
    init(message: String, runID: String?, inventory: GradientCaptureInventory?, summaries: [GradientEpochSummary]) {
        self.message = message
        self.runID = runID
        self.inventory = inventory
        self.summaries = summaries
    }
    private var captured: [GradientEpochSummary] { summaries.filter { $0.status == .captured } }
    private var selected: GradientEpochSummary? { captured.first { String($0.epoch) == selectedEpoch } ?? captured.last }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionLabel(title: "Measured gradients")
            Text(message).font(.caption).textSelection(.enabled)
            if let inventory = inventory {
                Text("\(inventory.parameterCount) trainable parameter tensors · \(inventory.elementCount) elements · after backward, before optimizer step")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            if let latest = summaries.last {
                Text("Coverage through epoch \(latest.epoch) · \(summaries.count) observed epochs · \(summaries.reduce(0) { $0 + $1.observedBatchCount }) captured batches · \(latest.captureBudgetRemaining) captures remaining")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            if !summaries.isEmpty {
                Text("\(captured.count) epochs sampled · \(summaries.filter { $0.status == .notScheduled }.count) outside the capture schedule · \(summaries.filter { $0.status == .budgetExhausted }.count) skipped after the capture budget was exhausted")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            if !captured.isEmpty {
                FlatChoiceControl(title: "Gradient epoch", selection: String(selected?.epoch ?? 0), options: captured.map { FlatChoiceOption(id: String($0.epoch), title: "Epoch \($0.epoch) · \($0.observedBatchCount) of \($0.totalBatches) batches observed") }) { selectedEpoch = $0 }
            }
            if let selected {
                Text("Epoch \(selected.epoch): \(selected.observedTrainingRows) of \(selected.totalTrainingRows) training rows contribute to these sampled gradient summaries.")
                    .font(.caption)
                if selected.observedBatchCount < selected.scheduledBatchCount {
                    Text("Capture budget exhausted during this epoch: \(selected.observedBatchCount) of \(selected.scheduledBatchCount) scheduled batches were measured. Training continued.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
                ForEach(selected.batches, id: \.batchIndex) { batch in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Batch \(batch.batchIndex + 1) · \(batch.trainingRows) rows").font(.caption.bold())
                        Text("L2 norm \(number(batch.l2Norm)) · maximum absolute value \(number(batch.maxAbs))")
                            .font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        Text("\(batch.presentParameterCount) parameter tensors present · \(batch.missingParameterCount) missing · \(batch.finiteElementCount) finite elements · \(batch.nonfiniteElementCount) non-finite · \(batch.zeroElementCount) zero")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading).background(MLTheme.panel)
                }
                Text("Norms cover present gradients only. Missing gradients can be legitimate; a large norm alone does not establish exploding gradients. Unavailable norms mean there were no present gradient elements or non-finite values were observed.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            if let inventory = inventory {
                DisclosureGroup("Parameter coverage") {
                    ForEach(inventory.parameters, id: \.name) { parameter in
                        Text("\(parameter.name) · \(parameter.shape.map(String.init).joined(separator: " × ")) · \(parameter.elementCount) elements")
                            .font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                    }
                }
            }
        }.onChange(of: runID) { _, _ in selectedEpoch = "" }
    }
    private func number(_ value: Double?) -> String { value.map { String(format: "%.6g", $0) } ?? "unavailable" }
}
