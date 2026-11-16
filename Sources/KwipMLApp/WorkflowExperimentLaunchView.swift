import SwiftUI
import KwipMLCore

struct WorkflowExperimentLaunchView: View {
    @ObservedObject var context: WorkflowControlNodeContext
    @ObservedObject var host: WorkflowWorkspaceSession
    @State private var editing = false
    @Environment(\.colorScheme) private var colorScheme
    private var specification: WorkflowExperimentSpecification? { context.experimentSpecification }
    private var summary: String {
        guard let specification else { return "Not configured" }
        return "\(specification.axes.reduce(1) { $0 * $1.values.count }) combinations"
    }
    var body: some View {
        TrainInspectorSection("Experiments", symbol: "square.grid.2x2", summary: summary, initiallyExpanded: false) {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Spacer()
                Button("Edit combinations") { editing = true }.buttonStyle(FlatButtonStyle())
                    .disabled(context.isBusy || context.experimentConfiguration == nil || host.experimentRunner?.isRunning == true)
                    .accessibilityIdentifier("train.experiments.edit")
            }
            if let specification {
                HStack {
                    Text("\(specification.axes.reduce(1) { $0 * $1.values.count }) combinations").font(.caption.weight(.medium))
                    Spacer()
                    Text("≤ \(specification.budget.maximumTotalEpochs) epochs").font(.caption.monospaced()).foregroundStyle(MLTheme.muted)
                }
                VStack(spacing: 0) {
                    ForEach(specification.axes, id: \.field) { axis in
                        ExperimentAxisValuesView(title: WorkflowControlContracts.experimentFields.first { $0.key == axis.field }?.title ?? axis.field, values: axis.values)
                        Divider()
                    }
                }
                Button("Run experiment group") { _ = host.runExperiments(train: context.authority.address) }
                    .buttonStyle(FlatButtonStyle()).disabled(context.isBusy || host.owner.hasPendingInspectorEdits || host.experimentRunner?.isRunning == true)
                    .accessibilityIdentifier("train.experiments.run")
            } else { Text("No combinations configured").font(.caption).foregroundStyle(MLTheme.muted) }
        }
        }.sheet(isPresented: $editing) { WorkflowExperimentEditor(context: context) }
    }
}
