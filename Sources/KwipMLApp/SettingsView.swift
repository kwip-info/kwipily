import SwiftUI
import KwipMLCore
struct SettingsView: View {
    @ObservedObject var store: WorkspaceStore
    @State private var rateDraft = ""
    @State private var rateError: String?
    var body: some View {
        ProjectConfigureShell(universal: false) { page in
            VStack(alignment: .leading, spacing: 18) {
                switch page {
                case .identity:
                    SectionLabel(title: "Identity")
                    TextField("Project name", text: Binding(get: { store.graph.name }, set: { store.renameProject($0) })).flatField().accessibilityLabel("Project name").frame(maxWidth: 430)
                case .training:
                    SectionLabel(title: "Compute")
                    FlatChoiceControl(title: "Compute device", selection: store.execution.device, options: [
                    FlatChoiceOption(id: "auto", title: "Automatic · prefer Apple GPU"),
                    FlatChoiceOption(id: "cpu", title: "CPU"),
                    FlatChoiceOption(id: "mps", title: store.availableDevices.contains("mps") ? "Apple GPU · MPS" : "Apple GPU · unavailable", isEnabled: store.availableDevices.contains("mps"))
                    ]) { value in store.setExecution { $0.device = value } }.frame(maxWidth: 430)
                    Text(store.capabilityStatus).font(.system(size: 12)).foregroundStyle(MLTheme.teal)
                    if let contract = store.runtimeExecutionContract {
                    ExecutionContractView(contract: contract)
                    if let embedding=store.embeddingRuntimeCapability {
                        Text("Worker lookup extension verified · int64 input · float32 weights. Component availability also depends on this app’s registered tools.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        Text(embedding.mpsMaximumNormExplanation).font(.caption).foregroundStyle(MLTheme.muted)
                        Text("The current Adam training recipe requires dense trainable gradients. Freezing a lookup table does not prevent maximum-norm updates during its forward pass.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }
                }
                Divider(); SectionLabel(title: "Run budget")
                control("Epochs", "Complete passes through the selected training rows.", store.execution.epochs, 1, 10000) { $0.epochs = $1 }
                control("Batch size", "Maximum observations per optimizer update. The final batch may be smaller.", store.execution.batchSize, 1, 100000) { $0.batchSize = $1 }
                control("Sample limit", "0 uses all rows. Otherwise select at least 20 rows, preserving both classes; requests larger than the source are rejected.", store.execution.sampleLimit, 0, 100000) { $0.sampleLimit = $1 }.disabled(store.execution.preparedTraining != nil)
                Divider(); SectionLabel(title: "Reproducibility and splits")
                control("Model seed", "Controls initial weights independently of data selection.", store.execution.seed, 0, 4294967295) { $0.seed = $1 }
                control("Split seed", "Controls reproducible stratified row selection and train/validation membership.", store.execution.splitSeed, 0, 4294967295) { $0.splitSeed = $1 }.disabled(store.execution.preparedTraining != nil)
                control("Validation percent", "Held-out portion of selected data. Changing this changes the evaluation population.", store.execution.validationPercent, 5, 50) { $0.validationPercent = $1 }.disabled(store.execution.preparedTraining != nil)
                if store.execution.preparedTraining != nil { Text("Prepared partitions are selected. Change sampling and split settings in the preparation canvas, then select its new outputs.").font(.caption).foregroundStyle(MLTheme.teal) }
                Text("For a split A/B test, keep architecture, model seed and training settings fixed. Run once, change only the split seed or validation percentage, then compare saved experiments.").font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                Button("Compare saved experiments") { store.activeSection = "Experiments" }.buttonStyle(FlatButtonStyle())
                Divider(); SectionLabel(title: "Optimizer")
                Text("Adam learning rate").font(.system(size:12,weight:.medium))
                HStack {
                    TextField("Learning rate", text: $rateDraft).flatField().accessibilityLabel("Learning rate").onSubmit { applyRate() }
                    Button("Apply") { applyRate() }.buttonStyle(FlatButtonStyle())
                }.frame(maxWidth:430)
                Text(rateError ?? "Enter a finite decimal greater than 0 and no larger than 1; press Return or Apply.").font(.caption).foregroundStyle(rateError == nil ? MLTheme.muted : .orange)
                case .webhooks:
                    EmptyView() // Only managed universal workflows support webhook jobs.
                case .schedules:
                    Text("Scheduled runs are available for saved workflow canvases.")
                case .resources:
                    if let projectID = store.activeProjectID {
                        ExecutionResourceSettingsView(store: store, projectID: projectID)
                        Text("Legacy project controls are locked during active work. App execution defaults remain editable in \(ProductBrand.name) → Settings.").font(.caption).foregroundStyle(MLTheme.muted)
                    } else {
                        Text("Save this project to set execution overrides. App execution defaults apply until then.").font(.caption).foregroundStyle(MLTheme.muted)
                    }
                case .capture:
                    if let projectID = store.activeProjectID {
                        TrainingCaptureSettingsView(store: store, projectID: projectID)
                        Divider()
                    } else {
                        Text("Save this project to set sample capture overrides. App capture defaults apply until then.").font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    GradientCaptureSettingsView(workspace: store, capture: store.gradientCapture)
                case .workspace:
                    ProjectCanvasSettingsView(workspace:store,canvas:store)
                case .files:
                    EmptyView()
                }
            }.frame(maxWidth: 650, alignment: .leading).frame(maxWidth: .infinity, alignment: .leading).disabled(store.isBusy)
        }.onAppear { rateDraft = String(store.execution.learningRate) }
        .onChange(of: store.execution.learningRate) { _, rate in rateDraft = String(rate) }
    }
    private func control(_ title: String, _ hint: String, _ value: Int, _ minimum: Int, _ maximum: Int, update: @escaping (inout ExecutionSettings, Int) -> Void) -> some View {
        ParameterControl(title: title, hint: hint, value: Double(value), minimum: minimum, maximum: maximum, enabled: !store.isBusy) { value in store.setExecution { update(&$0, Int(value)) } }.frame(maxWidth: 430)
    }
    private func applyRate() {
        guard let value = Double(rateDraft), value.isFinite, value > 0, value <= 1 else { rateError = "Learning rate must be a number greater than 0 and at most 1."; return }
        rateError = nil; store.setExecution { $0.learningRate = value }
    }
}
