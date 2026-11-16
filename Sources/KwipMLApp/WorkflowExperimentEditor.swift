import SwiftUI
import KwipMLCore

struct WorkflowExperimentEditor:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft:WorkflowExperimentDraft
    @State private var axisValues:ExperimentAxisValuesDraft
    private let expected:[String:JSONValue]
    @State private var message=""
    @State private var selectedTrain=""
    @State private var preview:WorkflowExperimentPlan?

    init(context:WorkflowControlNodeContext) {
        self.context=context
        expected=context.experimentConfiguration ?? context.configuration
        _draft=State(initialValue:.init(specification:context.experimentSpecification))
        _axisValues=State(initialValue:.init(specification:context.experimentSpecification))
    }
    private var baseEpochs:Int {
        let run: [String:JSONValue]
        if context.isTrain,case .object(let nested)=expected["runSetup"] {run=nested} else {run=expected}
        if case .integer(let value)=run["epochs"] {return Int(value)}
        return ExecutionSettings().epochs
    }
    private var summary:Result<(runs:Int,epochs:Int),Error> {
        Result { try axisValues.validate(); return try draft.summary(baseEpochs:baseEpochs) }
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Experiment combinations").font(.headline)
            Text("Add individual values to compare. Every combination is included; unchanged settings use the ordinary run.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            ScrollView {
                VStack(alignment:.leading,spacing:10) {
                    ForEach(WorkflowControlContracts.experimentFields,id:\.key) {field in
                        axisEditor(field)
                    }
                    Text("Maximum runs · 1–32").font(.caption)
                    TextField("Maximum runs",text:$draft.maximumRuns).flatField()
                    Text("Total epoch budget · 1–320000").font(.caption)
                    TextField("Total epoch budget",text:$draft.maximumTotalEpochs).flatField()
                    previewSection
                }
            }
            switch summary {
            case .success(let totals):
                Text("\(totals.runs) combinations · \(totals.epochs) total epochs").font(.caption)
            case .failure(let error):
                Text(error.localizedDescription).font(.caption).foregroundStyle(MLTheme.rose)
            }
            Text("Saving configures the canvas only. Ordinary Run still uses its single-run settings. Run the experiment group from this Train.")
                .font(.caption2).foregroundStyle(MLTheme.muted)
            if !message.isEmpty {Text(message).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {
                Button("Cancel") {dismiss()}.buttonStyle(FlatButtonStyle())
                Spacer()
                Button("Clear") {save(nil)}.buttonStyle(FlatButtonStyle()).disabled(context.isBusy)
                Button("Save combinations") {
                    do { try axisValues.validate(); save(try draft.specification()) } catch {message=error.localizedDescription}
                }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy || !canSave)
            }
        }.padding(20).frame(width:480,height:640).background(MLTheme.background)
        .onAppear {
            pending(true)
            if context.experimentTrainNodes.count == 1 {selectedTrain=context.experimentTrainNodes[0].address.localID}
        }
        .onChange(of:draft) {_,_ in preview=nil}
        .onChange(of:selectedTrain) {_,_ in preview=nil}
        .onDisappear {pending(false)}
    }
    private func updateValues(_ change: (inout ExperimentAxisValuesDraft) -> Void) {
        change(&axisValues)
        draft = axisValues.applying(to: draft)
    }
    private func axisEditor(_ field: WorkflowRunField) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(field.title).font(.caption.weight(.semibold)).foregroundStyle(colorScheme == .dark ? ComponentVisualFamily.training.primary : ComponentVisualFamily.training.secondary)
                Spacer()
                Button { updateValues { $0.add(field.key) } } label: { Label("Add value", systemImage: "plus") }
                    .buttonStyle(FlatButtonStyle()).disabled((axisValues.values[field.key]?.count ?? 0) >= 32)
                    .accessibilityIdentifier("experiment.\(field.key).add")
            }
            ForEach(Array((axisValues.values[field.key] ?? []).enumerated()), id: \.offset) { index, text in
                HStack(spacing: 7) {
                    Text(String(index + 1)).font(.caption2.monospaced()).foregroundStyle(MLTheme.muted).frame(width: 18)
                    if case .choice(let choices) = field.kind {
                        FlatChoiceControl(title: field.title, selection: text, options: choices.map { .init(id: $0, title: $0) }, placeholder: "Choose value") { next in
                            updateValues { $0.set(next, key: field.key, index: index) }
                        }
                    } else {
                        TextField("Value", text: Binding(get: { axisValues.values[field.key]?[index] ?? "" }, set: { next in
                            updateValues { $0.set(next, key: field.key, index: index) }
                        })).flatField().accessibilityLabel("\(field.title) value \(index + 1)")
                    }
                    Button { updateValues { $0.remove(field.key, index: index) } } label: { Image(systemName: "xmark") }
                        .buttonStyle(.plain).accessibilityLabel("Remove \(field.title) value \(index + 1)")
                }
            }
        }.padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private var previewSection:some View {
        VStack(alignment:.leading,spacing:8) {
            Text("Validate connected training path").font(.caption)
            if !context.isTrain {Picker("Train component",selection:$selectedTrain) {
                Text("Choose Train").tag("")
                ForEach(context.experimentTrainNodes,id:\.address) {node in
                    Text(node.name.isEmpty ? node.address.localID:node.name).tag(node.address.localID)
                }
            }.labelsHidden()}
            Button("Preview exact runs") {
                do {
                    try axisValues.validate()
                    let specification=try draft.specification()
                    preview=try context.previewExperiment(specification,
                        train:.init(domain:.control,localID:selectedTrain),expected:expected)
                    message=""
                } catch {preview=nil;message=error.localizedDescription}
            }.buttonStyle(FlatButtonStyle()).disabled(context.isBusy || selectedTrain.isEmpty || !canSave)
            if let preview {
                Text("Validated \(preview.variants.count) runs · \(preview.totalEpochs) epochs")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(preview.variants,id:\.ordinal) {variant in
                    let settings=variant.execution.settings
                    VStack(alignment:.leading,spacing:3) {
                        Text("Run \(variant.ordinal)").font(.caption.bold())
                        Text("Device \(settings.device) · Seed \(settings.seed) · Batch \(settings.batchSize)")
                        Text("Epochs \(settings.epochs) · Learning rate \(String(settings.learningRate))")
                    }.font(.caption2).textSelection(.enabled)
                }
                Text("This checks the graph and settings. Data files and device availability are checked when executing; inputs have not been captured.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private var canSave:Bool {if case .success=summary {return true};return false}
    private func pending(_ value:Bool) {
        let key="experimentEditor"
        context.setNumericPending(origin:context.numericOrigin(key:key),key:key,pending:value)
    }
    private func save(_ specification:WorkflowExperimentSpecification?) {
        pending(false)
        if context.setExperimentSpecification(specification,expected:expected) {dismiss()}
        else {
            pending(true)
            message=context.statusMessage.isEmpty ? "This editor is no longer current. Close it and reopen Train.":context.statusMessage
        }
    }
}
