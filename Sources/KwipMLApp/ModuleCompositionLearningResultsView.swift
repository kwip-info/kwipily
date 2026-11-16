import SwiftUI
import KwipMLCore

/// Receives only current public projections. Private assessment fixtures never enter this view.
struct ModuleCompositionLearningResultsView: View {
    let task: ModuleCompositionLearningTask
    let projection: ModuleCompositionFormativeProjection
    let metadata: [ModuleCompositionLearningMetadata]
    var nodeLabels: [String: String] = [:]
    @State private var selectedFixture = ""
    @State private var selectedCheck = ""
    private var fixtureIndex: Int { projection.fixtures.firstIndex { $0.fixtureID == selectedFixture } ?? 0 }
    var body: some View {
        if projection.fixtures.indices.contains(fixtureIndex), metadata.indices.contains(fixtureIndex) {
            let fixture = projection.fixtures[fixtureIndex]
            let record = metadata[fixtureIndex]
            VStack(alignment: .leading, spacing: 12) {
                Text("Measured construction").font(.system(size: 13, weight: .semibold))
                Text("The checker runs your construction with four small examples. These comparisons show the difference from an independent calculation: zero means a match within the check’s tolerance. A structural failure can also produce a positive failure marker.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                FlatChoiceControl(title: "Example", selection: fixture.fixtureID,
                    options: projection.fixtures.enumerated().map { .init(id: $0.element.fixtureID, title: "Example \($0.offset + 1)") }) { selectedFixture = $0 }
                let checks = task.measuredShapes.keys.sorted()
                let key = checks.contains(selectedCheck) ? selectedCheck : (checks.first ?? "")
                FlatChoiceControl(title: "Measured check", selection: key,
                    options: checks.map { .init(id: $0, title: ModuleCompositionLearningTask.label($0)) }) { selectedCheck = $0 }
                if let tensor = fixture.tensors[key] {
                    LearningTensorTable(tensor: tensor, title: ModuleCompositionLearningTask.label(key) + " · comparison", signed: true)
                        .id(fixture.fixtureID + ":" + key)
                }
                InspectorSection("Container structure", symbol: "square.stack.3d.up", identifier: "lesson.composition.containers") {
                    Text("A sequence runs its children in order. Collections store children; the connections decide how their results are used.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    ForEach(Array(record.nativeContainers.enumerated()), id: \.offset) { _, container in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(ModuleCompositionPresentation.title(container.kind)).font(.caption.weight(.semibold))
                                Spacer()
                                Text(container.matches ? "Matches connected structure" : "Structure differs")
                                    .font(.caption).foregroundStyle(container.matches ? MLTheme.teal : MLTheme.rose)
                            }
                            Text("Stored children: " + container.keys.joined(separator: " → ")).font(.caption)
                            Text(container.path).font(.system(size: 10, design: .monospaced)).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                        }.padding(8).frame(maxWidth: .infinity, alignment: .leading).overlay(Rectangle().stroke(MLTheme.rule))
                    }
                }
                InspectorSection("Learned settings · after checks", symbol: "slider.horizontal.3", identifier: "lesson.composition.owners") {
                    Text("Weights combine incoming numbers; offsets shift the result. Each private copy owns its own settings." + (task == .buildReuse ? " One weight is deliberately held fixed during the behavior check." : ""))
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    Text(record.independentStorage ? "Private settings use separate storage." : "Private settings share storage unexpectedly.")
                        .font(.caption).foregroundStyle(record.independentStorage ? MLTheme.teal : MLTheme.rose)
                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                        GridRow {
                            Text("Component"); Text("Setting"); Text("Array shape"); Text("After checks")
                        }.font(.caption.weight(.semibold)).foregroundStyle(MLTheme.muted)
                        ForEach(Array(record.nativeOwners.enumerated()), id: \.offset) { _, owner in
                            GridRow {
                                Text(nodeLabels[owner.node] ?? owner.node).lineLimit(2).help(owner.name)
                                Text(owner.parameter == "bias" ? "Offset" : "Weights")
                                Text("[" + owner.shape.map(String.init).joined(separator: ", ") + "]").monospacedDigit()
                                Text(owner.requiresGrad ? "Can adjust" : "Fixed for check")
                            }.font(.caption)
                            Divider().gridCellColumns(4)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
                InspectorSection("Execution order", symbol: "list.number", identifier: "lesson.composition.invocations", initiallyExpanded: false) {
                    ForEach(Array(record.invocations.filter { $0.phase == "completed" }.enumerated()), id: \.offset) { index, invocation in
                        HStack(spacing: 10) {
                            Text(String(index + 1)).monospacedDigit().foregroundStyle(MLTheme.muted).frame(width: 24, alignment: .trailing)
                            Text(nodeLabels[invocation.node] ?? invocation.node)
                            Spacer()
                            Text("Completed").foregroundStyle(MLTheme.teal)
                        }.font(.caption).padding(.vertical, 4)
                        Divider()
                    }
                }
            }
        }
    }
}
