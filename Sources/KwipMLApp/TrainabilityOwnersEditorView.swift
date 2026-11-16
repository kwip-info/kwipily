import SwiftUI
import KwipMLCore

/// Callers supply the shared outer Configuration and Information frames.
struct TrainabilityOwnersEditorView: View {
    @ObservedObject var context: TrainabilityInspectorContext
    var inlinePresentation = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !inlinePresentation { Text("Choose which weights and biases may change. Inherit keeps the model’s original setting; Trainable allows adjustments; Frozen keeps the array unchanged. This does not change training or evaluation mode.")
                .font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(Array(context.rows.enumerated()), id: \.offset) { index, value in
                if case .object(let row) = value {
                    let selected = context.owners.first { $0.node == row["node"]?.stringValue && $0.parameter == row["parameter"]?.stringValue }
                    VStack(alignment: .leading, spacing: 9) {
                        HStack {
                            Text("Rule \(index + 1)").font(.caption.weight(.semibold))
                            Spacer()
                            Button { _ = context.removeOwner(at: index) } label: { Image(systemName: "trash") }
                                .buttonStyle(.plain).accessibilityLabel("Remove weights or biases \(index + 1)")
                        }
                        FlatChoiceControl(title: "Weights or biases", selection: selected?.id ?? "", options: context.owners.map { .init(id: $0.id, title: $0.title) }, placeholder: "Choose weights or biases") { _ = context.setOwner($0, at: index) }
                        if selected == nil {
                            Text("Layer or parameter unavailable. Choose a replacement or remove this override.")
                                .font(.caption).foregroundStyle(MLTheme.rose)
                        }
                        FlatChoiceControl(title: "Adjustment behavior", selection: row["mode"]?.stringValue ?? "inherit", options: [.init(id: "inherit", title: "Inherit"), .init(id: "trainable", title: "Trainable"), .init(id: "frozen", title: "Frozen")]) { _ = context.setMode($0, at: index) }
                    }.padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
            FlatChoiceControl(title: "Add weights or biases", selection: "", options: context.owners.filter { owner in
                !context.rows.contains { value in
                    guard case .object(let row) = value else { return false }
                    return row["node"] == .string(owner.node) && row["parameter"] == .string(owner.parameter)
                }
            }.map { .init(id: $0.id, title: $0.title) }, placeholder: "Choose weights or biases") { _ = context.addOwner($0) }
                .disabled(context.rows.count >= TrainabilityConfiguration.maximumOwners)
            if context.owners.isEmpty { Text("Connect a model to the intended Train block to resolve its weights and biases.").font(.caption).foregroundStyle(MLTheme.muted) }
            if let message = context.validationMessage { Text(message).font(.caption).foregroundStyle(MLTheme.rose) }
            if let rejection = context.rejection { Text(rejection).font(.caption).foregroundStyle(MLTheme.rose) }
            if !inlinePresentation { HStack {
                Button("Apply") { _ = context.apply() }.buttonStyle(FlatButtonStyle()).disabled(!context.canApply)
                Button("Discard") { _ = context.discard() }.buttonStyle(FlatButtonStyle()).disabled(!context.hasChanges)
            } }
        }.disabled(context.isBusy)
    }
}
