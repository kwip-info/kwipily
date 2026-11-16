import SwiftUI
import KwipMLCore

/// The caller supplies the existing Configuration/Information inspector frames.
struct GradientRequestsEditorView: View {
    @ObservedObject var context: GradientPolicyInspectorContext
    var inlinePresentation = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !inlinePresentation { Text("Choose a result and ask how changing an input or learned setting changes it. Requests inspect the final model in evaluation mode; they do not perform another training update.")
                .font(.caption).foregroundStyle(MLTheme.muted) }
            numeric("accumulation_steps", title: "Small groups per update")
            FlatChoiceControl(title: "Combine changes", selection: context.draftConfiguration["gradient_scale"]?.stringValue ?? "mean", options: [.init(id: "mean", title: "Mean"), .init(id: "sum", title: "Sum")]) { _ = context.setGradientScale($0) }
            if !inlinePresentation { Text("The combination must match the selected scoring rule. The final short group still makes an update.")
                .font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(Array(context.requests.enumerated()), id: \.offset) { index, row in request(index, row) }
            Button { _ = context.addRequest() } label: { Label("Add derivative request", systemImage: "plus") }
                .buttonStyle(FlatButtonStyle()).disabled(context.requests.count >= 32 || context.hasUncommittedNumbers)
            if context.endpoints.isEmpty { Text("Connect a model to the intended Train block to resolve its real input and result endpoints.").font(.caption).foregroundStyle(MLTheme.muted) }
            if let message = context.validationMessage { Text(message).font(.caption).foregroundStyle(MLTheme.rose) }
            if let rejection = context.rejection { Text(rejection).font(.caption).foregroundStyle(MLTheme.rose) }
            if !inlinePresentation { Text("Requesting values also requires the separate Selected derivative values capture permission. This editor does not enable it.")
                .font(.caption).foregroundStyle(MLTheme.muted) }
            if !inlinePresentation { HStack {
                Button("Apply") { _ = context.apply() }.disabled(!context.canApply)
                Button("Discard") { _ = context.discard() }.disabled(!context.hasChanges && !context.hasUncommittedNumbers)
                Spacer(minLength: 0)
                if context.hasChanges || context.hasUncommittedNumbers { Text("Unapplied").font(.caption2).foregroundStyle(MLTheme.muted) }
            }.buttonStyle(FlatButtonStyle()) }
        }.disabled(context.isBusy).id(context.resetGeneration)
    }
    private func request(_ index: Int, _ row: [String: JSONValue]) -> some View {
        let output: GradientEndpointChoice? = {
            guard case .object(let selected) = row["output"] else { return nil }
            return context.endpoints.first { $0.node == selected["node"]?.stringValue && $0.port == selected["port"]?.stringValue }
        }()
        return VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text("Request \(index + 1)").font(.caption.weight(.semibold))
                Spacer()
                Button { _ = context.moveRequest(from: index, to: index - 1) } label: { Image(systemName: "arrow.up") }
                    .disabled(index == 0 || context.hasUncommittedNumbers).accessibilityLabel("Move request \(index + 1) earlier")
                Button { _ = context.moveRequest(from: index, to: index + 1) } label: { Image(systemName: "arrow.down") }
                    .disabled(index + 1 >= context.requests.count || context.hasUncommittedNumbers).accessibilityLabel("Move request \(index + 1) later")
                Button { _ = context.removeRequest(at: index) } label: { Image(systemName: "trash") }
                    .disabled(context.hasUncommittedNumbers).accessibilityLabel("Remove request \(index + 1)")
            }.buttonStyle(.plain)
            TextField("Request name: letters, numbers, underscore or dash", text: Binding(get: { row["id"]?.stringValue ?? "" }, set: { _ = context.setRequestID($0, at: index) })).flatField()
            FlatChoiceControl(title: "Result endpoint", selection: output?.id ?? "", options: context.endpoints.filter { $0.dtype == .float32 }.map { .init(id: $0.id, title: $0.title) }, placeholder: "Choose a current result") { _ = context.setOutput($0, at: index) }
            if output == nil, case .object(let selected) = row["output"] {
                Text("Unavailable result: \(selected["node"]?.stringValue ?? "").\(selected["port"]?.stringValue ?? ""). Choose a replacement or remove this request.").font(.caption).foregroundStyle(MLTheme.rose)
            }
            FlatChoiceControl(title: "Observation partition", selection: row["partition"]?.stringValue ?? "validation", options: [.init(id: "training", title: "Training observations"), .init(id: "validation", title: "Held-out observations")]) { _ = context.setPartition($0, at: index) }
            numeric("request.\(index).row_start", title: "First row (starts at 0)")
            numeric("request.\(index).row_count", title: "Number of rows (1–32)")
            FlatChoiceControl(title: "Combine selected result values", selection: row["reduction"]?.stringValue ?? "sum", options: [.init(id: "sum", title: "Sum"), .init(id: "mean", title: "Mean"), .init(id: "upstream", title: "Use a signed influence array")]) { _ = context.setReduction($0, at: index) }
                .disabled(context.hasUncommittedNumbers)
            if row["reduction"] == .string("upstream") {
                if !inlinePresentation { Text("The signed influence array weights each selected result value. Its shape is the row count followed by the result dimensions. Enter numbers in row order, separated by commas or spaces. When resizing, edit both shape and values, then use the array to commit them together.").font(.caption).foregroundStyle(MLTheme.muted) }
                numeric("request.\(index).upstream.shape", title: "Influence array shape")
                numeric("request.\(index).upstream.values", title: "Signed influence values")
            }
            targets(row, index: index)
            FlatChoiceControl(title: "When a derivative is unavailable", selection: row["unavailable"]?.stringValue ?? "error", options: [.init(id: "error", title: "Refuse the request"), .init(id: "reportUnavailable", title: "Report why it is unavailable")]) { _ = context.setUnavailable($0, at: index) }
            if !inlinePresentation { Text("Integer inputs, frozen settings or disconnected paths can have no derivative values. Unavailable does not mean zero. Unsupported native backward operations still refuse execution.").font(.caption).foregroundStyle(MLTheme.muted) }
        }.padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    @ViewBuilder private func targets(_ row: [String: JSONValue], index: Int) -> some View {
        Text("What may change?").font(.caption.weight(.semibold))
        if case .array(let targets) = row["targets"] {
            ForEach(Array(targets.enumerated()), id: \.offset) { targetIndex, value in
                if case .object(let target) = value {
                    HStack {
                        Text(targetTitle(target)).font(.caption).foregroundStyle(targetAvailable(target) ? Color.primary : MLTheme.rose)
                        Spacer()
                        Button { _ = context.removeTarget(at: targetIndex, request: index) } label: { Image(systemName: "trash") }
                            .buttonStyle(.plain).accessibilityLabel("Remove derivative target \(targetIndex + 1)")
                    }
                }
            }
        }
        FlatChoiceControl(title: "Add model input", selection: "", options: context.endpoints.filter { $0.isModelInput && [.float32, .int64].contains($0.dtype) }.map { .init(id: $0.id, title: $0.title) }, placeholder: "Choose an input") { _ = context.addInputTarget($0, at: index) }
        FlatChoiceControl(title: "Add learned setting", selection: "", options: context.owners.map { .init(id: $0.id, title: $0.title) }, placeholder: "Choose weights or biases") { _ = context.addParameterTarget($0, at: index) }
    }
    private func targetAvailable(_ target: [String: JSONValue]) -> Bool {
        if target["kind"] == .string("input") { return context.endpoints.contains { $0.isModelInput && $0.node == target["node"]?.stringValue && $0.port == target["port"]?.stringValue } }
        return context.owners.contains { $0.node == target["node"]?.stringValue && $0.parameter == target["parameter"]?.stringValue }
    }
    private func targetTitle(_ target: [String: JSONValue]) -> String {
        let name = (target["node"]?.stringValue ?? "") + " · " + (target[target["kind"] == .string("input") ? "port" : "parameter"]?.stringValue ?? "")
        return (targetAvailable(target) ? "" : "Unavailable · ") + name
    }
    private func numeric(_ key: String, title: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption)
            HStack {
                TextField(title, text: Binding(get: { context.rawNumericDrafts[key] ?? numericText(key) }, set: { _ = context.setRawNumericDraft(key, text: $0, origin: context.numericOrigin(key)) }))
                    .flatField().onSubmit { _ = context.commitNumericDraft(key) }
                Button(key.contains(".upstream.") ? "Use array" : "Use value") { _ = context.commitNumericDraft(key) }
                    .buttonStyle(FlatButtonStyle()).disabled(context.rawNumericDrafts[key] == nil)
            }
        }
    }
    private func numericText(_ key: String) -> String {
        if key == "accumulation_steps" { return context.draftConfiguration[key]?.integerValue.map(String.init) ?? "" }
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 3, let index = Int(parts[1]), context.requests.indices.contains(index) else { return "" }
        let row = context.requests[index]
        if parts.count == 3 { return row[parts[2]]?.integerValue.map(String.init) ?? "" }
        guard case .object(let upstream) = row["upstream"], case .array(let values) = upstream[parts[3]] else { return "" }
        return values.map { value in value.integerValue.map(String.init) ?? value.doubleValue.map { NumericValue.decimal($0).text } ?? "" }.joined(separator: ", ")
    }
}
