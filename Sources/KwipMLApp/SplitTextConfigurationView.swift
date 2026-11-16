import SwiftUI
import KwipMLCore

struct SplitTextConfigurationView<Context: DataNodeEditing>: View {
    @ObservedObject var context: Context
    let node: DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    private var accent: Color { colorScheme == .dark ? ComponentVisualFamily.data.primary : ComponentVisualFamily.data.secondary }
    @State private var draft = try! SplitTextConfigurationDraft(configuration: SplitTextConfigurationDraft.defaults)
    @State private var original: [String: JSONValue] = [:]
    @State private var error: String?
    @State private var origin = ""
    @State private var search = ""
    @State private var dragField: UUID?
    @State private var numericText: [String: String] = [:]
    private var invalidNumeric: Bool { numericText.values.contains { text in guard let value = Int(text) else { return true }; return !(1...10_000_000).contains(value) } }
    private var pending: Bool { draft.configuration != original || invalidNumeric }
    private var schema: [DataColumnSchema] { context.upstreamSchema }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Source column").fontWeight(.semibold)
            FlatChoiceControl(title: "Source column", selection: draft.string("source"), options: schema.filter { $0.dtype == "string" }.map { .init(id: $0.name, title: $0.name) }, accent:accent) { draft.values["source"] = .string($0) }
            if schema.isEmpty { text("source", "Exact source column") }
            choice("mode", "Split mode", [("delimited", "Delimited"), ("fixedWidth", "Fixed width")])
            if draft.string("mode") == "delimited" {
                choice("outputMode", "Output", [("columns", "Columns"), ("rows", "Rows")])
                delimiterControls
                checkbox("quoted", "Treat separators inside quotes as literal text")
                if draft.bool("quoted") { text("quote", "Quote character") }
                choice("escaping", "Escaping", [("double", "Double quote"), ("escape", "Escape character"), ("none", "None")])
                if draft.string("escaping") == "escape" { text("escape", "Escape character") }
            } else {
                choice("measurement", "Measurement", [("codepoints", "Unicode code points"), ("utf8", "UTF-8 bytes")])
            }
            fieldsGrid
            if draft.expands {
                choice("emptyTokens", "Empty tokens", [("keep", "Keep"), ("drop", "Drop")])
                checkbox("ordinal", "Include token ordinal")
                if draft.bool("ordinal") { text("ordinalName", "Ordinal output name") }
            } else if draft.string("mode") == "delimited" {
                choice("extraFields", "Extra fields", [("error", "Stop run"), ("drop", "Drop"), ("remainder", "Remainder in final field")])
                choice("shortFields", "Short fields", [("missing", "Missing"), ("empty", "Empty text"), ("error", "Stop run")])
            } else {
                choice("shortRecord", "Short records", [("missing", "Missing fields"), ("error", "Stop run")])
            }
            checkbox("trim", "Remove leading and trailing whitespace")
            choice("conversionFailure", "Conversion failure", [("error", "Stop run"), ("missing", "Missing")])
            choice("missingSource", "Missing source", [("preserve", "Preserve missing"), ("error", "Stop run")])
            checkbox("retainSource", "Keep source column in output")
            HStack {
                Button("Apply", action: apply).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Button("Revert", action: reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer()
                if pending { Image(systemName: "pencil.circle").accessibilityLabel("Unapplied Split Text changes") }
            }
            if let error { Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
        }.font(.caption).tint(MLTheme.teal)
            .onAppear(perform: reload)
            .onChange(of: node.id) { _, _ in reload() }
            .onChange(of: node.config) { _, next in if !pending { reload() } else if next != original { error = "Saved configuration changed. Revert to load it; your draft is preserved." } }
            .onChange(of: draft.configuration) { _, _ in context.setNumericPending(origin: origin, key: "splitText", pending: pending) }
            .onDisappear { context.setNumericPending(origin: origin, key: "splitText", pending: false) }
    }
    private var delimiterControls: some View {
        VStack(alignment: .leading, spacing: 4) {
            FlatChoiceControl(title: "Separator preset", selection: [",", "\t", "|", ";"].contains(draft.string("delimiter")) ? draft.string("delimiter") : "custom", options: [( ",", "Comma"), ("\t", "Tab"), ("|", "Pipe"), (";", "Semicolon"), ("custom", "Custom")].map { .init(id: $0.0, title: $0.1) }) { if $0 != "custom" { draft.values["delimiter"] = .string($0) } else { draft.values["delimiter"] = .string("::") } }
            text("delimiter", "Literal separator")
        }
    }
    private var fieldsGrid: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(draft.expands ? "Token output" : "Output fields").fontWeight(.semibold)
            if !draft.expands { TextField("Find output field", text: $search).flatField() }
            ThemedScrollView(.horizontal) {
                VStack(alignment: .leading, spacing: 5) {
                    HStack { Text("Order").frame(width: 28); Text("Name").frame(width: 100); if draft.string("mode") == "fixedWidth" { Text("Start").frame(width: 56); Text("Width").frame(width: 56) }; Text("Type").frame(width: 88) }
                    ThemedScrollView {
                        LazyVStack(alignment: .leading, spacing: 5) {
                    ForEach(draft.expands ? Array(draft.fields.prefix(1)) : draft.fields.filter { search.isEmpty || $0.name.localizedCaseInsensitiveContains(search) }) { field in
                        HStack(spacing: 5) {
                            Image(systemName: "line.3.horizontal").frame(width: 28)
                                .onDrag { dragField = field.id; return NSItemProvider(object: field.id.uuidString as NSString) }
                            TextField("Output name", text: fieldString(field.id, \.name)).flatField().frame(width: 100)
                            if draft.string("mode") == "fixedWidth" {
                                TextField("Start", text: fieldNumber(field.id, "start", \.start)).flatField().frame(width: 56)
                                TextField("Width", text: fieldNumber(field.id, "width", \.width)).flatField().frame(width: 56)
                            }
                            FlatChoiceControl(title: "Output type", selection: field.type, options: [("string", "Text"), ("int64", "Integer"), ("float64", "Number"), ("bool", "Boolean")].map { .init(id: $0.0, title: $0.1) }, accent:accent) { next in if let i = draft.fields.firstIndex(where: { $0.id == field.id }) { draft.fields[i].type = next } }.frame(width: 88)
                            if !draft.expands {
                                Button { move(field.id, -1) } label: { Image(systemName: "arrow.up") }.disabled(draft.fields.first?.id == field.id)
                                Button { move(field.id, 1) } label: { Image(systemName: "arrow.down") }.disabled(draft.fields.last?.id == field.id)
                                Button { draft.fields.removeAll { $0.id == field.id }; numericText = numericText.filter { !$0.key.hasPrefix(field.id.uuidString) } } label: { Image(systemName: "minus") }.disabled(draft.fields.count == 1)
                            }
                        }.buttonStyle(FlatButtonStyle())
                            .onDrop(of: ["public.text"], isTargeted: nil) { _ in
                                guard let from = dragField, let i = draft.fields.firstIndex(where: { $0.id == from }), let j = draft.fields.firstIndex(where: { $0.id == field.id }) else { return false }
                                let row = draft.fields.remove(at: i); draft.fields.insert(row, at: j); dragField = nil; return true
                            }
                    }
                        }
                    }.frame(height: min(230, max(42, CGFloat(draft.expands ? 1 : draft.fields.count) * 38)))
                }
            }.frame(maxHeight: 260)
            if !draft.expands { Button("+ Add field") { draft.fields.append(.init(name: "part_\(draft.fields.count + 1)")) }.buttonStyle(FlatButtonStyle()).disabled(draft.fields.count >= 128) }
        }
    }
    private func fieldString(_ id: UUID, _ key: WritableKeyPath<SplitTextConfigurationDraft.Field, String>) -> Binding<String> { Binding(get: { draft.fields.first { $0.id == id }?[keyPath: key] ?? "" }, set: { value in if let i = draft.fields.firstIndex(where: { $0.id == id }) { draft.fields[i][keyPath: key] = value } }) }
    private func fieldNumber(_ id: UUID, _ label: String, _ key: WritableKeyPath<SplitTextConfigurationDraft.Field, Int>) -> Binding<String> {
        let token = id.uuidString + ":" + label
        return Binding(get: { numericText[token] ?? String(draft.fields.first { $0.id == id }?[keyPath: key] ?? 1) }, set: { text in
            numericText[token] = text
            if let value = Int(text), let index = draft.fields.firstIndex(where: { $0.id == id }) { draft.fields[index][keyPath: key] = value }
            context.setNumericPending(origin: origin, key: "splitText", pending: pending)
        })
    }
    private func move(_ id: UUID, _ delta: Int) { if let i = draft.fields.firstIndex(where: { $0.id == id }), draft.fields.indices.contains(i + delta) { draft.fields.swapAt(i, i + delta) } }
    private func text(_ key: String, _ title: String) -> some View { VStack(alignment: .leading, spacing: 4) { Text(title); TextField(title, text: Binding(get: { draft.string(key) }, set: { draft.values[key] = .string($0) })).flatField() } }
    private func checkbox(_ key: String, _ title: String) -> some View { Toggle(title, isOn: Binding(get: { draft.bool(key) }, set: { draft.values[key] = .bool($0) })).toggleStyle(FileInputCheckboxStyle(tint:accent)) }
    private func choice(_ key: String, _ title: String, _ options: [(String, String)]) -> some View { VStack(alignment: .leading, spacing: 4) { Text(title); FlatChoiceControl(title: title, selection: draft.string(key), options: options.map { .init(id: $0.0, title: $0.1) }, accent:accent) { draft.values[key] = .string($0) } } }
    private func reload() {
        context.setNumericPending(origin: origin, key: "splitText", pending: false)
        do { original = node.config; draft = try .init(configuration: node.config); numericText = [:]; origin = context.numericOrigin(key: "splitText"); error = nil } catch { self.error = error.localizedDescription }
    }
    private func apply() {
        do { guard !invalidNumeric else { throw DataFlowContractError("Start and Width must be whole numbers from 1 through 10,000,000.") }; let config = try draft.validatedConfiguration(schema: schema); if context.applyTableConfiguration(config, expected: original) { original = config; error = nil; context.setNumericPending(origin: origin, key: "splitText", pending: false) } else { error = context.statusMessage } } catch { self.error = error.localizedDescription }
    }
}
