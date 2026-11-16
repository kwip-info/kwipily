import SwiftUI
import KwipMLCore

struct ColumnsConfigurationView<Context: DataNodeEditing>: View {
    @ObservedObject var context: Context
    let node: DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft = try! ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
    @State private var original: [String: JSONValue] = [:]
    @State private var search = ""
    @State private var dragStartIndex: Int?
    @State private var selected: String?
    @State private var exactName = ""
    @State private var error: String?
    @State private var editOrigin = ""
    private var accent: Color { colorScheme == .dark ? ComponentVisualFamily.data.primary : ComponentVisualFamily.data.secondary }
    private var pending: Bool { draft.hasUserChanges(comparedTo: original, schema: schema) }
    private var requiresSchemaReview: Bool { draft.requiresSchemaReview(comparedTo: original, schema: schema) }
    private var schema: [DataColumnSchema] { context.upstreamSchema }
    private var filtered: [ColumnsConfigurationDraft.Row] {
        draft.rows.filter { search.isEmpty || $0.source.localizedCaseInsensitiveContains(search) || $0.outputName.localizedCaseInsensitiveContains(search) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Input schema", systemImage: "tablecells")
                Spacer()
                Text(schema.isEmpty ? "Unresolved" : "\(schema.count) columns")
            }.font(.caption).foregroundStyle(accent).padding(9)
                .background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            grid
            policy("Cast failures", value: $draft.castFailure, choices: [("error", "Stop run"), ("null", "Use missing value")])
            Toggle("Trim text before casting", isOn: $draft.trimText).toggleStyle(.checkbox)
            policy("Missing values", value: $draft.missingPolicy, choices: [("preserve", "Keep missing values"), ("error", "Stop run")])
            policy("New source columns", value: $draft.newColumns, choices: [("keep", "Keep unchanged"), ("drop", "Exclude"), ("error", "Require grid choices")])
            if schema.isEmpty {
                HStack {
                    TextField("Exact source column", text: $exactName).flatField()
                    Button("Add") { draft.add(source: exactName); exactName = "" }.buttonStyle(FlatButtonStyle())
                        .disabled(exactName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            HStack {
                Button("Apply", action: apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10)).disabled(!pending && !requiresSchemaReview)
                Button("Revert", action: reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer()
                if pending { Image(systemName: "pencil.circle").foregroundStyle(accent).help("Unapplied grid changes").accessibilityLabel("Unapplied Columns changes") }
            }
            if requiresSchemaReview {
                Text("Review new source columns, then Apply grid choices.").font(.caption).foregroundStyle(accent)
            }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
        }.font(.caption).tint(accent)
            .onAppear(perform: reload)
            .onChange(of: node.id) { _, _ in reload() }
            .onChange(of: node.config) { _, next in
                if !pending { reload() }
                else if next != original { error = "Saved configuration changed. Revert to load it; your draft is preserved." }
            }
            .onChange(of: schema) { _, next in draft.reconcile(schema: next) }
            .onChange(of: draft.configuration) { _, _ in publishPending() }
            .onDisappear { context.setNumericPending(origin: editOrigin, key: "columnsGrid", pending: false) }
    }
    private var grid: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Columns").fontWeight(.semibold)
                Spacer(minLength: 8)
                TextField("Find column", text: $search).flatField().frame(maxWidth: 160).accessibilityLabel("Find Columns grid row")
            }.padding(7).background(MLTheme.panel)
            ThemedScrollView(.horizontal) {
                VStack(spacing: 0) {
                    HStack(spacing: 0) {
                        heading("", width: 20); heading("Keep", width: 30)
                        heading("Source", width: 82); heading("Output name", width: 98); heading("Type", width: 82)
                    }.padding(.horizontal, 4).background(MLTheme.panel)
                    ThemedScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(filtered) { row in gridRow(row) }
                            if filtered.isEmpty { Text(search.isEmpty ? "No columns available" : "No matching columns").foregroundStyle(MLTheme.muted).padding(12) }
                        }
                    }.frame(height: min(266, max(80, CGFloat(filtered.count) * 40)))
                }.frame(width: 320)
            }
            HStack(spacing: 6) {
                Button("Select all") { draft.setAllIncluded(true) }
                Button("Clear") { draft.setAllIncluded(false) }
                Spacer(minLength: 4)
                moveButton(-1, symbol: "arrow.up", title: "Move selected column earlier")
                moveButton(1, symbol: "arrow.down", title: "Move selected column later")
                Text("\(draft.keptCount) / \(draft.rows.count) kept").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }.buttonStyle(FlatButtonStyle()).font(.system(size: 10)).padding(6).background(MLTheme.panel)
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func heading(_ text: String, width: CGFloat) -> some View {
        Text(text).font(.system(size: 9, weight: .semibold)).frame(width: width, height: 26, alignment: .leading)
    }
    private func gridRow(_ row: ColumnsConfigurationDraft.Row) -> some View {
        let missing = !schema.isEmpty && !schema.contains { $0.name == row.source }
        return HStack(spacing: 0) {
            Image(systemName: "line.3.horizontal").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                .frame(width: 20, height: 36).contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 4, coordinateSpace: .global)
                    .onChanged { value in
                        guard context.isCurrent, !context.isBusy else { return }
                        if dragStartIndex == nil { dragStartIndex = draft.rows.firstIndex { $0.source == row.source } }
                        guard let origin = dragStartIndex else { return }
                        selected = row.source
                        let target = min(draft.rows.count - 1, max(0, origin + Int((value.translation.height / 40).rounded())))
                        draft.move(row.source, to: target)
                    }
                    .onEnded { _ in dragStartIndex = nil })
                .accessibilityLabel("Drag \(row.source) to reorder")
            Toggle("Keep \(row.source)", isOn: Binding(get: { row.included }, set: { value in draft.update(row.source) { $0.included = value } }))
                .labelsHidden().toggleStyle(.checkbox).frame(width: 30)
            HStack(spacing: 2) {
                Text(row.source).lineLimit(1).truncationMode(.middle)
                if missing {
                    Button { draft.removeMissing(row.source, schema: schema) } label: { Image(systemName: "xmark.circle").foregroundStyle(MLTheme.rose) }
                        .buttonStyle(.borderless).help("Remove missing source column").accessibilityLabel("Remove missing column \(row.source)")
                }
            }.frame(width: 78, alignment: .leading).padding(.trailing, 4)
                .foregroundStyle(missing ? MLTheme.rose : row.included ? Color.primary : MLTheme.muted)
                .help("\(row.source) · \(schema.first { $0.name == row.source }?.dtype ?? "unresolved")")
            TextField("Output name", text: Binding(get: { row.outputName }, set: { value in draft.update(row.source) { $0.outputName = value } }))
                .flatField().frame(width: 94).padding(.trailing, 4).disabled(!row.included)
                .accessibilityLabel("Output name for \(row.source)")
            FlatChoiceControl(title: "Type for \(row.source)", selection: row.targetType,
                options: typeOptions(for: row), compact: true, accent: accent) { value in
                draft.update(row.source) { $0.targetType = value }
            }.frame(width: 82).disabled(!row.included)
        }.font(.system(size: 10)).padding(.horizontal, 4).frame(height: 40)
            .background(selected == row.source ? accent.opacity(0.12) : row.included ? MLTheme.background : MLTheme.panel)
            .overlay(alignment: .bottom) { Rectangle().fill(MLTheme.rule).frame(height: 1) }
            .contentShape(Rectangle()).simultaneousGesture(TapGesture().onEnded { selected = row.source })

    }
    private func typeOptions(for row: ColumnsConfigurationDraft.Row) -> [FlatChoiceOption] {
        let dtype = schema.first { $0.name == row.source }?.dtype
        return [("keep", "Keep"), ("float64", "Float64"), ("int64", "Int64"), ("string", "Text"), ("bool", "Boolean"), ("image", "Image")].map { id, title in
            .init(id: id, title: title, isEnabled: id == "keep" || dtype == nil || (id == "image" ? dtype == "image" : dtype != "image"))
        }
    }
    private func moveButton(_ delta: Int, symbol: String, title: String) -> some View {
        let index = draft.rows.firstIndex { $0.source == selected }
        return Button { if let selected { draft.move(selected, by: delta) } } label: { Image(systemName: symbol) }
            .disabled(index == nil || !draft.rows.indices.contains((index ?? 0) + delta)).accessibilityLabel(title)
    }
    private func policy(_ title: String, value: Binding<String>, choices: [(String, String)]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
            FlatChoiceControl(title: title, selection: value.wrappedValue, options: choices.map { .init(id: $0.0, title: $0.1) }) { value.wrappedValue = $0 }
        }
    }
    private func publishPending() { context.setNumericPending(origin: editOrigin, key: "columnsGrid", pending: pending) }
    private func reload() {
        context.setNumericPending(origin: editOrigin, key: "columnsGrid", pending: false)
        do {
            original = node.config
            draft = try ColumnsConfigurationDraft(configuration: node.config)
            draft.reconcile(schema: schema)
            editOrigin = context.numericOrigin(key: "columnsGrid"); error = nil; selected = nil; dragStartIndex = nil
            publishPending()
        } catch { self.error = error.localizedDescription }
    }
    private func apply() {
        do {
            let config = try draft.validatedConfiguration(schema: schema)
            if context.applyTableConfiguration(config, expected: original) {
                original = config; error = nil; publishPending()
            } else { error = context.statusMessage }
        } catch { self.error = error.localizedDescription }
    }
}
