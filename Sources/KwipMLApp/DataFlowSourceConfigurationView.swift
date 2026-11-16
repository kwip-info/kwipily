import SwiftUI
import KwipMLCore

/// Draft-only editing; file reads occur only through the supervised source service.
struct DataFlowSourceConfigurationView<Context: DataNodeEditing>: View {
    @Environment(\.colorScheme) private var colorScheme
    private var toolColor: Color { colorScheme == .dark ? ComponentVisualFamily.data.primary : ComponentVisualFamily.data.secondary }
    @ObservedObject var context: Context
    let node: DataFlowNode
    @State private var formatDrafts: [String: [String: JSONValue]] = [:]
    private var format: TableInputConfiguration.Format { (try? TableInputConfiguration.format(of: draft)) ?? .csv }
    private var isRemoteDatabase: Bool { format == .database && (try? TableInputConfiguration.options(in: draft))?["backend"] != .string("sqlite") }
    private var readerOptions: Binding<[String: JSONValue]> {
        Binding(get: { (try? TableInputConfiguration.options(in: draft)) ?? [:] }, set: { draft["options"] = .object($0) })
    }
    @State private var original: [String: JSONValue] = [:]
    @State private var draft = CSVParsingConfiguration.legacyDefaults
    @State private var skipRecords = "0"
    @State private var skipLines = "0"
    @State private var skipRowsAfterHeader = "0"
    private var isPolars: Bool { CSVParsingConfiguration.isPolars(draft) }
    @State private var markers: [String] = []
    @State private var error: String?
    @State private var loaded = false
    private let draftKey = "csvParsingDraft"
    private var configuration: [String: JSONValue] {
        guard format == .csv else { return draft }
        var result = draft
        result["skipRecords"] = Int64(skipRecords).map(JSONValue.integer) ?? .string(skipRecords)
        if isPolars {
            result["skipLines"] = Int64(skipLines).map(JSONValue.integer) ?? .string(skipLines)
            result["skipRowsAfterHeader"] = Int64(skipRowsAfterHeader).map(JSONValue.integer) ?? .string(skipRowsAfterHeader)
        }
        result["nullMarkers"] = .array(markers.map(JSONValue.string))
        return result
    }
    private var remoteSourceTitle: String {
        guard case .object(let options) = configuration["options"] else { return "Database connection" }
        let table = options["table"]?.stringValue ?? ""
        return table.isEmpty ? "Database connection" : (options["backend"]?.stringValue ?? "Database").capitalized + " · " + table
    }
    private var changed: Bool { loaded && configuration != (try? TableInputConfiguration.validate(original)) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Format").font(.caption)
                FlatChoiceControl(title: "Table Input format", selection: format.rawValue, options: TableInputConfiguration.Format.allCases.map { .init(id: $0.rawValue, title: $0.title) }) { selection in
                    guard let selected = TableInputConfiguration.Format(rawValue: selection) else { return }
                    switchFormat(to: selected)
                }
            }.foregroundStyle(toolColor).padding(8).background(toolColor.opacity(0.10))
            VStack(alignment: .leading, spacing: 4) {
                Label(format == .api ? "API · captured table":isRemoteDatabase ? remoteSourceTitle : (context.sourceDisplayName ?? "Choose a local file"), systemImage: "doc.text").font(.caption.weight(.semibold)).textSelection(.enabled)
                if let status = context.sourceInspectionStatus {
                    Label(status.title, systemImage: status.symbolName)
                        .font(.caption2).foregroundStyle(status.isProblem ? MLTheme.rose : toolColor)
                } else {
                    Label(context.sourceIsLinked ? "Source linked" : format == .api ? "No snapshot captured":isRemoteDatabase ? "Not connected" : "No file selected", systemImage: context.sourceIsLinked ? "link" : "doc.badge.plus")
                        .font(.caption2).foregroundStyle(MLTheme.muted)
                }
            }.padding(9).frame(maxWidth: .infinity, alignment: .leading)
                .background(toolColor.opacity(0.06)).overlay(Rectangle().stroke(toolColor.opacity(0.45), lineWidth: 1))
            HStack {
                if !isRemoteDatabase && format != .api {
                    Button("Choose file…") { context.chooseSource() }
                        .help("Replace the local source. Unapplied parsing edits are preserved; preview uses saved settings.")
                }
                Button("Refresh preview") { context.refreshSource() }.disabled(changed || !context.sourceIsLinked)
            }.buttonStyle(FlatButtonStyle())
            if format == .csv {
                csvFields
                if isPolars {
                    TableInputFormatFields(format: .csv, options: Binding(get: { draft }, set: { next in draft = next }))
                }
            }
            else if format == .api,let store=context.databaseOutputs {
                APITableConfigurationView(context:context,store:store,options:readerOptions,output:false,inspect:{if changed {error="Apply your API settings before fetching a snapshot."} else if !context.connectDatabaseSource(){error=context.statusMessage}})
            } else {
                TableInputFormatFields(format: format, options: readerOptions)
                if isRemoteDatabase,let store=context.databaseOutputs {
                    DatabaseConnectionProfilePicker(store:store,address:.init(domain:.data,localID:node.id),config:readerOptions.wrappedValue,allowComponentLogin:true) {p in var values=readerOptions.wrappedValue;for (k,v) in p.endpoint {values[k]=v};readerOptions.wrappedValue=values}
                }
                if isRemoteDatabase,context.databaseOutputs?.profile(for:.init(domain:.data,localID:node.id)) != nil {
                    Button("Connect") {if !context.connectDatabaseSource(){error=context.statusMessage}}.buttonStyle(FlatButtonStyle()).disabled(changed)
                } else if isRemoteDatabase, let projectID = context.sourceDatabaseProjectID {
                    TableInputDatabaseCredentialsView(projectID: projectID, nodeID: node.id, options: configuration, canConnect: !changed && context.isCurrent && !context.isBusy, isCurrent: {
                        context.isCurrent && !context.isBusy && context.node?.config == original && !changed
                    }) {
                        if !context.connectDatabaseSource() { error = context.statusMessage }
                    }
                }
            }
            HStack {
                Button(format == .api ? "Apply":"Apply & preview", action: apply).buttonStyle(FlatButtonStyle()).foregroundStyle(toolColor).background(toolColor.opacity(0.12)).disabled(!changed && !context.sourceIsLinked)
                Button("Revert") { reload() }.buttonStyle(FlatButtonStyle()).disabled(!changed)
            }
            if changed { Label("Unapplied changes", systemImage: "pencil").font(.caption).foregroundStyle(MLTheme.rose) }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).fixedSize(horizontal: false, vertical: true) }
            if let status = context.sourceInspectionStatus, status.phase == .failed {
                Text(status.csvDiagnostic?.detail ?? status.detail)
                    .font(.caption).foregroundStyle(MLTheme.rose).fixedSize(horizontal: false, vertical: true)
            }
        }
        .tint(toolColor)
        .onAppear { reload() }
        .onChange(of: node.config) { _, next in
            if !changed { reload(clearFormatDrafts: false) } else if next != original { error = "Saved settings changed. Revert to reload before applying." }
        }
        .onChange(of: changed) { _, pending in context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: pending) }
        .onDisappear { context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: false) }
    }
    @ViewBuilder private var csvFields: some View {
            Label("CSV dialect", systemImage: "tablecells").font(.caption.weight(.semibold)).foregroundStyle(toolColor)
            characterField("Delimiter", key: "separator", choices: [("Comma", ","), ("Tab", "\t"), ("Semicolon", ";"), ("Pipe", "|")])
            characterField("Quote character", key: "quoteChar", choices: [("Double quote", "\""), ("Single quote", "'"), ("None", "")])
            if isPolars {
                choice("Encoding", key: "encoding", values: [("utf-8-sig", "UTF-8 · accept BOM"), ("utf8-lossy", "UTF-8 · replace invalid bytes")])
            } else {
                choice("Encoding", key: "encoding", values: [("utf-8-sig", "UTF-8 · accept BOM"), ("utf-8", "UTF-8 · preserve BOM"), ("utf-16", "UTF-16 · BOM required"), ("cp1252", "Windows-1252")])
                choice("Record endings", key: "recordEndings", values: [("auto", "Auto · LF, CRLF or CR"), ("lf", "LF"), ("crlf", "CRLF"), ("cr", "CR")])
                Toggle("Allow newlines inside quoted fields", isOn: flag("allowQuotedNewlines")).font(.caption).toggleStyle(.checkbox)
            }
            choice("Column names", key: "headerMode", values: [("firstRecord", "First record"), ("generated", isPolars ? "Generate column_0, column_1, …" : "Generate column_1, column_2, …")])
            InspectorSection("Detailed parsing", symbol: "text.alignleft", identifier: "csv.detailed", initiallyExpanded: false) {
                if !isPolars {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Quote escaping").font(.caption)
                        HStack {
                            FlatChoiceControl(title: "Quote escaping", selection: text("escapeMode"), options: [("double", "Doubled quote"), ("backslash", "Backslash"), ("none", "None"), ("custom", "Custom")].map { .init(id: $0.0, title: $0.1) }) { draft["escapeMode"] = .string($0) }
                            TextField("Character", text: Binding(get: { text("escapeMode") == "backslash" ? "\\" : text("escapeChar") }, set: { draft["escapeChar"] = .string($0); draft["escapeMode"] = .string("custom") }))
                                .flatField().frame(width: 48).accessibilityLabel("Custom escape character")
                        }
                    }
                    Toggle("Ignore spaces after delimiter", isOn: flag("skipInitialSpace")).font(.caption).toggleStyle(.checkbox)
                }
                Divider()
                Text("Records").font(.caption.weight(.semibold)).foregroundStyle(toolColor)
                Text("Records to skip before header / data").font(.caption)
                TextField("Records to skip", text: $skipRecords).flatField().accessibilityLabel("Records to skip")
                if isPolars {
                    Text("Physical lines to skip before parsing").font(.caption)
                    TextField("Lines to skip", text: $skipLines).flatField().accessibilityLabel("Physical lines to skip")
                    Text("Records to skip after header").font(.caption)
                    TextField("Records after header", text: $skipRowsAfterHeader).flatField().accessibilityLabel("Records to skip after header")
                    Text("Comment prefix · empty disables").font(.caption)
                    TextField("Comment prefix", text: string("commentPrefix")).flatField().accessibilityLabel("Comment prefix")
                } else {
                    Toggle("Skip blank records", isOn: flag("skipBlankRecords")).font(.caption).toggleStyle(.checkbox)
                }
                Divider()
                Text("Empty values").font(.caption.weight(.semibold)).foregroundStyle(toolColor)
                choice("Empty field", key: "emptyField", values: [("text", "Keep empty text"), ("null", "Read as null")])
                Text("Null markers · exact, case-sensitive text").font(.caption)
                ForEach(markers.indices, id: \.self) { index in
                    HStack {
                        TextField("Null marker", text: Binding(get: { markers[index] }, set: { markers[index] = $0 })).flatField().accessibilityLabel("Null marker \(index + 1)")
                        Button { markers.remove(at: index) } label: { Image(systemName: "minus") }.accessibilityLabel("Remove null marker \(index + 1)")
                    }
                }
                Button("Add null marker") { markers.append("") }.disabled(markers.count >= 32).font(.caption)
            }
            if !isPolars {
                Button("Use Polars CSV options") {
                    draft = CSVParsingConfiguration.polarsMigration(configuration)
                    loadSkipCounts()
                }.buttonStyle(FlatButtonStyle())
                    .help("Switch this draft to Polars parsing. Compatible settings are kept; custom escaping, unsupported encodings and legacy record controls reset. Apply & preview saves the change.")
            }
    }
    private func switchFormat(to selected: TableInputConfiguration.Format) {
        guard selected != format else { return }
        formatDrafts[format.rawValue] = configuration
        draft = formatDrafts[selected.rawValue] ?? TableInputConfiguration.defaults(for: selected)
        loadCSVFields(); error = nil
    }
    private func loadCSVFields() {
        loadSkipCounts()
        if case .array(let values) = draft["nullMarkers"] { markers = values.compactMap { if case .string(let text) = $0 { return text }; return nil } } else { markers = [] }
    }
    private func text(_ key: String) -> String { if case .string(let value) = draft[key] { return value }; return "" }
    private func string(_ key: String) -> Binding<String> { Binding(get: { text(key) }, set: { draft[key] = .string($0) }) }
    private func flag(_ key: String) -> Binding<Bool> { Binding(get: { if case .bool(let value) = draft[key] { return value }; return false }, set: { draft[key] = .bool($0) }) }
    private func choice(_ title: String, key: String, values: [(String, String)]) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: text(key), options: values.map { .init(id: $0.0, title: $0.1) }) { draft[key] = .string($0) }
        }
    }
    private func characterField(_ title: String, key: String, choices: [(String, String)]) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            HStack {
                FlatChoiceControl(title: title, selection: choices.contains(where: { $0.1 == text(key) }) ? text(key) : "__custom", options: choices.map { .init(id: $0.1, title: $0.0) } + [.init(id: "__custom", title: "Custom")]) { value in draft[key] = .string(value == "__custom" ? " " : value) }
                TextField("Character", text: string(key)).flatField().frame(width: 48).accessibilityLabel("Custom \(title.lowercased())")
            }
        }
    }
    private func reload(clearFormatDrafts: Bool = true) {
        original = node.config
        do { draft = try TableInputConfiguration.validate(node.config); error = nil }
        catch { draft = node.config; self.error = error.localizedDescription }
        if clearFormatDrafts { formatDrafts = [:] }
        loadCSVFields()
        loaded = true
        context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: false)
    }
    private func loadSkipCounts() {
        func count(_ key: String) -> String {
            switch draft[key] { case .integer(let value): return String(value); case .string(let invalid): return invalid; default: return "0" }
        }
        skipRecords = count("skipRecords"); skipLines = count("skipLines"); skipRowsAfterHeader = count("skipRowsAfterHeader")
    }
    private func apply() {
        do {
            let checked = try TableInputConfiguration.validate(configuration)
            _ = try TableInputConfiguration.transportJSON(checked)
            guard context.configure(checked, expected: original) else { error = context.statusMessage; return }
            original = checked; draft = checked; formatDrafts[format.rawValue] = checked; error = nil
            context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: false)
            if !isRemoteDatabase && format != .api && context.sourceIsLinked && !context.refreshSource() { error = context.statusMessage }
        } catch { self.error = error.localizedDescription }
    }
}

struct SourceInspectionStatusView: View {
    let status: SourceInspectionStatus?
    var body: some View {
        if let status {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: status.symbolName).foregroundStyle(status.isProblem ? MLTheme.rose : MLTheme.muted)
                VStack(alignment: .leading, spacing: 5) {
                    Text(status.title).font(.caption.weight(.semibold))
                    if let diagnostic = status.csvDiagnostic, diagnostic.record > 0 {
                        HStack(spacing: 14) {
                            Text("Record \(diagnostic.record)")
                            if diagnostic.line > 0 { Text("Line \(diagnostic.line)") }
                        }.font(.caption.monospacedDigit()).foregroundStyle(MLTheme.muted)
                    }
                    Text(status.csvDiagnostic?.message ?? status.detail).font(.caption).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    if let action = status.action { Text(action).font(.caption2).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true) }
                }
            }.padding(9).frame(maxWidth: .infinity, alignment: .leading).background(MLTheme.background)
                .overlay(Rectangle().stroke(status.isProblem ? MLTheme.rose : MLTheme.rule))
        }
    }
}
