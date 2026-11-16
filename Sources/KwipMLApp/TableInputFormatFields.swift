import SwiftUI
import KwipMLCore

/// Ordinary reader controls shared by local Table Input formats; drafts are owned by the inspector.
struct TableInputFormatFields: View {
    let format: TableInputConfiguration.Format
    @Binding var options: [String: JSONValue]
    private var fields: [TableInputConfiguration.Field] {
        let fields = TableInputConfiguration.fields(for: format, options: options).filter { !(format == .excel && options["has_header"] == .bool(false) && $0.key == "read_options.header_row") }
        if format == .excel, let engine = options["engine"], engine != .string("calamine") { return fields.filter { $0.key != "table_name" } }
        if format == .parquet, options["use_pyarrow"] == .bool(true) {
            return fields.filter { !["parallel", "use_statistics", "low_memory", "missing_columns", "extra_columns", "schema"].contains($0.key) && !$0.key.hasPrefix("cast_options.") }
        }
        if format == .parquet { return fields.filter { $0.key != "memory_map" } }
        if format == .database, options["backend"] == nil || options["backend"] == .string("sqlite") {
            return fields.filter { !["host", "port", "database", "tlsMode"].contains($0.key) }
        }
        return fields
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(fields.filter { !$0.advanced }) { field in readerField(field) }
            if fields.contains(where: \.advanced) {
                InspectorSection("Detailed reading", symbol: "slider.horizontal.3", identifier: "table-input.\(format.rawValue).detailed", initiallyExpanded: false) {
                    ForEach(fields.filter(\.advanced)) { field in readerField(field) }
                }
            }
        }
    }
    private func value(_ field: TableInputConfiguration.Field) -> Binding<JSONValue> {
        Binding(get: { TableInputConfiguration.value(for: field.key, in: options) ?? field.defaultValue }, set: { TableInputConfiguration.set($0, for: field.key, in: &options) })
    }
    private func text(_ field: TableInputConfiguration.Field) -> Binding<String> {
        let value = value(field)
        return Binding(get: {
            switch value.wrappedValue { case .string(let text): return text; case .integer(let count): return String(count); case .null: return ""; default: return "" }
        }, set: { text in
            if case .integer(_, _, let optional) = field.kind {
                value.wrappedValue = text.isEmpty && optional ? .null : Int64(text).map(JSONValue.integer) ?? .string(text)
            } else {
                if ["sheet_name", "table_name"].contains(field.key), !text.isEmpty {
                    options["sheet_id"] = .null
                    options[field.key == "sheet_name" ? "table_name" : "sheet_name"] = .null
                    if options["engine"] == nil || options["engine"] == .string("calamine") {
                        TableInputConfiguration.set(field.key == "table_name" ? .null : .integer(0), for: "read_options.header_row", in: &options)
                    }
                }
                value.wrappedValue = text.isEmpty && field.defaultValue == .null ? .null : .string(text)
            }
        })
    }
    private func optionTitle(_ value: String, field: TableInputConfiguration.Field) -> String {
        if field.key == "missingColumns", value == "insert" { return "Insert missing fields" }
        if field.key == "eolChar" { return value == "\n" ? "LF / CRLF" : "CR" }
        if field.key == "parallel" { return ["auto": "Automatic", "columns": "Columns", "row_groups": "Row groups", "none": "Serial", "prefiltered": "Prefiltered"][value] ?? value }
        return ["": "Automatic", "raise": "Report error", "insert": "Insert null columns", "forbid": "Forbid casts", "upcast": "Allow wider integers", "allow-float": "Allow integer-to-float", "allow": "Allow", "coerce": "Coerce", "strict": "Strict", "text": "Plain text", "markdown": "Keep Markdown", "html": "Keep HTML", "newline": "Keep line breaks", "space": "Use spaces", "label": "Link label", "url": "Link address", "both": "Label and address", "reject": "Report merged cells", "expand": "Expand merged cells", "verifyFull": "Verify TLS", "disabledLoopback": "Unencrypted loopback only", "iso8601": "ISO 8601 text", "error": "Report error", "base64": "Base64 text", "hex": "Hexadecimal text", "sqlite": "SQLite", "postgresql": "PostgreSQL", "mysql": "MySQL"][value] ?? value
    }
    @ViewBuilder private func readerField(_ field: TableInputConfiguration.Field) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            switch field.kind {
            case .floatCast:
                Text(field.title).font(.caption)
                FlatChoiceControl(title: field.title, selection: { if case .array(let choices) = value(field).wrappedValue { if choices.count == 1, case .string(let choice) = choices[0] { return choice }; return "both" }; return text(field).wrappedValue }(), options: [.init(id: "forbid", title: "Forbid casts"), .init(id: "upcast", title: "Allow wider floats"), .init(id: "downcast", title: "Allow narrower floats"), .init(id: "both", title: "Allow wider and narrower floats")]) { selected in
                    value(field).wrappedValue = selected == "both" ? .array([.string("downcast"), .string("upcast")]) : .string(selected)
                }
            case .character:
                Text(field.title).font(.caption)
                HStack {
                    FlatChoiceControl(title: field.title, selection: ["\n", "\r"].contains(text(field).wrappedValue) ? text(field).wrappedValue : "__custom", options: [.init(id: "\n", title: "LF / CRLF"), .init(id: "\r", title: "CR"), .init(id: "__custom", title: "Custom")]) { selected in value(field).wrappedValue = .string(selected == "__custom" ? "~" : selected) }
                    if !["\n", "\r"].contains(text(field).wrappedValue) { TextField("Byte", text: text(field)).flatField().frame(width: 48).accessibilityLabel(field.title) }
                }
            case .integerChoice(let choices):
                Text(field.title).font(.caption)
                FlatChoiceControl(title: field.title, selection: text(field).wrappedValue, options: choices.map { .init(id: String($0), title: [0: "Minimal quoting", 1: "Quote all fields", 2: "Quote nonnumeric fields", 3: "No quoting"][Int($0)] ?? String($0)) }) { if let choice = Int64($0) { value(field).wrappedValue = .integer(choice) } }
            case .listOrMap:
                TableInputMarkerSelection(title: field.title, value: value(field))
            case .toggle:
                Toggle(field.title, isOn: Binding(get: { value(field).wrappedValue == .bool(true) }, set: { selected in if field.key == "has_header", options["engine"] == nil || options["engine"] == .string("calamine") {
                    let named: Bool = { if case .string(let name) = options["table_name"] { return !name.isEmpty }; return false }()
                    TableInputConfiguration.set(!selected || named ? .null : .integer(0), for: "read_options.header_row", in: &options)
                }
                if field.key == "use_pyarrow" {
                    if !selected { options.removeValue(forKey: "pyarrow_options"); options["memory_map"] = .bool(true) }
                    else { options["schema"] = .object([:]); options["parallel"] = .string("auto"); options["use_statistics"] = .bool(true); options["low_memory"] = .bool(false); options["missing_columns"] = .string("raise"); options["extra_columns"] = .string("raise"); options["cast_options"] = .object(["integer_cast": .string("forbid"), "float_cast": .string("forbid"), "categorical_to_string": .string("forbid")]) }
                }; value(field).wrappedValue = .bool(selected) }))
                    .font(.caption).toggleStyle(.checkbox)
            case .choice(let choices):
                Text(field.title).font(.caption)
                FlatChoiceControl(title: field.title, selection: text(field).wrappedValue, options: choices.map { .init(id: $0, title: optionTitle($0, field: field)) }) { selected in
                    if field.key == "engine" {
                        options.removeValue(forKey: "engine_options"); options.removeValue(forKey: "read_options")
                        if selected != "calamine" { options["table_name"] = .null }
                    }
                    if field.key == "backend" {
                        options["port"] = .integer(selected == "mysql" ? 3306 : 5432)
                        options["schema"] = selected == "postgresql" ? .string("public") : selected == "sqlite" ? .string("main") : options["database"] ?? .string("")
                    }
                    value(field).wrappedValue = .string(selected)
                }
            case .text(_, let multiline):
                Text(field.title).font(.caption)
                if multiline {
                    TextField(field.title, text: text(field), axis: .vertical).lineLimit(3...8).flatField().accessibilityLabel(field.title)
                } else { TextField(field.title, text: text(field)).flatField().accessibilityLabel(field.title) }
            case .integer(_, _, let optional):
                Text(field.title).font(.caption)
                TextField(optional ? "Automatic / all" : field.title, text: text(field)).flatField().accessibilityLabel(field.title)
            case .stringList, .columns:
                TableInputStringListField(title: field.title, columns: field.kind == .columns, value: value(field))
            case .columnsOrRange:
                TableInputColumnSelection(title: field.title, value: value(field))
            case .integerOrList:
                TableInputIntegerSelection(title: field.title, value: value(field))
            case .choiceOrMap(let choices):
                TableInputTypeSelection(title: field.title, choices: choices, value: value(field))
            case .stringMap, .nullMarkerMap, .schema:
                TableInputDictionaryField(title: field.title, schema: field.kind == .schema, value: value(field))
            }
        }.help(field.help)
    }
}

private struct TableInputStringListField: View {
    let title: String
    let columns: Bool
    var numeric: Bool = false
    @Binding var value: JSONValue
    @State private var indexMode = false
    private var indexes: Bool { numeric || indexMode }
    private var strings: [String] { if case .array(let values) = value { return values.map { switch $0 { case .string(let text): return text; case .integer(let number): return String(number); default: return "" } } }; return [] }
    private func publish(_ strings: [String], indexes: Bool? = nil) { value = .array(strings.map { text in (indexes ?? self.indexes) ? Int64(text).map(JSONValue.integer) ?? .string(text) : .string(text) }) }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            if columns {
                FlatChoiceControl(title: "Select columns by", selection: indexes ? "index" : "name", options: [.init(id: "name", title: "Column names"), .init(id: "index", title: "Column indexes")]) { mode in indexMode = mode == "index"; publish(strings, indexes: indexMode) }
            }
            ForEach(strings.indices, id: \.self) { index in
                HStack {
                    TextField("Column", text: Binding(get: { strings.indices.contains(index) ? strings[index] : "" }, set: { text in var next = strings; guard next.indices.contains(index) else { return }; next[index] = text; publish(next) }))
                        .flatField().accessibilityLabel("\(title) \(index + 1)")
                    Button { var next = strings; next.remove(at: index); publish(next) } label: { Image(systemName: "minus") }.accessibilityLabel("Remove \(title.lowercased()) \(index + 1)")
                }
            }
            Button("Add item") { publish(strings + [indexes ? "0" : ""]) }.font(.caption).disabled(strings.count >= 128)
        }.onAppear {
            if case .array(let values) = value, !values.isEmpty { indexMode = values.allSatisfy { if case .integer = $0 { return true }; return false } }
        }
    }
}

private struct TableInputDictionaryField: View {
    struct Entry: Identifiable { let id = UUID(); var name: String; var value: String }
    let title: String
    let schema: Bool
    var choices: [String]? = nil
    @Binding var value: JSONValue
    @State private var entries: [Entry] = []
    @State private var published: JSONValue = .null
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            ForEach($entries) { $entry in
                HStack {
                    TextField(schema ? "Column" : "Name", text: $entry.name).flatField().accessibilityLabel("\(title) name")
                    if schema || choices != nil {
                        FlatChoiceControl(title: "Column type", selection: entry.value, options: (choices ?? TableInputConfiguration.scalarTypes).map { .init(id: $0, title: $0) }) { entry.value = $0 }
                    } else { TextField("Value", text: $entry.value).flatField().accessibilityLabel("\(title) value") }
                    Button { entries.removeAll { $0.id == entry.id } } label: { Image(systemName: "minus") }.accessibilityLabel("Remove \(title.lowercased()) entry")
                }
            }
            Button("Add \(schema ? "column" : "entry")") { entries.append(Entry(name: "", value: choices?.first ?? (schema ? "String" : ""))) }.font(.caption).disabled(entries.count >= 128)
        }
        .onAppear(perform: reload)
        .onChange(of: entries.map { [$0.name, $0.value] }) { _, _ in publish() }
        .onChange(of: value) { _, next in if next != published { reload() } }
    }
    private func reload() {
        guard case .object(let object) = value else { return }
        entries = object.keys.sorted().map { key in Entry(name: key, value: { if case .string(let text) = object[key] { return text }; return "" }()) }
        published = value
    }
    private func publish() {
        // Keep an invalid draft invalid instead of silently merging duplicate column names.
        if Set(entries.map(\.name)).count != entries.count {
            published = .array(entries.map { .object(["name": .string($0.name), "value": .string($0.value)]) })
        } else { published = .object(Dictionary(uniqueKeysWithValues: entries.map { ($0.name, .string($0.value)) })) }
        value = published
    }
}

private struct TableInputColumnSelection: View {
    let title: String
    @Binding var value: JSONValue
    private var isRange: Bool { if case .string = value { return true }; return false }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: isRange ? "range" : "columns", options: [.init(id: "columns", title: "Named / indexed columns"), .init(id: "range", title: "Excel column range")]) { value = $0 == "range" ? .string("") : .array([]) }
            if isRange { TextField("A:C", text: Binding(get: { if case .string(let range) = value { return range }; return "" }, set: { value = .string($0) })).flatField().accessibilityLabel("\(title) range") }
            else { TableInputStringListField(title: "Columns", columns: true, value: $value) }
        }
    }
}
private struct TableInputIntegerSelection: View {
    let title: String
    @Binding var value: JSONValue
    private var isList: Bool { if case .array = value { return true }; return false }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: isList ? "indexes" : "count", options: [.init(id: "count", title: "Initial row count"), .init(id: "indexes", title: "Selected row indexes")]) { value = $0 == "indexes" ? .array([]) : .integer(0) }
            if isList { TableInputStringListField(title: "Row indexes", columns: false, numeric: true, value: $value) }
            else { TextField("0", text: Binding(get: { switch value { case .integer(let count): return String(count); case .string(let invalid): return invalid; default: return "" } }, set: { value = Int64($0).map(JSONValue.integer) ?? .string($0) })).flatField().accessibilityLabel("\(title) count") }
        }
    }
}
private struct TableInputTypeSelection: View {
    let title: String
    let choices: [String]
    @Binding var value: JSONValue
    @State private var selection = "__default"
    private func reload() { switch value { case .object: selection = "__columns"; case .string(let text): selection = text; case .null: selection = "__default"; default: break } }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: selection, options: [.init(id: "__default", title: "Automatic"), .init(id: "__columns", title: "Per column")] + choices.map { .init(id: $0, title: $0) }) { selected in
                selection = selected
                value = selected == "__columns" ? .object([:]) : selected == "__default" ? .null : .string(selected)
            }
            if selection == "__columns" { TableInputDictionaryField(title: "Column types", schema: false, choices: choices, value: $value) }
        }.onAppear(perform: reload).onChange(of: value) { _, _ in reload() }
    }
}

private struct TableInputMarkerSelection: View {
    let title: String
    @Binding var value: JSONValue
    @State private var perColumn = false
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FlatChoiceControl(title: title, selection: perColumn ? "columns" : "global", options: [.init(id: "global", title: "Global markers"), .init(id: "columns", title: "Per-column markers")]) { selected in
                perColumn = selected == "columns"; value = perColumn ? .object([:]) : .array([])
            }
            if perColumn { TableInputDictionaryField(title: "Column markers", schema: false, value: $value) }
            else { TableInputStringListField(title: "Markers", columns: false, value: $value) }
        }.onAppear { if case .object = value { perColumn = true } }
    }
}

/// Credential text never enters source configuration, code exports, History or worker input.
struct TableInputDatabaseCredentialsView: View {
    let projectID: String
    let nodeID: String
    let options: [String: JSONValue]
    let canConnect: Bool
    let isCurrent: () -> Bool
    let connect: () -> Void
    @State private var username = ""
    @State private var password = ""
    @State private var hasSavedCredentials = false
    @State private var credentialError: String?
    var body: some View {
        InspectorSection("Connection credentials", symbol: "key", identifier: "table-input.database.credentials", initiallyExpanded: true) {
            VStack(alignment: .leading, spacing: 7) {
                Label(!canConnect ? "Apply connection settings first" : hasSavedCredentials ? "Credentials saved in Keychain" : "Credentials required", systemImage: hasSavedCredentials ? "key.fill" : "key")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                TextField("Username", text: $username).textContentType(.username).flatField().accessibilityLabel("Database username")
                SecureField("Password", text: $password).textContentType(.password).flatField().accessibilityLabel("Database password")
                HStack {
                    Button("Save & connect", action: saveAndConnect).buttonStyle(FlatButtonStyle()).disabled(!canConnect || username.isEmpty)
                    if hasSavedCredentials { Button("Connect") { if canConnect && isCurrent() { connect() } }.buttonStyle(FlatButtonStyle()).disabled(!canConnect) }
                }
                if let credentialError { Text(credentialError).font(.caption).foregroundStyle(MLTheme.rose).fixedSize(horizontal: false, vertical: true) }
            }
        }
        .help("Credentials stay in this Mac's private Keychain. Apply the connection settings before connecting.")
        .onAppear { if canConnect { loadSavedStatus() } }
        .onChange(of: canConnect) { _, allowed in
            if allowed { loadSavedStatus() } else { password = ""; hasSavedCredentials = false }
        }
        .onDisappear { username = ""; password = "" }
    }
    private func connectionIdentity() throws -> String { try DatabaseSourceSettings.connectionIdentity(TableInputConfiguration.options(in: options)) }
    private func loadSavedStatus() {
        guard isCurrent() else { return }
        do {
            let saved = try DatabaseSourceKeychain().read(projectID: projectID, nodeID: nodeID, connectionID: connectionIdentity())
            username = saved?.username ?? ""; password = ""; hasSavedCredentials = saved != nil; credentialError = nil
        } catch { hasSavedCredentials = false; credentialError = "Could not read this connection's private Keychain credentials." }
    }
    private func saveAndConnect() {
        guard canConnect, isCurrent() else { return }
        do {
            try DatabaseSourceKeychain().write(.init(username: username, password: password), projectID: projectID, nodeID: nodeID, connectionID: connectionIdentity())
            password = ""; hasSavedCredentials = true; credentialError = nil
            connect()
        } catch { password = ""; credentialError = "Could not save credentials in this Mac's private Keychain." }
    }
}
