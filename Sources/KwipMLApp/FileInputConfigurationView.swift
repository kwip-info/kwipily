import SwiftUI
import KwipMLCore

/// Reader drafts remain local until a canonical apply; picking uses the visible source mode.
struct FileInputConfigurationView<Context: DataNodeEditing>: View {
    @Environment(\.colorScheme) private var colorScheme
    @ObservedObject var context: Context
    let node: DataFlowNode
    @State private var original: [String: JSONValue] = [:]
    @State private var draft = FileInputConfiguration.newDefaults
    @State private var reader = "text"
    @State private var customExtension = ""
    @State private var error: String?
    @State private var loaded = false
    private let draftKey = "fileInputDraft"
    private var tint: Color { colorScheme == .dark ? ComponentVisualFamily.data.primary : ComponentVisualFamily.data.secondary }
    private var changed: Bool { loaded && draft != (try? FileInputConfiguration.validate(original)) }
    private var directory: Bool { text("sourceMode") == "directory" }
    private var extensions: [String] { strings(draft["extensions"]) }
    private var options: [String: JSONValue] {
        guard case .object(let readers) = draft["readers"], case .object(let values) = readers[reader] else { return [:] }
        return values
    }
    private var fields: [FileInputOption] { FileInputConfiguration.readerDefinitions[reader]?.fields ?? [] }
    private var detailedFields: [FileInputOption] { fields.filter { !primary($0.key) && visible($0.key) } }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            choice("Source", key: "sourceMode", choices: [("file", "Single file"), ("directory", "Directory")])
                .foregroundStyle(tint).padding(8).background(tint.opacity(0.10))
            VStack(alignment: .leading, spacing: 4) {
                Label(context.sourceDisplayName ?? (directory ? "Choose a local folder" : "Choose a local file"), systemImage: directory ? "folder" : "doc.text")
                    .font(.caption.weight(.semibold)).textSelection(.enabled)
                if let status = context.sourceInspectionStatus {
                    Label(status.title, systemImage: status.symbolName).font(.caption2)
                        .foregroundStyle(status.isProblem ? MLTheme.rose : MLTheme.muted)
                } else if context.sourceIsLinked { Label("Source linked", systemImage: "link").font(.caption2).foregroundStyle(MLTheme.muted) }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                .background(tint.opacity(0.06)).overlay(Rectangle().stroke(tint.opacity(0.45)))
            HStack {
                Button(directory ? "Choose folder…" : "Choose file…", action: choose)
                    .help("Choose using the displayed source settings. Valid changes are saved before opening the picker.")
                Button("Refresh preview") { if !context.refreshSource() { error = context.statusMessage } }
                    .disabled(changed || !context.sourceIsLinked)
            }.buttonStyle(FlatButtonStyle())
            if directory {
                Toggle("Include subfolders", isOn: flag("includeSubfolders")).font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint))
                InspectorSection("File types · \(extensions.isEmpty ? "All" : String(extensions.count))", symbol: "doc.on.doc", identifier: "file-input.extensions", initiallyExpanded: false) {
                    Toggle("All file types", isOn: Binding(get: { extensions.isEmpty }, set: { draft["extensions"] = .array(($0 ? [] : FileInputConfiguration.allExtensions).map(JSONValue.string)) }))
                        .font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint))
                    if !extensions.isEmpty {
                        ForEach(FileInputConfiguration.readerOrder, id: \.self) { family in
                            Text(FileInputConfiguration.readerDefinitions[family]?.title ?? family).font(.caption.weight(.semibold)).foregroundStyle(tint)
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), alignment: .leading)], alignment: .leading, spacing: 6) {
                                ForEach(FileInputConfiguration.extensionGroups[family] ?? [], id: \.self) { ext in
                                    Toggle(ext.uppercased(), isOn: Binding(get: { extensions.contains(ext) }, set: { setMember(ext, key: "extensions", included: $0) }))
                                        .font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint))
                                }
                            }
                        }
                        ForEach(extensions.filter { !FileInputConfiguration.allExtensions.contains($0) }, id: \.self) { ext in
                            HStack { Text(ext).font(.caption); Spacer(); Button { setMember(ext, key: "extensions", included: false) } label: { Image(systemName: "xmark") }.accessibilityLabel("Remove file type \(ext)") }
                        }
                        HStack {
                            TextField("Custom extension", text: $customExtension).flatField()
                            Button("Add") {
                                let ext = customExtension.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
                                if !ext.isEmpty { setMember(ext, key: "extensions", included: true); customExtension = "" }
                            }.buttonStyle(FlatButtonStyle())
                        }
                    }
                }
            }
            choice("Read", key: "readMode", choices: [("content", "Metadata and content"), ("metadata", "Metadata only")])
            if text("readMode") == "content" {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Reader settings").font(.caption)
                    FlatChoiceControl(title: "File Input reader settings", selection: reader, options: FileInputConfiguration.readerOrder.map { .init(id: $0, title: FileInputConfiguration.readerDefinitions[$0]?.title ?? $0) }) { reader = $0 }
                }.foregroundStyle(tint).padding(8).background(tint.opacity(0.10))
                ForEach(fields.filter { primary($0.key) && visible($0.key) }) { field in optionField(field) }
                if !detailedFields.isEmpty {
                    InspectorSection("Detailed reading", symbol: "slider.horizontal.3", identifier: "file-input.\(reader).detailed", initiallyExpanded: false) {
                        ForEach(detailedFields) { field in optionField(field) }
                    }
                }
            }
            InspectorSection("Discovery & limits", symbol: "line.3.horizontal.decrease", identifier: "file-input.discovery", initiallyExpanded: false) {
                if directory {
                    Toggle("Include hidden files", isOn: flag("includeHidden")).font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint))
                    stringList("Include patterns", key: "includePatterns")
                    stringList("Exclude patterns", key: "excludePatterns")
                    choice("Symbolic links", key: "symlinks", choices: [("skip", "Skip"), ("error", "Report error")])
                    number("Maximum files", key: "maxFiles")
                    if bool("includeSubfolders") { number("Maximum folder depth", key: "maxDepth") }
                }
                choice("Unreadable content", key: "errorPolicy", choices: [("fail", "Stop with an error"), ("record", "Keep a diagnostic row"), ("skip", "Skip and report")])
                choice("Unrecognized file type", key: "unknownFiles", choices: [("metadata", "Read metadata only"), ("error", "Report error"), ("skip", "Skip and report")])
                FileInputSizeField(title: "Maximum content per file", value: draftValue("maxFileBytes"))
                FileInputSizeField(title: "Maximum collection content", value: draftValue("maxTotalBytes"))
            }
            InspectorSection("Output fields", symbol: "tablecells", identifier: "file-input.output-fields", initiallyExpanded: false) {
                ForEach(FileInputConfiguration.metadataFields, id: \.self) { key in
                    Toggle(metadataTitle(key), isOn: Binding(get: { strings(draft["metadataFields"]).contains(key) }, set: { setMember(key, key: "metadataFields", included: $0) }))
                        .font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint)).disabled(key == "file_id")
                }
            }
            HStack {
                Button("Apply & preview") { _ = apply(refresh: true) }.buttonStyle(FlatButtonStyle()).foregroundStyle(tint).background(tint.opacity(0.12))
                    .disabled(!changed && !context.sourceIsLinked)
                Button("Revert", action: reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
            }
            if changed { Label("Unapplied changes", systemImage: "pencil").font(.caption).foregroundStyle(MLTheme.rose) }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).fixedSize(horizontal: false, vertical: true) }
        }
        .tint(tint)
        .onAppear(perform: reload)
        .onChange(of: context.contextIdentity) { _, _ in reload() }
        .onChange(of: context.sourceDisplayName) { _, name in
            if let family = FileInputReaderSelection.reader(sourceMode: text("sourceMode"), sourceName: name) { reader = family }
        }
        .onChange(of: draft) { _, _ in context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: changed) }
    }
    private func primary(_ key: String) -> Bool {
        switch reader {
        case "text": return ["encoding", "errors", "newline", "units"].contains(key)
        case "pdf": return ["extraction_mode", "pages", "units"].contains(key)
        case "docx", "odt": return ["units", "include_tables"].contains(key)
        case "rtf": return ["encoding", "errors", "units"].contains(key)
        default: return ["frames", "frameStart", "frameEnd", "orientation", "colorMode"].contains(key)
        }
    }
    private func visible(_ key: String) -> Bool {
        if ["frameStart", "frameEnd"].contains(key) { return false }
        if key.hasPrefix("layout_") { return options["extraction_mode"] == .string("layout") }
        if ["orientations", "space_width"].contains(key) { return options["extraction_mode"] == .string("plain") }
        if key == "include_nested_tables" { return options["include_tables"] == .bool(true) }
        if key == "keep_empty_lines" { return options["units"] == .string("line") }
        return true
    }
    @ViewBuilder private func optionField(_ field: FileInputOption) -> some View {
        if reader == "pdf", field.key == "pages" {
            FileInputPageSelectionField(value: optionValue(field.key))
        } else if reader == "pdf", field.key == "orientations" {
            VStack(alignment: .leading, spacing: 6) {
                Text("Text orientations").font(.caption)
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], alignment: .leading, spacing: 8) {
                    ForEach([0, 90, 180, 270], id: \.self) { angle in
                        Toggle("\(angle)°", isOn: Binding(get: {
                            if case .array(let values) = options["orientations"] { return values.contains(.integer(Int64(angle))) }; return false
                        }, set: { enabled in
                            var values: [JSONValue] = []; if case .array(let existing) = options["orientations"] { values = existing }
                            values.removeAll { $0 == .integer(Int64(angle)) }; if enabled { values.append(.integer(Int64(angle))) }; setOption("orientations", .array(values))
                        })).font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint)).accessibilityLabel("Read text rotated \(angle) degrees")
                    }
                }
            }
        } else if field.kind == "boolean" {
            Toggle(field.title, isOn: Binding(get: { options[field.key] == .bool(true) }, set: { setOption(field.key, .bool($0)) })).font(.caption).toggleStyle(FileInputCheckboxStyle(tint: tint))
        } else {
            VStack(alignment: .leading, spacing: 5) {
                Text(field.title).font(.caption)
                if field.kind == "choice" {
                    FlatChoiceControl(title: field.title, selection: options[field.key]?.stringValue ?? "", options: (field.choices ?? []).map { .init(id: $0.value, title: $0.title) }) { setOption(field.key, .string($0)) }
                } else {
                    FileInputDraftEntry(title: field.title, value: optionValue(field.key), parse: { FileInputDraftText.value($0, kind: field.kind) })
                }
            }
            if reader == "image", field.key == "frames", options["frames"] == .string("range") {
                FileInputFrameRangeFields(first: optionValue("frameStart"), last: optionValue("frameEnd"))
            }
        }
    }
    private func optionValue(_ key: String) -> Binding<JSONValue> { .init(get: { options[key] ?? .null }, set: { setOption(key, $0) }) }
    private func draftValue(_ key: String) -> Binding<JSONValue> { .init(get: { draft[key] ?? .null }, set: { draft[key] = $0 }) }
    private func text(_ key: String) -> String { draft[key]?.stringValue ?? "" }
    private func bool(_ key: String) -> Bool { draft[key] == .bool(true) }
    private func flag(_ key: String) -> Binding<Bool> { .init(get: { bool(key) }, set: { draft[key] = .bool($0) }) }
    private func strings(_ value: JSONValue?) -> [String] { if case .array(let values) = value { return values.compactMap(\.stringValue) }; return [] }
    private func setMember(_ item: String, key: String, included: Bool) {
        var items = strings(draft[key]); items.removeAll { $0 == item }; if included { items.append(item) }
        // Empty extension list is explicitly All, never an accidental last-checkbox removal.
        if key == "extensions", items.isEmpty, !included { error = "Keep a file type selected, or choose All file types."; return }
        draft[key] = .array(items.map(JSONValue.string)); error = nil
    }
    private func setOption(_ key: String, _ value: JSONValue) {
        var next = options; next[key] = value
        guard case .object(var readers) = draft["readers"] else { return }
        readers[reader] = .object(next); draft["readers"] = .object(readers)
    }
    private func choice(_ title: String, key: String, choices: [(String, String)]) -> some View {
        VStack(alignment: .leading, spacing: 5) { Text(title).font(.caption); FlatChoiceControl(title: title, selection: text(key), options: choices.map { .init(id: $0.0, title: $0.1) }) { draft[key] = .string($0) } }
    }
    private func number(_ title: String, key: String) -> some View {
        VStack(alignment: .leading, spacing: 5) { Text(title).font(.caption); TextField(title, text: Binding(get: { FileInputDraftText.display(draft[key] ?? .integer(0)) }, set: { draft[key] = FileInputDraftText.value($0, kind: "integer") })).flatField().accessibilityLabel(title) }
    }
    private func stringList(_ title: String, key: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            let values = strings(draft[key])
            ForEach(values.indices, id: \.self) { index in
                HStack { TextField(title, text: Binding(get: { strings(draft[key]).indices.contains(index) ? strings(draft[key])[index] : "" }, set: { value in var next = strings(draft[key]); if next.indices.contains(index) { next[index] = value; draft[key] = .array(next.map(JSONValue.string)) } })).flatField()
                    Button { var next = strings(draft[key]); if next.indices.contains(index) { next.remove(at: index); draft[key] = .array(next.map(JSONValue.string)) } } label: { Image(systemName: "xmark") }.accessibilityLabel("Remove \(title) \(index + 1)") }
            }
            Button("Add pattern") { draft[key] = .array((values + [""]).map(JSONValue.string)) }.buttonStyle(FlatButtonStyle())
        }
    }
    private func metadataTitle(_ key: String) -> String { ["file_id":"File identity", "name":"Name", "relative_path":"Relative path", "extension":"Extension", "detected_type":"Detected type", "byte_size":"Size in bytes", "modified_at":"Modified time", "created_at":"Created time", "content_sha256":"Content fingerprint"][key] ?? key }
    private func reload() { original = node.config; draft = (try? FileInputConfiguration.validate(original)) ?? original; error = nil; loaded = true; if let family = FileInputReaderSelection.reader(sourceMode: text("sourceMode"), sourceName: context.sourceDisplayName) { reader = family }; context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: false) }
    @discardableResult private func apply(refresh: Bool) -> Bool {
        do {
            let checked = try FileInputConfiguration.validate(draft)
            if checked != original, !context.configure(checked, expected: original) { error = context.statusMessage; return false }
            original = checked; draft = checked; error = nil
            context.setNumericPending(origin: context.numericOrigin(key: draftKey), key: draftKey, pending: false)
            if refresh, context.sourceIsLinked, !context.refreshSource() { error = context.statusMessage; return false }
            return true
        } catch { self.error = error.localizedDescription; return false }
    }
    private func choose() { guard !changed || apply(refresh: false) else { return }; if !context.chooseSource() { error = context.statusMessage } }
}

enum FileInputDraftText {
    static func display(_ value: JSONValue) -> String {
        switch value {
        case .string(let text): return text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r").replacingOccurrences(of: "\t", with: "\\t")
        case .integer(let number): return String(number)
        case .number(let number): return String(number)
        case .array(let values): return values.map(display).joined(separator: ", ")
        default: return ""
        }
    }
    static func value(_ text: String, kind: String) -> JSONValue {
        if kind == "integer" { return Int64(text).map(JSONValue.integer) ?? .string(text) }
        if kind == "number" { return Double(text).map(JSONValue.number) ?? .string(text) }
        if kind == "integerList" { if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .array([]) }; return .array(text.split(separator: ",", omittingEmptySubsequences: false).map { Int64($0.trimmingCharacters(in: .whitespacesAndNewlines)).map(JSONValue.integer) ?? .string(String($0)) }) }
        var result = "", escaped = false
        for character in text {
            if escaped { result += ["n":"\n", "r":"\r", "t":"\t", "\\":"\\"][String(character)] ?? ("\\" + String(character)); escaped = false }
            else if character == "\\" { escaped = true } else { result.append(character) }
        }
        if escaped { result += "\\" }
        return .string(result)
    }
}

struct FileInputInformationView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Read a single local file or collect matching files from a folder. File identity and relative paths keep observations traceable. File types filter directories; an explicitly chosen single file uses its own reader.")
            Text("Text and documents produce text content. Images produce typed, captured pixel content; color changes are explicit, and no resizing or normalization is added. Multiple selected frames produce separate rows. Configuration shows inclusive frame numbers starting at 1; recorded decoder indexes remain zero-based. PDF page selections accept numbers and ranges such as 1, 3–5.")
            Text("Preview reads a bounded file sample. Image previews show metadata; run the flow to capture pixels and open them in Results. View requests complete selected data. Working-data record counts and percentages select files before extraction; one file can produce multiple lines, pages or frames. Sampling does not establish that unselected files can be read.")
            Text("Include and exclude patterns match relative paths. All file types is explicit. Symbolic links are skipped or reported, never followed outside the granted folder. Source permissions and original paths stay private.")
            Text("Unreadable files may stop the run, become diagnostic rows, or be skipped with reported counts. Content limits use binary units (KiB or MiB) and preserve exact byte counts. They apply to captured content, not metadata-only file sizes. Limits are enforced separately for captured bytes, expanded documents and decoded pixels. Scanned PDFs need OCR, which is outside this reader. Encrypted PDFs are reported rather than requesting a password here.")
            Text("Reader settings are saved separately for each format. Decoding replacement/ignore can lose text; explicit color conversion can change channels or bit depth. Original-depth pixels remain inspectable without automatic scaling. Image columns are not numeric training features; a future image-preparation adapter will define that conversion.")
        }.font(.caption).foregroundStyle(MLTheme.muted)
    }
}
