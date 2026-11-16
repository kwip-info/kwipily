import SwiftUI
import KwipMLCore

/// Holds what the user is typing until focus leaves; intermediate drafts stay invalid,
/// rather than rounding values or silently replacing an incomplete number.
struct FileInputDraftEntry: View {
    let title: String
    @Binding var value: JSONValue
    var format: (JSONValue) -> String = FileInputDraftText.display
    var parse: (String) -> JSONValue
    @State private var text = ""
    @FocusState private var focused: Bool
    var body: some View {
        TextField(title, text: Binding(get: { text }, set: { text = $0; value = parse($0) }))
            .flatField().accessibilityLabel(title).focused($focused)
            .onAppear { text = format(value) }
            .onChange(of: value) { _, next in if !focused { text = format(next) } }
            .onChange(of: focused) { _, active in if !active { text = format(value) } }
            .onSubmit { focused = false }
    }
}

enum FileInputSizeUnit: String, CaseIterable {
    case bytes = "B", kibibytes = "KiB", mebibytes = "MiB"
    var multiplier: Int64 { switch self { case .bytes: 1; case .kibibytes: 1024; case .mebibytes: 1_048_576 } }
    static func preferred(_ value: JSONValue) -> Self {
        guard case .integer(let bytes) = value, bytes > 0 else { return .bytes }
        if bytes % Self.mebibytes.multiplier == 0 { return .mebibytes }
        return bytes % Self.kibibytes.multiplier == 0 ? .kibibytes : .bytes
    }
    func display(_ value: JSONValue) -> String {
        guard case .integer(let bytes) = value else { return FileInputDraftText.display(value) }
        if bytes % multiplier == 0 { return String(bytes / multiplier) }
        return String(Double(bytes) / Double(multiplier))
    }
    func parse(_ raw: String) -> JSONValue {
        guard let number = Double(raw.trimmingCharacters(in: .whitespacesAndNewlines)), number.isFinite else { return .string(raw) }
        let bytes = number * Double(multiplier)
        guard bytes.isFinite, bytes >= Double(Int64.min), bytes < Double(Int64.max), bytes.rounded(.towardZero) == bytes else { return .string(raw) }
        return .integer(Int64(bytes))
    }
}

struct FileInputSizeField: View {
    let title: String
    @Binding var value: JSONValue
    @State private var unit = FileInputSizeUnit.bytes
    @State private var loaded = false
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            HStack(spacing: 6) {
                FileInputDraftEntry(title: title, value: $value, format: unit.display, parse: unit.parse).id(unit)
                FlatChoiceControl(title: title + " unit", selection: unit.rawValue, options: FileInputSizeUnit.allCases.map { .init(id: $0.rawValue, title: $0.rawValue) }) { selected in
                    if let next = FileInputSizeUnit(rawValue: selected) { unit = next }
                }.frame(width: 82)
            }
            if case .string = value { Text("Enter an amount that equals a whole number of bytes.").font(.caption2).foregroundStyle(MLTheme.rose) }
        }
        .help("Binary units: 1 KiB = 1,024 bytes; 1 MiB = 1,048,576 bytes. Switching units preserves the exact byte limit.")
        .onAppear { if !loaded { unit = .preferred(value); loaded = true } }
    }
}

enum FileInputFrameNumber {
    static func display(_ value: JSONValue) -> String {
        guard case .integer(let index) = value, index < Int64.max else { return FileInputDraftText.display(value) }
        return String(index + 1)
    }
    static func parse(_ raw: String) -> JSONValue {
        guard let number = Int64(raw.trimmingCharacters(in: .whitespacesAndNewlines)), number >= 1 else { return .string(raw) }
        return .integer(number - 1)
    }
}

struct FileInputFrameRangeFields: View {
    @Binding var first: JSONValue
    @Binding var last: JSONValue
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            frameField("From frame", value: $first)
            frameField("Through frame", value: $last)
        }.help("Inclusive frame numbers, starting at 1. The stored decoder indexes remain zero-based.")
    }
    private func frameField(_ title: String, value: Binding<JSONValue>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption)
            FileInputDraftEntry(title: title, value: value, format: FileInputFrameNumber.display, parse: FileInputFrameNumber.parse)
            if case .string = value.wrappedValue { Text("Use a frame number of 1 or higher.").font(.caption2).foregroundStyle(MLTheme.rose) }
        }.frame(maxWidth: .infinity)
    }
}

enum FileInputPageSelection {
    static func selecting(_ mode: String, current: JSONValue) -> JSONValue {
        if mode == "all" { return .array([]) }
        return current == .array([]) ? .string("") : current
    }
    /// Ordered, one-based selections. Bound expansion before allocating a range.
    static func parse(_ raw: String) -> JSONValue {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return .string(raw) }
        var result: [JSONValue] = [], seen = Set<Int64>()
        for piece in text.replacingOccurrences(of: "–", with: "-").split(separator: ",", omittingEmptySubsequences: false) {
            let bounds = piece.split(separator: "-", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard (1...2).contains(bounds.count), let low = Int64(bounds[0]), (1...1_000_000).contains(low), let high = bounds.count == 1 ? low : Int64(bounds[1]), high >= low, high <= 1_000_000, high - low < 1000, result.count + Int(high - low + 1) <= 1000 else { return .string(raw) }
            for page in low...high { guard seen.insert(page).inserted else { return .string(raw) }; result.append(.integer(page)) }
        }
        return result.isEmpty ? .string(raw) : .array(result)
    }
    static func display(_ value: JSONValue) -> String {
        guard case .array(let values) = value else { if case .string(let raw) = value { return raw }; return "" }
        let pages = values.compactMap { value -> Int64? in if case .integer(let page) = value { return page }; return nil }
        guard pages.count == values.count else { return values.map(FileInputDraftText.display).joined(separator: ", ") }
        var pieces: [String] = [], index = 0
        while index < pages.count {
            let first = pages[index]; var last = first
            while index + 1 < pages.count, last < Int64.max, pages[index + 1] == last + 1 { index += 1; last = pages[index] }
            pieces.append(first == last ? String(first) : "\(first)–\(last)"); index += 1
        }
        return pieces.joined(separator: ", ")
    }
}

struct FileInputPageSelectionField: View {
    @Binding var value: JSONValue
    private var all: Bool { value == .array([]) }
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Pages").font(.caption)
            FlatChoiceControl(title: "PDF page selection", selection: all ? "all" : "selected", options: [.init(id: "all", title: "All pages"), .init(id: "selected", title: "Selected pages")]) { next in value = FileInputPageSelection.selecting(next, current: value) }
            if !all {
                FileInputDraftEntry(title: "Page numbers or ranges · e.g. 1, 3–5", value: $value, format: FileInputPageSelection.display, parse: FileInputPageSelection.parse)
                if case .string = value { Text("Use unique page numbers or inclusive ranges, starting at 1.").font(.caption2).foregroundStyle(MLTheme.rose) }
            }
        }.help("Read pages in the entered order. Select at most 1,000 pages, or choose All pages.")
    }
}

enum FileInputReaderSelection {
    static func reader(sourceMode: String?, sourceName: String?) -> String? {
        guard sourceMode == "file", let sourceName else { return nil }
        let ext = URL(fileURLWithPath: sourceName).pathExtension.lowercased()
        return FileInputConfiguration.readerOrder.first { FileInputConfiguration.extensionGroups[$0]?.contains(ext) == true }
    }
}

/// AppKit's checkbox style does not consistently honor the per-tool SwiftUI tint.
/// Keep checkbox semantics while drawing the data family's selected state explicitly.
struct FileInputCheckboxStyle: ToggleStyle {
    let tint: Color
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        Button { configuration.isOn.toggle() } label: {
            HStack(alignment: .center, spacing: 6) {
                ZStack {
                    RoundedRectangle(cornerRadius: 2).fill(configuration.isOn ? tint : MLTheme.background)
                    RoundedRectangle(cornerRadius: 2).stroke(tint.opacity(configuration.isOn ? 1 : 0.55), lineWidth: 1)
                    if configuration.isOn { Image(systemName: "checkmark").font(.system(size: 9, weight: .bold)).foregroundStyle(.white) }
                }.frame(width: 14, height: 14)
                configuration.label
            }.contentShape(Rectangle())
        }.buttonStyle(.plain).opacity(enabled ? 1 : 0.45)
            .accessibilityRepresentation {
                Toggle(isOn: configuration.$isOn) { configuration.label }.toggleStyle(.checkbox)
            }
    }
}
