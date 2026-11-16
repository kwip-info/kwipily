import Foundation

/// Only parser-owned codes and positions cross into presentation; raw source/error text does not.
struct CSVSourceDiagnostic: Codable, Equatable {
    let code: String
    let record: Int
    let line: Int
    let message: String

    private static let prefix = "kwip-csv-diagnostic:"
    private static let messages = [
        "byteLimit": "The source exceeds the 10 MB limit. Use a smaller file.",
        "encoding": "The file could not be decoded. Choose the encoding used by this file.",
        "recordEnding": "The record ending differs from the selected setting. Choose Auto or the matching ending.",
        "quotedNewline": "A quoted field spans multiple lines. Enable quoted newlines or correct this record.",
        "header": "Headers must contain 1–128 unique, nonempty names. Correct the header or use generated names.",
        "columnLimit": "The table must contain 1–128 columns. Check the delimiter or use a narrower table.",
        "rowWidth": "This record has a different field count from the table. Check the delimiter, quoting and blank-record settings.",
        "rowLimit": "The source exceeds 100,000 observations. Use a smaller file.",
        "syntax": "This record has invalid quoting or escaping, or an oversized field. Check the quote and escape settings.",
        "empty": "No observations remain after the header and skipped records. Check those settings or choose another file."
    ]
    static func decodeEvent(_ data: Data) -> Self? {
        guard data.count <= 4096,
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              event["event"] as? String == "error",
              let diagnostic = event["csvDiagnostic"],
              let encoded = try? JSONSerialization.data(withJSONObject: diagnostic) else { return nil }
        return decodePayload(encoded)
    }
    private static func decodePayload(_ data: Data) -> Self? {
        guard data.count <= 2048,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == ["code", "record", "line", "message"],
              let decoded = try? JSONDecoder().decode(Self.self, from: data),
              let safeMessage = messages[decoded.code], decoded.message.utf8.count <= 1024,
              (0...10_000_000).contains(decoded.record), (0...10_000_000).contains(decoded.line) else { return nil }
        return .init(code: decoded.code, record: decoded.record, line: decoded.line, message: safeMessage)
    }
    var transport: String {
        Self.prefix + ((try? JSONEncoder().encode(self))?.base64EncodedString() ?? "")
    }
    static func decodeTransport(_ value: String) -> Self? {
        guard value.hasPrefix(prefix), value.utf8.count <= 4096,
              let data = Data(base64Encoded: String(value.dropFirst(prefix.count))) else { return nil }
        return decodePayload(data)
    }
    var detail: String {
        let position = record > 0 ? "Record \(record)" + (line > 0 ? ", line \(line)" : "") + ": " : ""
        return position + message
    }
}
