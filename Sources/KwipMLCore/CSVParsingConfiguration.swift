import Foundation

/// Version-one delimited-text settings. Missing values preserve saved CSV-source semantics.
public enum CSVParsingConfiguration {
    public static let legacyDefaults: [String: JSONValue] = [
        "separator": .string(","), "quoteChar": .string("\""), "escapeMode": .string("double"), "escapeChar": .string("\\"),
        "encoding": .string("utf-8-sig"), "recordEndings": .string("auto"),
        "allowQuotedNewlines": .bool(true), "headerMode": .string("firstRecord"),
        "skipRecords": .integer(0), "skipBlankRecords": .bool(false), "skipInitialSpace": .bool(false),
        "emptyField": .string("null"), "nullMarkers": .array([])
    ]
    public static let newDefaults: [String: JSONValue] = [
        "parserProfile": .string("polars"), "separator": .string(","), "quoteChar": .string("\""),
        "encoding": .string("utf-8-sig"), "headerMode": .string("firstRecord"),
        "skipRecords": .integer(0), "skipLines": .integer(0), "skipRowsAfterHeader": .integer(0),
        "commentPrefix": .string(""), "emptyField": .string("text"), "nullMarkers": .array([])
    ].merging(TableInputConfiguration.csvAdvancedDefaults) { _, value in value }
    public static let allowedKeys = Set(legacyDefaults.keys).union(newDefaults.keys)
    public static func isPolars(_ config: [String: JSONValue]) -> Bool { config["parserProfile"] == .string("polars") }

    /// Explicit editing migration. Unsupported legacy dialect settings reset only at the owner's request.
    public static func polarsMigration(_ config: [String: JSONValue]) -> [String: JSONValue] {
        var result = newDefaults
        var characters = result
        for key in ["separator", "quoteChar"] { if let value = config[key] { characters[key] = value } }
        if (try? validate(characters)) != nil { result = characters }
        for key in newDefaults.keys where !["parserProfile", "separator", "quoteChar"].contains(key) {
            if let value = config[key] {
                var candidate = result; candidate[key] = value
                if (try? validate(candidate)) != nil { result[key] = value }
            }
        }
        return result
    }
    public static func isSeparator(_ value: String) -> Bool {
        value.unicodeScalars.count == 1 && !["\r", "\n", "\0"].contains(value)
    }
    public static func validate(_ config: [String: JSONValue]) throws -> [String: JSONValue] {
        if config["parserProfile"] != nil { return try validatePolars(config) }
        guard Set(config.keys).isSubset(of: Set(legacyDefaults.keys)) else { throw DataFlowContractError("Unknown CSV parsing option.") }
        let result = legacyDefaults.merging(config) { _, new in new }
        func text(_ key: String) throws -> String {
            guard case .string(let value) = result[key] else { throw DataFlowContractError("Invalid CSV setting: \(key).") }; return value
        }
        let delimiter = try text("separator"), quote = try text("quoteChar"), escape = try text("escapeMode")
        guard isSeparator(delimiter) else { throw DataFlowContractError("Delimiter must be one character, excluding a newline or NUL.") }
        guard quote.isEmpty || isSeparator(quote) else { throw DataFlowContractError("Quote must be one character, or None.") }
        let escapeChar = try text("escapeChar")
        guard isSeparator(escapeChar) else { throw DataFlowContractError("Escape must be one character, excluding a newline or NUL.") }
        guard escape != "custom" || ![delimiter, quote].contains(escapeChar) else { throw DataFlowContractError("Escape must differ from delimiter and quote.") }
        guard delimiter != quote, !(escape == "backslash" && [delimiter, quote].contains("\\")) else { throw DataFlowContractError("Delimiter, quote and escape characters must differ.") }
        let choices = ["escapeMode": ["double", "backslash", "none", "custom"], "encoding": ["utf-8-sig", "utf-8", "utf-16", "cp1252"], "recordEndings": ["auto", "lf", "crlf", "cr"], "headerMode": ["firstRecord", "generated"], "emptyField": ["text", "null"]]
        for (key, values) in choices { guard try values.contains(text(key)) else { throw DataFlowContractError("Choose a supported CSV \(key).") } }
        for key in ["allowQuotedNewlines", "skipBlankRecords", "skipInitialSpace"] {
            guard case .bool = result[key] else { throw DataFlowContractError("Invalid CSV switch: \(key).") }
        }
        guard case .integer(let skip) = result["skipRecords"], (0...100_000).contains(skip) else { throw DataFlowContractError("Records to skip must be a whole number from 0 to 100,000.") }
        guard case .array(let markers) = result["nullMarkers"], markers.count <= 32 else { throw DataFlowContractError("Use at most 32 null markers.") }
        var seen = Set<String>()
        for marker in markers {
            guard case .string(let value) = marker, !value.isEmpty, value.unicodeScalars.count <= 200, !value.contains("\0"), seen.insert(value).inserted else { throw DataFlowContractError("Null markers must be distinct, nonempty text of at most 200 characters.") }
        }
        return result
    }
    private static func validatePolars(_ config: [String: JSONValue]) throws -> [String: JSONValue] {
        guard isPolars(config) else { throw DataFlowContractError("Choose a supported CSV parser profile.") }
        // Older shared configuration editors may carry this switch. The explicit native profile
        // selects Polars blank-line behavior; it is not an option in the normalized profile.
        var native = config
        if let value = native.removeValue(forKey: "skipBlankRecords"), case .bool = value {} else if config["skipBlankRecords"] != nil {
            throw DataFlowContractError("Invalid CSV switch: skipBlankRecords.")
        }
        for key in ["escapeMode", "escapeChar", "recordEndings", "allowQuotedNewlines", "skipInitialSpace"] {
            if let value = native.removeValue(forKey: key), value != legacyDefaults[key] {
                throw DataFlowContractError("This CSV option is not supported by Polars. Use legacy parsing or reset to Polars options.")
            }
        }
        guard Set(native.keys).isSubset(of: Set(newDefaults.keys)) else {
            throw DataFlowContractError("This CSV option is not supported by Polars. Use legacy parsing or reset to Polars options.")
        }
        let result = newDefaults.merging(native) { _, new in new }
        func text(_ key: String) throws -> String {
            guard case .string(let value) = result[key] else { throw DataFlowContractError("Invalid CSV setting: \(key).") }; return value
        }
        let separator = try text("separator"), quote = try text("quoteChar")
        guard isSeparator(separator), separator.utf8.count == 1 else { throw DataFlowContractError("Polars delimiter must be one byte, excluding newline or NUL.") }
        guard quote.isEmpty || (isSeparator(quote) && quote.utf8.count == 1) else { throw DataFlowContractError("Polars quote must be one byte, or None.") }
        guard separator != quote else { throw DataFlowContractError("Delimiter and quote must differ.") }
        for (key, values) in ["encoding": ["utf-8-sig", "utf8-lossy"], "headerMode": ["firstRecord", "generated"], "emptyField": ["text", "null"]] {
            guard try values.contains(text(key)) else { throw DataFlowContractError("Choose a supported Polars CSV \(key).") }
        }
        for key in ["skipRecords", "skipLines", "skipRowsAfterHeader"] {
            guard case .integer(let count) = result[key], (0...100_000).contains(count) else { throw DataFlowContractError("CSV skip counts must be whole numbers from 0 to 100,000.") }
        }
        guard result["skipRecords"] == .integer(0) || result["skipLines"] == .integer(0) else {
            throw DataFlowContractError("Choose either records to skip or physical lines to skip; Polars cannot combine both.")
        }
        let prefix = try text("commentPrefix")
        guard prefix.unicodeScalars.count <= 200, !prefix.contains("\0"), !prefix.contains("\r"), !prefix.contains("\n") else { throw DataFlowContractError("Comment prefix must be at most 200 characters without newline or NUL.") }
        guard case .array(let markers) = result["nullMarkers"], markers.count <= 32 else { throw DataFlowContractError("Use at most 32 null markers.") }
        var seen = Set<String>()
        for marker in markers {
            guard case .string(let value) = marker, !value.isEmpty, value.unicodeScalars.count <= 200, !value.contains("\0"), seen.insert(value).inserted else { throw DataFlowContractError("Null markers must be distinct, nonempty text of at most 200 characters.") }
        }
        let advanced = result.filter { TableInputConfiguration.csvAdvancedDefaults.keys.contains($0.key) }
        _ = try TableInputConfiguration.validateCSVAdvanced(advanced)
        if case .object(let perColumn) = result["columnNullValues"], !perColumn.isEmpty, !markers.isEmpty {
            throw DataFlowContractError("Choose either global null markers or per-column null markers.")
        }
        if case .string(let eol) = result["eolChar"], eol == separator || eol == quote { throw DataFlowContractError("Record terminator must differ from delimiter and quote.") }
        if result["decimalComma"] == .bool(true), separator == "," { throw DataFlowContractError("Comma decimals need a different delimiter.") }
        return result
    }
    public static func transportJSON(_ config: [String: JSONValue]) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(validate(config))
        guard data.count <= (isPolars(config) ? 32_768 : 4096), let text = String(data: data, encoding: .utf8) else { throw DataFlowContractError("CSV parsing settings exceed the transport limit.") }
        return text
    }
}
