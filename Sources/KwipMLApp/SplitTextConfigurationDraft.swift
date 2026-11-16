import Foundation
import KwipMLCore

struct SplitTextConfigurationDraft: Equatable {
    struct Field: Equatable, Identifiable {
        var id = UUID()
        var name: String
        var type: String = "string"
        var start: Int = 1
        var width: Int = 1
    }
    static let types = ["string", "int64", "float64", "bool"]
    static let defaults: [String: JSONValue] = [
        "mode": .string("delimited"), "source": .string(""), "missingSource": .string("preserve"), "retainSource": .bool(false),
        "delimiter": .string(","), "quoted": .bool(true), "quote": .string("\""), "escaping": .string("double"), "escape": .string("\\"),
        "outputMode": .string("columns"), "fields": .array([.object(["name": .string("part"), "type": .string("string"), "start": .integer(1), "width": .integer(1)])]),
        "extraFields": .string("error"), "shortFields": .string("missing"), "emptyTokens": .string("keep"), "ordinal": .bool(false), "ordinalName": .string("token_ordinal"),
        "measurement": .string("codepoints"), "shortRecord": .string("missing"), "trim": .bool(false), "conversionFailure": .string("error")]
    var values: [String: JSONValue]
    var fields: [Field]
    init(configuration: [String: JSONValue]) throws {
        guard Set(configuration.keys) == Set(Self.defaults.keys), case .array(let rows) = configuration["fields"], (1...128).contains(rows.count) else { throw DataFlowContractError("Split Text needs explicit settings and 1–128 fields.") }
        values = configuration
        fields = try rows.map { value in
            guard case .object(let row) = value, Set(row.keys) == ["name", "type", "start", "width"], case .string(let name) = row["name"], case .string(let type) = row["type"], case .integer(let start) = row["start"], case .integer(let width) = row["width"] else { throw DataFlowContractError("Invalid Split Text field grid.") }
            return Field(name: name, type: type, start: Int(start), width: Int(width))
        }
        try validate()
    }
    func string(_ key: String) -> String { if case .string(let value) = values[key] { return value }; return "" }
    func bool(_ key: String) -> Bool { if case .bool(let value) = values[key] { return value }; return false }
    var expands: Bool { string("mode") == "delimited" && string("outputMode") == "rows" }
    var configuration: [String: JSONValue] {
        var result = values
        result["fields"] = .array(fields.map { .object(["name": .string($0.name), "type": .string($0.type), "start": .integer(Int64($0.start)), "width": .integer(Int64($0.width))]) })
        return result
    }
    private func validate() throws {
        let choices = ["mode": ["delimited", "fixedWidth"], "missingSource": ["preserve", "error"], "escaping": ["double", "escape", "none"], "outputMode": ["columns", "rows"], "extraFields": ["error", "drop", "remainder"], "shortFields": ["missing", "empty", "error"], "emptyTokens": ["keep", "drop"], "measurement": ["codepoints", "utf8"], "shortRecord": ["missing", "error"], "conversionFailure": ["error", "missing"]]
        for (key, allowed) in choices where !allowed.contains(string(key)) { throw DataFlowContractError("Choose a supported Split Text \(key) policy.") }
        for key in ["retainSource", "quoted", "ordinal", "trim"] { guard case .bool = values[key] else { throw DataFlowContractError("Invalid Split Text checkbox.") } }
        guard string("source").unicodeScalars.count <= 200, validName(string("ordinalName")), (1...128).contains(string("delimiter").unicodeScalars.count), string("quote").unicodeScalars.count == 1, string("escape").unicodeScalars.count == 1 else { throw DataFlowContractError("Choose a delimiter and single Unicode code point quote and escape characters.") }
        if string("mode") == "delimited" {
            if bool("quoted") && string("delimiter").contains(string("quote")) { throw DataFlowContractError("Delimiter cannot contain the active quote.") }
            if string("escaping") == "escape" && (string("delimiter").contains(string("escape")) || (bool("quoted") && string("escape") == string("quote"))) { throw DataFlowContractError("Escape must differ from quote and delimiter characters.") }
        }
        guard !fields.isEmpty, fields.count <= 128, fields.allSatisfy({ validName($0.name) && Self.types.contains($0.type) && (1...10_000_000).contains($0.start) && (1...10_000_000).contains($0.width) }) else { throw DataFlowContractError("Fields need bounded names, scalar types and positive starts/widths.") }
        let names = (expands ? Array(fields.prefix(1)) : fields).map(\.name)
        guard Set(names).count == names.count else { throw DataFlowContractError("Output names must be unique.") }
    }
    private func validName(_ value: String) -> Bool { !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && value.unicodeScalars.count <= 200 }
    func validatedConfiguration(schema: [DataColumnSchema]) throws -> [String: JSONValue] {
        try validate()
        guard validName(string("source")) else { throw DataFlowContractError("Choose the source text column.") }
        if !schema.isEmpty { _ = try outputSchema(schema) }
        return configuration
    }
    func outputSchema(_ schema: [DataColumnSchema]) throws -> [DataColumnSchema] {
        try validate()
        guard let source = schema.first(where: { $0.name == string("source") }), source.dtype == "string" else { throw DataFlowContractError("Choose a current upstream Text column.") }
        var output = schema.filter { bool("retainSource") || $0.name != source.name }
        output += (expands ? Array(fields.prefix(1)) : fields).map { DataColumnSchema(name: $0.name, dtype: $0.type, nullable: true) }
        if expands && bool("ordinal") { output.append(DataColumnSchema(name: string("ordinalName"), dtype: "int64", nullable: false)) }
        guard output.count <= 128, Set(output.map(\.name)).count == output.count else { throw DataFlowContractError("Output names collide with retained columns or exceed 128 columns.") }
        return output
    }
}
