import Foundation
import KwipMLCore

/// Guided controls serialize the same allowlisted language as Formula's expression editor.
struct FormulaGuidedTemporal: Equatable {
    static let kinds = ["temporal_parse", "temporal_format", "temporal_timezone", "temporal_part", "temporal_difference"]
    var kind = "temporal_parse", column = "", otherColumn = ""
    var dtype = "datetime", pattern = "iso", sourceTimezone = "UTC", timezone = "UTC"
    var invalid = "stop", ambiguous = "stop", nonexistent = "stop"
    var part = "year", units = "seconds", representation = "decimal"
    init(kind: String = "temporal_parse", column: String = "") { self.kind = kind; self.column = column }
    var title: String {
        switch kind {
        case "temporal_format": return "Format date/time"
        case "temporal_timezone": return "Convert timezone"
        case "temporal_part": return "Extract date/time part"
        case "temporal_difference": return "Date/time difference"
        default: return "Parse date/time"
        }
    }
    var outputType: String {
        switch kind {
        case "temporal_format": return "string"
        case "temporal_timezone": return "datetime"
        case "temporal_part": return "int64"
        case "temporal_difference": return representation == "integer" ? "int64" : "float64"
        default: return dtype
        }
    }
    var expression: String {
        let options: [String]
        switch kind {
        case "temporal_format": options = [pattern, timezone]
        case "temporal_timezone": options = [timezone]
        case "temporal_part": options = [part, timezone]
        case "temporal_difference": options = [units, representation]
        default: options = [dtype, pattern, sourceTimezone, timezone, invalid, ambiguous, nonexistent, "v1"]
        }
        let columns = kind == "temporal_difference" ? "[\(column)], [\(otherColumn)], " : "[\(column)], "
        return "\(kind)(\(columns)" + options.map(FormulaGuidedRegex.quote).joined(separator: ", ") + ")"
    }
    init?(expression: String) {
        let text = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let open = text.firstIndex(of: "("), text.hasSuffix(")") else { return nil }
        let function = String(text[..<open])
        guard Self.kinds.contains(function) else { return nil }
        self.init(kind: function)
        var tail = String(text[text.index(after: open)..<text.index(before: text.endIndex)])
        func takeColumn() -> String? {
            tail = tail.trimmingCharacters(in: .whitespacesAndNewlines)
            guard tail.first == "[", let end = tail.firstIndex(of: "]") else { return nil }
            let name = String(tail[tail.index(after: tail.startIndex)..<end])
            tail = String(tail[tail.index(after: end)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard tail.first == "," else { return nil }; tail.removeFirst()
            return name
        }
        guard let first = takeColumn() else { return nil }; column = first
        if function == "temporal_difference" {
            guard let second = takeColumn() else { return nil }; otherColumn = second
        }
        guard let bytes = ("[" + tail + "]").data(using: .utf8),
              let options = try? JSONDecoder().decode([String].self, from: bytes) else { return nil }
        switch function {
        case "temporal_format": guard options.count == 2 else { return nil }; pattern = options[0]; timezone = options[1]
        case "temporal_timezone": guard options.count == 1 else { return nil }; timezone = options[0]
        case "temporal_part": guard options.count == 2 else { return nil }; part = options[0]; timezone = options[1]
        case "temporal_difference": guard options.count == 2 else { return nil }; units = options[0]; representation = options[1]
        default:
            guard options.count == 8, options[7] == "v1" else { return nil }
            dtype = options[0]; pattern = options[1]; sourceTimezone = options[2]; timezone = options[3]
            invalid = options[4]; ambiguous = options[5]; nonexistent = options[6]
        }
    }
    /// Schema guidance for temporal expressions; execution still validates full expression types.
    static func metadata(for expression: String, outputType: String, schema: [DataColumnSchema], depth: Int = 0) throws -> [String:JSONValue] {
        guard depth <= 32 else { throw DataFlowContractError("Temporal expression nesting exceeds 32 levels.") }
        let text = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.first == "(", text.last == ")" {return try metadata(for:String(text.dropFirst().dropLast()),outputType:outputType,schema:schema,depth:depth+1)}
        if text.first == "[", text.last == "]", !text.dropFirst().dropLast().contains("]"),
           let column = schema.first(where: { $0.name == String(text.dropFirst().dropLast()) && $0.dtype == outputType }), let metadata = column.temporal { return metadata }
        guard let open = text.firstIndex(of: "("), text.last == ")" else { throw DataFlowContractError("Temporal output requires a typed source or explicit Parse.") }
        let function = String(text[..<open])
        let body = String(text[text.index(after: open)..<text.index(before: text.endIndex)])
        var arguments: [String] = [], current = "", nesting = 0, quoted = false, escaped = false, bracket = false
        for character in body {
            if quoted {
                current.append(character)
                if escaped { escaped = false } else if character == "\\" { escaped = true } else if character == "\"" { quoted = false }
                continue
            }
            if character == "\"" { quoted = true }
            else if character == "[" { bracket = true }
            else if character == "]" { bracket = false }
            else if !bracket && character == "(" { nesting += 1 }
            else if !bracket && character == ")" { nesting -= 1 }
            if character == "," && nesting == 0 && !bracket { arguments.append(current); current = "" } else { current.append(character) }
        }
        guard !quoted, nesting == 0, !bracket else { throw DataFlowContractError("Invalid temporal expression arguments.") }
        arguments.append(current)
        func literal(_ index: Int) throws -> String {
            guard arguments.indices.contains(index), let bytes = arguments[index].trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8), let value = try? JSONDecoder().decode(String.self,from:bytes) else { throw DataFlowContractError("Temporal options must be explicit quoted strings.") }
            return value
        }
        if function == "coalesce" {
            let values = try arguments.filter { $0.trimmingCharacters(in:.whitespacesAndNewlines).lowercased() != "null" }.map { try metadata(for:$0,outputType:outputType,schema:schema,depth:depth+1) }
            guard let first = values.first, values.allSatisfy({$0 == first}) else { throw DataFlowContractError("Temporal coalesce needs matching precision and timezone metadata.") }
            return first
        }
        let timezone: String
        if function == "temporal_parse" {
            guard arguments.count == 9, try literal(1) == outputType, try literal(8) == "v1" else { throw DataFlowContractError("Parse output type or version differs from its expression.") }
            timezone = try literal(4)
        } else if function == "temporal_timezone" {
            guard arguments.count == 2, outputType == "datetime" else { throw DataFlowContractError("Convert timezone produces DateTime.") }
            timezone = try literal(1)
        } else { throw DataFlowContractError("Temporal output requires a typed source, Parse, Convert timezone or compatible coalesce.") }
        var result: [String:JSONValue] = ["version":.integer(1),"unit":.string(outputType == "date" ? "day":"microsecond")]
        if outputType == "datetime" { result["timezone"] = .string(timezone) }
        guard TemporalColumnContract.validMetadata(.object(result),dtype:outputType) else { throw DataFlowContractError("Choose valid temporal precision and timezone metadata.") }
        return result
    }

}
