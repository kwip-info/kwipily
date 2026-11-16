import Foundation
import KwipMLCore

enum TableEditing {
    private static func validName(_ value: String) -> Bool { !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && value.unicodeScalars.count <= 200 }
    static func selection(columns: [String], rename: [String: String], available: [String]?) throws -> [String: JSONValue] {
        guard (1...128).contains(columns.count), columns.allSatisfy(validName), Set(columns).count == columns.count else { throw DataFlowContractError("Select 1–128 distinct, nonempty column names of at most 200 characters.") }
        guard Set(rename.keys).isSubset(of: Set(columns)) else { throw DataFlowContractError("Rename only selected source columns.") }
        if let available, !Set(columns).isSubset(of: Set(available)) { throw DataFlowContractError("A selected column is absent from the known upstream schema.") }
        let outputs = columns.map { rename[$0] ?? $0 }
        guard outputs.allSatisfy(validName), Set(outputs).count == outputs.count else { throw DataFlowContractError("Output names must be nonempty and unique. Apply simultaneous renames together.") }
        return ["columns": .array(columns.map(JSONValue.string)), "rename": .object(rename.mapValues(JSONValue.string))]
    }
    static func filter(column: String, comparison: String, value: JSONValue, dtype: String?) throws -> [String: JSONValue] {
        guard validName(column), ["eq", "ne", "lt", "le", "gt", "ge"].contains(comparison) else { throw DataFlowContractError("Choose one exact column and a declared comparison.") }
        let literalType: String
        switch value {
        case .string: literalType = "string"
        case .bool: literalType = "bool"
        case .integer: literalType = "number"
        case .number(let number): guard number.isFinite else { throw DataFlowContractError("Numeric filter literals must be finite.") }; literalType = "number"
        default: throw DataFlowContractError("A filter literal must be text, a number or a Boolean.")
        }
        if let dtype {
            let expected = ["int64", "float64"].contains(dtype) ? "number" : dtype
            guard expected == literalType else { throw DataFlowContractError("Literal type differs from \(dtype). Cast the column explicitly; filters never coerce values.") }
        }
        return ["column": .string(column), "comparison": .string(comparison), "value": value]
    }
}

struct DataColumnSchema: Equatable {
    var name: String
    var dtype: String
    var nullable: Bool?
    var temporal: [String: JSONValue]? = nil
}
