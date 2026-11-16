import Foundation
import KwipMLCore

/// Private draft only. Persistence, execution readiness and Undo belong to the node owner.
struct DataColumnConfigurationDraft: Equatable {
    enum Mode: Equatable { case selection, cast }
    struct Row: Identifiable {
        let name: String
        let dtype: String
        let included: Bool
        let order: Int?
        let isMissing: Bool
        var id: String { name }
    }
    static let targetTypes = ["float64", "int64", "string", "bool"]
    let mode: Mode
    private var base: [String: JSONValue]
    private(set) var columns: [String]
    private var renames: [String: String]
    private var conversions: [String: String]

    init(configuration: [String: JSONValue], mode: Mode) {
        self.mode = mode; base = configuration
        if mode == .selection, case .array(let values) = configuration["columns"] {
            columns = values.compactMap { if case .string(let value) = $0 { return value }; return nil }
        } else if mode == .cast, case .object(let values) = configuration["columns"] {
            columns = values.keys.sorted()
        } else { columns = [] }
        if case .object(let values) = configuration["rename"] {
            renames = values.compactMapValues { if case .string(let value) = $0 { return value }; return nil }
        } else { renames = [:] }
        if mode == .cast, case .object(let values) = configuration["columns"] {
            conversions = values.compactMapValues { if case .string(let value) = $0 { return value }; return nil }
        } else { conversions = [:] }
    }
    var configuration: [String: JSONValue] {
        if mode == .selection {
            return ["columns": .array(columns.map(JSONValue.string)), "rename": .object(renames.mapValues(JSONValue.string))]
        }
        var result = base
        result["columns"] = .object(conversions.filter { columns.contains($0.key) }.mapValues(JSONValue.string))
        return result
    }
    var missingPolicy: String {
        if case .string(let value) = base["missingPolicy"] { return value }; return "error"
    }
    mutating func setMissingPolicy(_ value: String) { base["missingPolicy"] = .string(value) }
    func outputName(for name: String) -> String { renames[name] ?? name }
    func targetType(for name: String) -> String { conversions[name] ?? "float64" }
    mutating func setOutputName(_ output: String, for name: String) { renames[name] = output == name ? nil : output }
    mutating func setTargetType(_ type: String, for name: String) { conversions[name] = type }
    mutating func setIncluded(_ included: Bool, for name: String) {
        if included {
            guard !columns.contains(name), columns.count < 128 else { return }
            columns.append(name)
            if mode == .cast, conversions[name] == nil { conversions[name] = "float64" }
        } else {
            columns.removeAll { $0 == name }; renames[name] = nil; conversions[name] = nil
        }
    }
    mutating func move(_ name: String, by delta: Int) {
        guard let index = columns.firstIndex(of: name), columns.indices.contains(index + delta) else { return }
        columns.swapAt(index, index + delta)
    }
    func rows(schema: [DataColumnSchema]) -> [Row] {
        let names = columns + schema.map(\.name).filter { !columns.contains($0) }
        return names.map { name in
            let column = schema.first { $0.name == name }
            return Row(name: name, dtype: column?.dtype ?? "unresolved", included: columns.contains(name),
                       order: columns.firstIndex(of: name).map { $0 + 1 }, isMissing: !schema.isEmpty && column == nil)
        }
    }
    func validatedConfiguration(schema: [DataColumnSchema]) throws -> [String: JSONValue] {
        if mode == .selection {
            return try TableEditing.selection(columns: columns, rename: renames, available: schema.isEmpty ? nil : schema.map(\.name))
        }
        guard (1...128).contains(columns.count), Set(columns).count == columns.count,
              columns.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.unicodeScalars.count <= 200 }) else {
            throw DataFlowContractError("Choose 1–128 distinct, nonempty column names of at most 200 characters.")
        }
        guard schema.isEmpty || Set(columns).isSubset(of: Set(schema.map(\.name))) else {
            throw DataFlowContractError("A conversion column is absent from the known upstream schema. Choose its current name explicitly.")
        }
        guard Set(conversions.keys) == Set(columns), conversions.values.allSatisfy(Self.targetTypes.contains),
              ["error", "preserve"].contains(missingPolicy) else {
            throw DataFlowContractError("Choose an explicit supported type for each conversion and a supported missing-value policy.")
        }
        return configuration
    }
}
