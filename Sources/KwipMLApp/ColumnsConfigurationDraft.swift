import Foundation
import KwipMLCore

/// Atomic column-grid draft. No schema reads, execution or graph mutation.
struct ColumnsConfigurationDraft: Equatable {
    struct Row: Equatable, Identifiable {
        let source: String
        var included: Bool
        var outputName: String
        var targetType: String
        var id: String { source }
    }
    static let types = ["keep", "float64", "int64", "string", "bool", "image"]
    static let defaults: [String: JSONValue] = ["rows": .array([]), "castFailure": .string("error"),
        "trimText": .bool(false), "newColumns": .string("keep"), "missingPolicy": .string("preserve")]
    private(set) var rows: [Row]
    var castFailure: String
    var trimText: Bool
    var newColumns: String
    var missingPolicy: String

    init(configuration: [String: JSONValue]) throws {
        guard Set(configuration.keys) == Set(Self.defaults.keys),
              case .array(let values) = configuration["rows"], values.count <= 128,
              case .string(let failure) = configuration["castFailure"], ["error", "null"].contains(failure),
              case .bool(let trim) = configuration["trimText"],
              case .string(let fresh) = configuration["newColumns"], ["keep", "drop", "error"].contains(fresh),
              case .string(let missing) = configuration["missingPolicy"], ["error", "preserve"].contains(missing) else {
            throw DataFlowContractError("Choose supported Columns policies and at most 128 column rows.")
        }
        rows = try values.map { value in
            guard case .object(let row) = value, Set(row.keys) == ["source", "included", "outputName", "targetType"],
                  case .string(let source) = row["source"], case .bool(let included) = row["included"],
                  case .string(let name) = row["outputName"], case .string(let type) = row["targetType"],
                  Self.validName(source), Self.validName(name), Self.types.contains(type) else {
                throw DataFlowContractError("Each column needs an exact source, output name, Keep choice and supported type.")
            }
            return Row(source: source, included: included, outputName: name, targetType: type)
        }
        castFailure = failure; trimText = trim; newColumns = fresh; missingPolicy = missing
        try validateRows()
    }
    var configuration: [String: JSONValue] {
        ["rows": .array(rows.map { .object(["source": .string($0.source), "included": .bool($0.included),
            "outputName": .string($0.outputName), "targetType": .string($0.targetType)]) }),
         "castFailure": .string(castFailure), "trimText": .bool(trimText),
         "newColumns": .string(newColumns), "missingPolicy": .string(missingPolicy)]
    }
    func hasUserChanges(comparedTo saved: [String: JSONValue], schema: [DataColumnSchema]) -> Bool {
        // Schema reconciliation populates visible choices; it is not a user edit.
        // Compare with the same projection while retaining exact saved config for Apply.
        guard var baseline = try? Self(configuration: saved) else { return true }
        baseline.reconcile(schema: schema)
        return self != baseline
    }
    func requiresSchemaReview(comparedTo saved: [String: JSONValue], schema: [DataColumnSchema]) -> Bool {
        guard let baseline = try? Self(configuration: saved), baseline.newColumns == "error" else { return false }
        return schema.contains { column in !baseline.rows.contains { $0.source == column.name } }
    }
    var keptCount: Int { rows.filter(\.included).count }
    private static func validName(_ name: String) -> Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && name.unicodeScalars.count <= 200
    }
    private func validateRows() throws {
        guard Set(rows.map(\.source)).count == rows.count else { throw DataFlowContractError("A source column appears more than once.") }
        let included = rows.filter(\.included)
        guard Set(included.map(\.outputName)).count == included.count else { throw DataFlowContractError("Output names must be distinct. Name swaps can be applied together.") }
    }
    mutating func reconcile(schema: [DataColumnSchema]) {
        // Preserve missing configured rows for actionable validation; do not silently repair them.
        for column in schema where !rows.contains(where: { $0.source == column.name }) && rows.count < 128 {
            rows.append(Row(source: column.name, included: newColumns == "keep", outputName: column.name, targetType: "keep"))
        }
    }
    mutating func add(source: String) {
        guard Self.validName(source), !rows.contains(where: { $0.source == source }), rows.count < 128 else { return }
        rows.append(Row(source: source, included: true, outputName: source, targetType: "keep"))
    }
    mutating func update(_ source: String, _ change: (inout Row) -> Void) {
        guard let index = rows.firstIndex(where: { $0.source == source }) else { return }
        change(&rows[index])
    }
    mutating func setAllIncluded(_ included: Bool) { for index in rows.indices { rows[index].included = included } }
    mutating func move(_ source: String, by delta: Int) {
        guard let index = rows.firstIndex(where: { $0.source == source }), rows.indices.contains(index + delta) else { return }
        rows.swapAt(index, index + delta)
    }
    mutating func move(_ source: String, to index: Int) {
        guard rows.indices.contains(index), let current = rows.firstIndex(where: { $0.source == source }), current != index else { return }
        let row = rows.remove(at: current)
        rows.insert(row, at: index)
    }
    mutating func move(_ source: String, before target: String) -> Bool {
        guard source != target, let index = rows.firstIndex(where: { $0.source == source }), rows.contains(where: { $0.source == target }) else { return false }
        let row = rows.remove(at: index)
        rows.insert(row, at: rows.firstIndex(where: { $0.source == target })!)
        return true
    }
    mutating func removeMissing(_ source: String, schema: [DataColumnSchema]) {
        guard !schema.isEmpty, !schema.contains(where: { $0.name == source }) else { return }
        rows.removeAll { $0.source == source }
    }
    func validatedConfiguration(schema: [DataColumnSchema]) throws -> [String: JSONValue] {
        let parsed = try Self(configuration: configuration)
        if !schema.isEmpty { _ = try parsed.outputSchema(schema) }
        else if rows.allSatisfy({ !$0.included }) && newColumns != "keep" {
            throw DataFlowContractError("Keep at least one output column.")
        }
        return configuration
    }
    func outputSchema(_ schema: [DataColumnSchema]) throws -> [DataColumnSchema] {
        try validateRows()
        guard rows.allSatisfy({ row in schema.contains { $0.name == row.source } }) else {
            throw DataFlowContractError("A configured source column is missing upstream. Remove its stale row or reconnect the intended input.")
        }
        let unconfigured = schema.filter { column in !rows.contains { $0.source == column.name } }
        guard newColumns != "error" || unconfigured.isEmpty else { throw DataFlowContractError("New input columns need explicit grid choices.") }
        var result = try rows.filter(\.included).map { row -> DataColumnSchema in
            var column = schema.first { $0.name == row.source }!
            column.name = row.outputName
            if row.targetType == "image" {
                guard column.dtype == "image" else { throw DataFlowContractError("Only an existing image column can retain the Image type. Read images with File Input first.") }
            } else if row.targetType != "keep" {
                guard column.dtype != "image" else { throw DataFlowContractError("Image columns cannot be cast to scalar values.") }
                column.dtype = row.targetType
                column.nullable = castFailure == "null" ? true : missingPolicy == "error" ? false : column.nullable
            }
            return column
        }
        if newColumns == "keep" { result += unconfigured }
        guard !result.isEmpty else { throw DataFlowContractError("Keep at least one output column.") }
        guard Set(result.map(\.name)).count == result.count else { throw DataFlowContractError("An output name conflicts with another kept or new column.") }
        return result
    }
}
