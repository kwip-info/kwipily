import Foundation

/// Operation-scoped workload ceilings. These never grant files, credentials or
/// permission, and do not replace bounded decoder/message frames.
public struct ExecutionWorkloadPolicy: Codable, Equatable {
    public enum Preset: String, Codable, CaseIterable { case legacy, light, balanced, throughput, custom }
    public enum Setting: String, CaseIterable {
        case scratchBytes, retainedBytes, tableRows, tableCells, tableColumns, sourceBytes
        case collectionMembers, collectionBytes, collectionMemberBytes, planNodes, planEdges, parallelTasks
    }
    public let schemaVersion: Int
    public let preset: Preset
    public let limits: [String: Int64]

    private struct Specification: Decodable {
        struct Field: Decodable { let id: String; let unit: String; let scope: String; let minimum: Int64 }
        let schemaVersion: Int
        let integerMaximum: Int64
        let settings: [Field]
        let presets: [String: [String: Int64]]
    }
    private static let specification: Result<Specification, Error> = Result {
        guard let url = Bundle.module.url(forResource: "execution-workload-policy", withExtension: "json") else {
            throw ExecutionBudgetError.invalid("The shared workload-policy specification is missing.")
        }
        let spec = try JSONDecoder().decode(Specification.self, from: Data(contentsOf: url))
        guard spec.schemaVersion == 1, spec.integerMaximum == 9_007_199_254_740_991,
              Set(spec.settings.map(\.id)) == Set(Setting.allCases.map(\.rawValue)),
              spec.settings.count == Setting.allCases.count,
              spec.settings.allSatisfy({ $0.minimum == 1 && $0.scope == "operation" }),
              Set(spec.presets.keys) == Set(Preset.allCases.filter { $0 != .custom }.map(\.rawValue)) else {
            throw ExecutionBudgetError.invalid("The shared workload-policy specification is invalid.")
        }
        return spec
    }
    public init(preset: Preset = .legacy, limits: [String: Int64]? = nil) throws {
        let spec = try Self.specification.get()
        guard let values = limits ?? spec.presets[preset.rawValue],
              Set(values.keys) == Set(Setting.allCases.map(\.rawValue)),
              values.values.allSatisfy({ $0 >= 1 && $0 <= spec.integerMaximum }),
              preset == .custom || values == spec.presets[preset.rawValue] else {
            throw ExecutionBudgetError.invalid("Workload allowances must be positive whole numbers and match their selected preset or Custom.")
        }
        schemaVersion = 1; self.preset = preset; self.limits = values
    }
    public func limit(_ setting: Setting) -> Int64 { limits[setting.rawValue]! }
    public func overriding(_ values: [String: Int64]) throws -> Self {
        guard Set(values.keys).isSubset(of: Set(Setting.allCases.map(\.rawValue))) else {
            throw ExecutionBudgetError.invalid("Unknown workload allowance.")
        }
        return try .init(preset: .custom, limits: limits.merging(values) { _, new in new })
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, preset, limits }
    private struct Field: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        guard Set(try decoder.container(keyedBy: Field.self).allKeys.map(\.stringValue)) == ["schemaVersion", "preset", "limits"] else {
            throw ExecutionBudgetError.invalid("Unknown or missing workload-policy fields.")
        }
        let fields = try decoder.container(keyedBy: CodingKeys.self)
        guard try fields.decode(Int.self, forKey: .schemaVersion) == 1 else {
            throw ExecutionBudgetError.invalid("Unsupported workload policy.")
        }
        try self.init(preset: fields.decode(Preset.self, forKey: .preset), limits: fields.decode([String: Int64].self, forKey: .limits))
    }
}
