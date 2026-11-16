import Foundation

/// Captured worker CPU policy and optional sampled resident-memory stop threshold.
/// Resident bytes do not represent GPU allocation or a whole-machine memory limit.
public struct ExecutionResourcePolicy: Codable, Equatable {
    public static let cpuThreadBounds = 1...64
    public static let residentByteBounds = 134_217_728...1_099_511_627_776
    public let schemaVersion: Int
    public let cpuThreads: Int
    public let maximumResidentBytes: Int?
    public let workloadPolicy: ExecutionWorkloadPolicy?

    public init(cpuThreads: Int, maximumResidentBytes: Int? = nil, workloadPolicy: ExecutionWorkloadPolicy? = nil) throws {
        guard Self.cpuThreadBounds.contains(cpuThreads) else {
            throw ExecutionBudgetError.invalid("Choose a CPU thread count from 1 through 64.")
        }
        if let maximumResidentBytes, !Self.residentByteBounds.contains(maximumResidentBytes) {
            throw ExecutionBudgetError.invalid("Choose a resident-memory threshold from 128 MiB through 1 TiB.")
        }
        schemaVersion = workloadPolicy == nil ? 1 : 2
        self.workloadPolicy = workloadPolicy
        self.cpuThreads = cpuThreads
        self.maximumResidentBytes = maximumResidentBytes
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, cpuThreads, maximumResidentBytes, workloadPolicy }
    private struct Field: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        let fields = Set(try decoder.container(keyedBy: Field.self).allKeys.map(\.stringValue))
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let version = try values.decode(Int.self, forKey: .schemaVersion)
        guard [1, 2].contains(version) else {
            throw ExecutionBudgetError.invalid("Unsupported execution-resource schema.")
        }
        let required = Set(["schemaVersion", "cpuThreads"]).union(version == 2 ? ["workloadPolicy"] : [])
        guard required.isSubset(of: fields), fields.isSubset(of: required.union(["maximumResidentBytes"])) else {
            throw ExecutionBudgetError.invalid("Unknown or missing execution-resource fields.")
        }
        let resident = try values.contains(.maximumResidentBytes) ? values.decode(Int.self, forKey: .maximumResidentBytes) : nil
        try self.init(cpuThreads: values.decode(Int.self, forKey: .cpuThreads), maximumResidentBytes: resident, workloadPolicy: version == 2 ? values.decode(ExecutionWorkloadPolicy.self, forKey: .workloadPolicy) : nil)
    }
}
