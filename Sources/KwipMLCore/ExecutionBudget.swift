import Foundation

/// Time policy is captured once per operation, not reread between worker phases.
/// Allocation, retention and capture policies have separate enforcement semantics.
public struct ExecutionBudgetPreferences: Codable, Equatable {
    public let schemaVersion: Int
    public let maximumRunSeconds: Int
    public static let maximumRunSecondsBounds = 1...86400
    public static let factory = ExecutionBudgetPreferences(validatedSeconds: 3600)

    public init(maximumRunSeconds: Int) throws {
        try validateExecutionSeconds(maximumRunSeconds)
        self.init(validatedSeconds: maximumRunSeconds)
    }
    private init(validatedSeconds: Int) {
        schemaVersion = 1
        maximumRunSeconds = validatedSeconds
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, maximumRunSeconds }
    public init(from decoder: Decoder) throws {
        try requireBudgetFields(decoder, required: ["schemaVersion", "maximumRunSeconds"])
        let values = try decoder.container(keyedBy: CodingKeys.self)
        guard try values.decode(Int.self, forKey: .schemaVersion) == 1 else {
            throw ExecutionBudgetError.invalid("Unsupported time-budget schema.")
        }
        try self.init(maximumRunSeconds: values.decode(Int.self, forKey: .maximumRunSeconds))
    }
}

/// Native admission provenance. Omitted quota means unlimited only when this
/// record exists; old budgets lacking provenance make no storage-policy claim.
public struct ExecutionPolicyProvenance: Codable, Equatable {
    public let schemaVersion: Int
    public let storageScope: String
    public let storageSource: ResolvedExecutionBudget.Source
    public let managedStorageLimitBytes: Int64?
    public let cpuThreadsSource: ResolvedExecutionBudget.Source
    public let maximumResidentBytesSource: ResolvedExecutionBudget.Source
    public init(managedStorageLimitBytes: Int64?, cpuThreadsSource: ResolvedExecutionBudget.Source,
                maximumResidentBytesSource: ResolvedExecutionBudget.Source) throws {
        if let managedStorageLimitBytes, managedStorageLimitBytes <= 0 {
            throw ExecutionBudgetError.invalid("Captured managed-storage allowance must be positive.")
        }
        schemaVersion = 1; storageScope = "workspace"; storageSource = .application
        self.managedStorageLimitBytes = managedStorageLimitBytes
        self.cpuThreadsSource = cpuThreadsSource; self.maximumResidentBytesSource = maximumResidentBytesSource
    }
    private enum CodingKeys: String, CodingKey {
        case schemaVersion, storageScope, storageSource, managedStorageLimitBytes, cpuThreadsSource, maximumResidentBytesSource
    }
    public init(from decoder: Decoder) throws {
        try requireBudgetFields(decoder, required: ["schemaVersion", "storageScope", "storageSource", "cpuThreadsSource", "maximumResidentBytesSource"], optional: ["managedStorageLimitBytes"])
        let values = try decoder.container(keyedBy: CodingKeys.self)
        guard try values.decode(Int.self, forKey: .schemaVersion) == 1,
              try values.decode(String.self, forKey: .storageScope) == "workspace",
              try values.decode(ResolvedExecutionBudget.Source.self, forKey: .storageSource) == .application else {
            throw ExecutionBudgetError.invalid("Unsupported native admission provenance.")
        }
        try self.init(managedStorageLimitBytes: values.contains(.managedStorageLimitBytes) ? values.decode(Int64.self, forKey: .managedStorageLimitBytes) : nil,
            cpuThreadsSource: values.decode(ResolvedExecutionBudget.Source.self, forKey: .cpuThreadsSource),
            maximumResidentBytesSource: values.decode(ResolvedExecutionBudget.Source.self, forKey: .maximumResidentBytesSource))
    }
}

/// Saved provenance distinguishes the requested setting from a stricter capability ceiling.
public struct ResolvedExecutionBudget: Codable, Equatable {
    public enum Source: String, Codable { case application, project }
    public let schemaVersion: Int
    public let requestedSeconds: Int
    public let maximumRunSeconds: Int
    public let source: Source
    public let capabilityMaximumSeconds: Int?
    /// Nil preserves historical one-thread behavior without rewriting old receipts.
    public let resources: ExecutionResourcePolicy?
    public let provenance: ExecutionPolicyProvenance?

    public static func resolve(global: ExecutionBudgetPreferences, projectMaximumRunSeconds: Int?,
                               capabilityMaximumSeconds: Int? = nil, resources: ExecutionResourcePolicy? = nil,
                               provenance: ExecutionPolicyProvenance? = nil) throws -> Self {
        guard provenance == nil || resources != nil else { throw ExecutionBudgetError.invalid("Resource provenance requires a captured resource policy.") }
        let requested = projectMaximumRunSeconds ?? global.maximumRunSeconds
        try validateExecutionSeconds(requested)
        if let ceiling = capabilityMaximumSeconds { try validateExecutionSeconds(ceiling) }
        return Self(requestedSeconds: requested,
                    source: projectMaximumRunSeconds == nil ? .application : .project,
                    capabilityMaximumSeconds: capabilityMaximumSeconds, resources: resources, provenance: provenance)
    }
    private init(requestedSeconds: Int, source: Source, capabilityMaximumSeconds: Int?, resources: ExecutionResourcePolicy?, provenance: ExecutionPolicyProvenance?) {
        schemaVersion = provenance != nil ? 3 : (resources == nil ? 1 : 2)
        self.provenance = provenance
        self.resources = resources
        self.requestedSeconds = requestedSeconds
        self.source = source
        self.capabilityMaximumSeconds = capabilityMaximumSeconds
        maximumRunSeconds = min(requestedSeconds, capabilityMaximumSeconds ?? requestedSeconds)
    }
    private enum CodingKeys: String, CodingKey {
        case schemaVersion, requestedSeconds, maximumRunSeconds, source, capabilityMaximumSeconds, resources, provenance
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let version = try values.decode(Int.self, forKey: .schemaVersion)
        guard version == 1 || version == 2 || version == 3 else {
            throw ExecutionBudgetError.invalid("Unsupported resolved execution-budget schema.")
        }
        var required: Set<String> = ["schemaVersion", "requestedSeconds", "maximumRunSeconds", "source"]
        if version >= 2 { required.insert("resources") }
        if version == 3 { required.insert("provenance") }
        try requireBudgetFields(decoder, required: required, optional: ["capabilityMaximumSeconds"])
        let resources = try version >= 2 ? values.decode(ExecutionResourcePolicy.self, forKey: .resources) : nil
        let provenance = try version == 3 ? values.decode(ExecutionPolicyProvenance.self, forKey: .provenance) : nil
        let requested = try values.decode(Int.self, forKey: .requestedSeconds)
        let ceiling = try values.decodeIfPresent(Int.self, forKey: .capabilityMaximumSeconds)
        try validateExecutionSeconds(requested)
        if let ceiling { try validateExecutionSeconds(ceiling) }
        self.init(requestedSeconds: requested, source: try values.decode(Source.self, forKey: .source),
                  capabilityMaximumSeconds: ceiling, resources: resources, provenance: provenance)
        guard try values.decode(Int.self, forKey: .maximumRunSeconds) == maximumRunSeconds else {
            throw ExecutionBudgetError.invalid("Effective time budget differs from the captured request and capability ceiling.")
        }
    }
}

public enum ExecutionBudgetError: Error, LocalizedError {
    case invalid(String)
    case wallTimeLimitExceeded(seconds: Int)
    public var errorDescription: String? {
        switch self {
        case .invalid(let reason): return "Execution budget: \(reason)"
        case .wallTimeLimitExceeded(let seconds):
            return "The captured \(seconds)-second operation budget is exhausted. Increase the time budget or reduce the work before starting a new run."
        }
    }
}
private func validateExecutionSeconds(_ value: Int) throws {
    guard ExecutionBudgetPreferences.maximumRunSecondsBounds.contains(value) else {
        throw ExecutionBudgetError.invalid("Choose a whole number of seconds from \(ExecutionBudgetPreferences.maximumRunSecondsBounds.lowerBound) through \(ExecutionBudgetPreferences.maximumRunSecondsBounds.upperBound).")
    }
}
private struct BudgetField: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
private func requireBudgetFields(_ decoder: Decoder, required: Set<String>, optional: Set<String> = []) throws {
    let fields = Set(try decoder.container(keyedBy: BudgetField.self).allKeys.map(\.stringValue))
    guard required.isSubset(of: fields), fields.isSubset(of: required.union(optional)) else {
        throw ExecutionBudgetError.invalid("Unknown or missing time-budget fields.")
    }
}

/// Ephemeral monotonic operation window; never serialize uptime into portable history.
/// Preparation and training must retain the same instance. Each supervisor receives
/// only its remaining allowance, so launching a new phase cannot reset the budget.
public struct ExecutionBudgetWindow {
    public let policy: ResolvedExecutionBudget
    private let startedAtUptime: TimeInterval

    public init(policy: ResolvedExecutionBudget, startedAtUptime: TimeInterval) throws {
        guard startedAtUptime.isFinite, startedAtUptime >= 0 else {
            throw ExecutionBudgetError.invalid("A finite monotonic start time is required.")
        }
        self.policy = policy
        self.startedAtUptime = startedAtUptime
    }
    public func remainingSeconds(atUptime now: TimeInterval) throws -> TimeInterval {
        guard now.isFinite, now >= startedAtUptime else {
            throw ExecutionBudgetError.invalid("Monotonic time moved backwards or is invalid.")
        }
        let remaining = Double(policy.maximumRunSeconds) - (now - startedAtUptime)
        guard remaining > 0 else {
            throw ExecutionBudgetError.wallTimeLimitExceeded(seconds: policy.maximumRunSeconds)
        }
        return remaining
    }
}
