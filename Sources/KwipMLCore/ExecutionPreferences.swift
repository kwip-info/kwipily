import Foundation

/// Saved application policy. Managed storage is workspace-wide; it cannot be
/// overridden by a project that shares that workspace with other projects.
public struct ExecutionPreferences: Codable, Equatable {
    public let workloadPolicy: ExecutionWorkloadPolicy?
    public let schemaVersion: Int
    public let maximumRunSeconds: Int
    public let cpuThreads: Int
    public let maximumResidentBytes: Int?
    public let managedStorageLimitBytes: Int64?
    public let sampleCapture: TrainingSampleCapturePolicy
    public let derivativeCapture: DerivativeCapturePolicy
    public let preparationCacheLimitBytes: Int
    public static let preparationCacheByteBounds = 1...200_000_000
    public static func preparationCacheByteBounds(workload: ExecutionWorkloadPolicy?) -> ClosedRange<Int> {
        1...(workload.map { Int($0.limit(.retainedBytes)) } ?? preparationCacheByteBounds.upperBound)
    }
    public static let factory = ExecutionPreferences(factory: ())

    private init(factory: Void) {
        workloadPolicy = nil
        derivativeCapture = .factory
        schemaVersion = 3; maximumRunSeconds = 3600; cpuThreads = 1
        maximumResidentBytes = nil; managedStorageLimitBytes = nil; sampleCapture = .factory; preparationCacheLimitBytes = 200_000_000
    }
    public init(maximumRunSeconds: Int = 3600, cpuThreads: Int = 1,
                maximumResidentBytes: Int? = nil, managedStorageLimitBytes: Int64? = nil,
                sampleCapture: TrainingSampleCapturePolicy = .factory, preparationCacheLimitBytes: Int = 200_000_000, derivativeCapture: DerivativeCapturePolicy = .factory, workloadPolicy: ExecutionWorkloadPolicy? = nil) throws {
        _ = try ExecutionBudgetPreferences(maximumRunSeconds: maximumRunSeconds)
        _ = try ExecutionResourcePolicy(cpuThreads: cpuThreads, maximumResidentBytes: maximumResidentBytes)
        if let managedStorageLimitBytes, managedStorageLimitBytes <= 0 {
            throw ExecutionBudgetError.invalid("Choose a positive managed-storage allowance or leave it unlimited.")
        }
        guard Self.preparationCacheByteBounds(workload: workloadPolicy).contains(preparationCacheLimitBytes) else {throw ExecutionBudgetError.invalid("Preparation cache allowance must fit the captured retained-data allowance.")}
        self.preparationCacheLimitBytes = preparationCacheLimitBytes
        self.workloadPolicy = workloadPolicy
        schemaVersion = workloadPolicy == nil ? (derivativeCapture.values ? 4 : 3) : 5; self.maximumRunSeconds = maximumRunSeconds; self.cpuThreads = cpuThreads
        self.maximumResidentBytes = maximumResidentBytes; self.managedStorageLimitBytes = managedStorageLimitBytes
        self.sampleCapture = sampleCapture
        self.derivativeCapture = derivativeCapture
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, maximumRunSeconds, cpuThreads, maximumResidentBytes, managedStorageLimitBytes, sampleCapture, preparationCacheLimitBytes, derivativeCapture, workloadPolicy }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let version = try values.decode(Int.self, forKey: .schemaVersion)
        guard [1, 2, 3, 4, 5].contains(version) else {
            throw ExecutionBudgetError.invalid("Unsupported execution-preferences schema.")
        }
        try executionPreferenceFields(decoder,
            required: Set(["schemaVersion", "maximumRunSeconds", "cpuThreads"]).union(version >= 2 ? ["sampleCapture"] : []).union(version >= 3 ? ["preparationCacheLimitBytes"] : []).union(version >= 4 ? ["derivativeCapture"] : []).union(version == 5 ? ["workloadPolicy"] : []),
            optional: ["maximumResidentBytes", "managedStorageLimitBytes"])
        try self.init(maximumRunSeconds: values.decode(Int.self, forKey: .maximumRunSeconds),
            cpuThreads: values.decode(Int.self, forKey: .cpuThreads),
            maximumResidentBytes: values.contains(.maximumResidentBytes) ? values.decode(Int.self, forKey: .maximumResidentBytes) : nil,
            managedStorageLimitBytes: values.contains(.managedStorageLimitBytes) ? values.decode(Int64.self, forKey: .managedStorageLimitBytes) : nil,
            sampleCapture: version >= 2 ? values.decode(TrainingSampleCapturePolicy.self, forKey: .sampleCapture) : .factory,
            preparationCacheLimitBytes: version >= 3 ? values.decode(Int.self, forKey: .preparationCacheLimitBytes) : 200_000_000,
            derivativeCapture: version >= 4 ? values.decode(DerivativeCapturePolicy.self, forKey: .derivativeCapture) : .factory,
            workloadPolicy: version == 5 ? values.decode(ExecutionWorkloadPolicy.self, forKey: .workloadPolicy) : nil)
    }
    public func encode(to encoder: Encoder) throws {
        var fields = encoder.container(keyedBy: CodingKeys.self)
        try fields.encode(schemaVersion, forKey: .schemaVersion)
        try fields.encode(maximumRunSeconds, forKey: .maximumRunSeconds)
        try fields.encode(cpuThreads, forKey: .cpuThreads)
        try fields.encodeIfPresent(maximumResidentBytes, forKey: .maximumResidentBytes)
        try fields.encodeIfPresent(managedStorageLimitBytes, forKey: .managedStorageLimitBytes)
        try fields.encode(sampleCapture, forKey: .sampleCapture)
        try fields.encode(preparationCacheLimitBytes, forKey: .preparationCacheLimitBytes)
        if schemaVersion >= 4 { try fields.encode(derivativeCapture, forKey: .derivativeCapture) }
        try fields.encodeIfPresent(workloadPolicy, forKey: .workloadPolicy)
    }
    public func replacingDerivativeCapture(_ policy: DerivativeCapturePolicy) throws -> Self {
        try .init(maximumRunSeconds: maximumRunSeconds, cpuThreads: cpuThreads,
            maximumResidentBytes: maximumResidentBytes, managedStorageLimitBytes: managedStorageLimitBytes,
            sampleCapture: sampleCapture, preparationCacheLimitBytes: preparationCacheLimitBytes, derivativeCapture: policy, workloadPolicy: workloadPolicy)
    }
    public func replacingSampleCapture(_ policy: TrainingSampleCapturePolicy) throws -> Self {
        try .init(maximumRunSeconds: maximumRunSeconds, cpuThreads: cpuThreads,
            maximumResidentBytes: maximumResidentBytes, managedStorageLimitBytes: managedStorageLimitBytes, sampleCapture: policy, preparationCacheLimitBytes: preparationCacheLimitBytes, derivativeCapture: derivativeCapture, workloadPolicy: workloadPolicy)
    }
    public func resettingResources() throws -> Self { try .init(sampleCapture: sampleCapture, derivativeCapture: derivativeCapture) }
    public func resolve(project: ProjectExecutionOverrides? = nil, capabilityMaximumSeconds: Int? = nil) throws -> ResolvedExecutionPreferences {
        let memory: Int?
        let memorySource: ResolvedExecutionBudget.Source
        switch project?.memory ?? .inherit {
        case .inherit: memory = maximumResidentBytes; memorySource = .application
        case .unlimited: memory = nil; memorySource = .project
        case .limit(let bytes): memory = bytes; memorySource = .project
        }
        // Newly admitted work must capture its storage contract even when saved
        // preferences predate workload settings. Keep the exact legacy ceilings;
        // nil remains meaningful only when decoding historical run evidence.
        let capturedWorkload: ExecutionWorkloadPolicy
        if let selected = project?.workloadPolicy ?? workloadPolicy { capturedWorkload = selected }
        else { capturedWorkload = try ExecutionWorkloadPolicy(preset: .legacy) }
        let resources = try ExecutionResourcePolicy(cpuThreads: project?.cpuThreads ?? cpuThreads, maximumResidentBytes: memory, workloadPolicy: capturedWorkload)
        let budget = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: maximumRunSeconds),
            projectMaximumRunSeconds: project?.maximumRunSeconds, capabilityMaximumSeconds: capabilityMaximumSeconds, resources: resources,
            provenance: .init(managedStorageLimitBytes: managedStorageLimitBytes,
                cpuThreadsSource: project?.cpuThreads == nil ? .application : .project, maximumResidentBytesSource: memorySource))
        return .init(budget: budget, storageLimitBytes: managedStorageLimitBytes,
            cpuThreadsSource: project?.cpuThreads == nil ? .application : .project, maximumResidentBytesSource: memorySource,
            sampleCapture: project?.sampleCapture ?? sampleCapture, sampleCaptureSource: project?.sampleCapture == nil ? .application : .project, preparationCacheLimitBytes: min(preparationCacheLimitBytes, resources.workloadPolicy.map { Int($0.limit(.retainedBytes)) } ?? Self.preparationCacheByteBounds.upperBound),
            derivativeCapture: project?.derivativeCapture ?? derivativeCapture, derivativeCaptureSource: project?.derivativeCapture == nil ? .application : .project)
    }
}

/// Memory requires three states: inheriting a finite global threshold differs
/// from explicitly requesting no resident-memory threshold in this project.
public struct ProjectExecutionOverrides: Codable, Equatable {
    public enum MemoryOverride: Codable, Equatable {
        case inherit, unlimited, limit(Int)
        private enum CodingKeys: String, CodingKey { case mode, maximumResidentBytes }
        public init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            let mode = try values.decode(String.self, forKey: .mode)
            switch mode {
            case "inherit", "unlimited":
                try executionPreferenceFields(decoder, required: ["mode"])
                self = mode == "inherit" ? .inherit : .unlimited
            case "limit":
                try executionPreferenceFields(decoder, required: ["mode", "maximumResidentBytes"])
                let bytes = try values.decode(Int.self, forKey: .maximumResidentBytes)
                _ = try ExecutionResourcePolicy(cpuThreads: 1, maximumResidentBytes: bytes)
                self = .limit(bytes)
            default: throw ExecutionBudgetError.invalid("Unsupported project memory mode.")
            }
        }
        public func encode(to encoder: Encoder) throws {
            var values = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .inherit: try values.encode("inherit", forKey: .mode)
            case .unlimited: try values.encode("unlimited", forKey: .mode)
            case .limit(let bytes):
                _ = try ExecutionResourcePolicy(cpuThreads: 1, maximumResidentBytes: bytes)
                try values.encode("limit", forKey: .mode)
                try values.encode(bytes, forKey: .maximumResidentBytes)
            }
        }
    }
    public let workloadPolicy: ExecutionWorkloadPolicy?
    public let schemaVersion: Int
    public let maximumRunSeconds: Int?
    public let cpuThreads: Int?
    public let memory: MemoryOverride
    public let sampleCapture: TrainingSampleCapturePolicy?
    public let derivativeCapture: DerivativeCapturePolicy?
    public static let factory = ProjectExecutionOverrides(factory: ())
    private init(factory: Void) { workloadPolicy = nil; derivativeCapture = nil; schemaVersion = 2; maximumRunSeconds = nil; cpuThreads = nil; memory = .inherit; sampleCapture = nil }
    public init(maximumRunSeconds: Int? = nil, cpuThreads: Int? = nil, memory: MemoryOverride = .inherit,
                sampleCapture: TrainingSampleCapturePolicy? = nil, derivativeCapture: DerivativeCapturePolicy? = nil, workloadPolicy: ExecutionWorkloadPolicy? = nil) throws {
        if let maximumRunSeconds { _ = try ExecutionBudgetPreferences(maximumRunSeconds: maximumRunSeconds) }
        if let cpuThreads { _ = try ExecutionResourcePolicy(cpuThreads: cpuThreads) }
        if case .limit(let bytes) = memory { _ = try ExecutionResourcePolicy(cpuThreads: 1, maximumResidentBytes: bytes) }
        self.workloadPolicy = workloadPolicy
        schemaVersion = workloadPolicy == nil ? (derivativeCapture == nil ? 2 : 3) : 4; self.maximumRunSeconds = maximumRunSeconds; self.cpuThreads = cpuThreads; self.memory = memory
        self.sampleCapture = sampleCapture
        self.derivativeCapture = derivativeCapture
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, maximumRunSeconds, cpuThreads, memory, sampleCapture, derivativeCapture, workloadPolicy }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let version = try values.decode(Int.self, forKey: .schemaVersion)
        guard [1, 2, 3, 4].contains(version) else {
            throw ExecutionBudgetError.invalid("Unsupported project execution-preferences schema.")
        }
        try executionPreferenceFields(decoder, required: Set(["schemaVersion", "memory"]).union(version == 3 ? ["derivativeCapture"] : []).union(version == 4 ? ["workloadPolicy"] : []),
            optional: Set(["maximumRunSeconds", "cpuThreads"]).union(version >= 2 ? ["sampleCapture"] : []).union(version == 4 ? ["derivativeCapture"] : []))
        try self.init(maximumRunSeconds: values.contains(.maximumRunSeconds) ? values.decode(Int.self, forKey: .maximumRunSeconds) : nil,
            cpuThreads: values.contains(.cpuThreads) ? values.decode(Int.self, forKey: .cpuThreads) : nil,
            memory: values.decode(MemoryOverride.self, forKey: .memory),
            sampleCapture: values.contains(.sampleCapture) ? values.decode(TrainingSampleCapturePolicy.self, forKey: .sampleCapture) : nil,
            derivativeCapture: values.contains(.derivativeCapture) ? values.decode(DerivativeCapturePolicy.self, forKey: .derivativeCapture) : nil,
            workloadPolicy: version == 4 ? values.decode(ExecutionWorkloadPolicy.self, forKey: .workloadPolicy) : nil)
    }
    public func encode(to encoder: Encoder) throws {
        var fields = encoder.container(keyedBy: CodingKeys.self)
        try fields.encode(schemaVersion, forKey: .schemaVersion)
        try fields.encodeIfPresent(maximumRunSeconds, forKey: .maximumRunSeconds)
        try fields.encodeIfPresent(cpuThreads, forKey: .cpuThreads)
        try fields.encode(memory, forKey: .memory)
        try fields.encodeIfPresent(sampleCapture, forKey: .sampleCapture)
        try fields.encodeIfPresent(derivativeCapture, forKey: .derivativeCapture)
        try fields.encodeIfPresent(workloadPolicy, forKey: .workloadPolicy)
    }
    public func replacingDerivativeCapture(_ policy: DerivativeCapturePolicy?) throws -> Self {
        try .init(maximumRunSeconds: maximumRunSeconds, cpuThreads: cpuThreads, memory: memory, sampleCapture: sampleCapture, derivativeCapture: policy, workloadPolicy: workloadPolicy)
    }
    public func replacingSampleCapture(_ policy: TrainingSampleCapturePolicy?) throws -> Self {
        try .init(maximumRunSeconds: maximumRunSeconds, cpuThreads: cpuThreads, memory: memory, sampleCapture: policy, derivativeCapture: derivativeCapture, workloadPolicy: workloadPolicy)
    }
    public func resettingResources() throws -> Self { try .init(sampleCapture: sampleCapture, derivativeCapture: derivativeCapture) }
}

/// Immutable selection captured by a launch. Time provenance travels in the
/// worker budget; native settings also retain each resource field's source.
public struct ResolvedExecutionPreferences: Equatable {
    public var workloadPolicy: ExecutionWorkloadPolicy? { budget.resources?.workloadPolicy }
    public let budget: ResolvedExecutionBudget
    public let storageLimitBytes: Int64?
    public let cpuThreadsSource: ResolvedExecutionBudget.Source
    public let maximumResidentBytesSource: ResolvedExecutionBudget.Source
    public let derivativeCapture: DerivativeCapturePolicy
    public let derivativeCaptureSource: ResolvedExecutionBudget.Source
    public let sampleCapture: TrainingSampleCapturePolicy
    public let sampleCaptureSource: ResolvedExecutionBudget.Source
    public let preparationCacheLimitBytes: Int
    init(budget: ResolvedExecutionBudget, storageLimitBytes: Int64?, cpuThreadsSource: ResolvedExecutionBudget.Source,
         maximumResidentBytesSource: ResolvedExecutionBudget.Source, sampleCapture: TrainingSampleCapturePolicy,
         sampleCaptureSource: ResolvedExecutionBudget.Source, preparationCacheLimitBytes: Int,
         derivativeCapture: DerivativeCapturePolicy = .factory, derivativeCaptureSource: ResolvedExecutionBudget.Source = .application) {
        self.budget = budget; self.storageLimitBytes = storageLimitBytes
        self.cpuThreadsSource = cpuThreadsSource; self.maximumResidentBytesSource = maximumResidentBytesSource
        self.sampleCapture = sampleCapture; self.sampleCaptureSource = sampleCaptureSource
        self.preparationCacheLimitBytes = preparationCacheLimitBytes
        self.derivativeCapture = derivativeCapture; self.derivativeCaptureSource = derivativeCaptureSource
    }
}

private struct ExecutionPreferenceField: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}
private func executionPreferenceFields(_ decoder: Decoder, required: Set<String>, optional: Set<String> = []) throws {
    let fields = Set(try decoder.container(keyedBy: ExecutionPreferenceField.self).allKeys.map(\.stringValue))
    guard required.isSubset(of: fields), fields.isSubset(of: required.union(optional)) else {
        throw ExecutionBudgetError.invalid("Unknown or missing execution-preference fields.")
    }
}
