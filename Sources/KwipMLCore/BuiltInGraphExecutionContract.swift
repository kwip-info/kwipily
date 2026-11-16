import Foundation

/// Live discovery of the bounded worker adapter, not a restriction on raw PyTorch exports.
public struct BuiltInGraphExecutionContract: Codable, Equatable {
    public struct Tensor: Codable, Equatable {
        public let rank: Int
        public let dtype: String
        public let layout: String
        public let noncontiguous: Bool
        public let minimumBatch: Int
        public let preservesRows: Bool
    }
    public struct Task: Codable, Equatable {
        public let kind: String
        public let inputFeatures: Int
        public let outputFeatures: Int
    }
    public struct Limits: Codable, Equatable {
        public let nodes: Int
        public let edges: Int
        public let parameters: Int
        public let configuredWidth: Int
        public let intermediateWidth: Int
    }
    public struct Operations: Codable, Equatable {
        public let inference: String
        public let training: String
        public let exactResume: String
        public let fineTune: String
        public let export: String
    }
    public let schemaVersion: Int
    public let adapterID: String
    public let runtimeVersion: String
    public let graphSchemaVersions: [Int]
    public let tensor: Tensor
    public let supportedDevices: [ContractDevice]
    public let availableDevices: [ContractDevice]
    public let biasModes: [Bool]
    public let task: Task
    public let limits: Limits
    public let operations: Operations
    public let finiteChecks: [String]
    public let rawModelSemantics: String

    private struct AnyKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.init(stringValue) }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: AnyKey.self)
        func requireKeys(_ actual: [AnyKey], _ expected: [String]) throws {
            guard Set(actual.map(\.stringValue)) == Set(expected) else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Execution discovery fields must exactly match schema 1."))
            }
        }
        try requireKeys(container.allKeys, ["schemaVersion", "adapterID", "runtimeVersion", "graphSchemaVersions", "tensor", "supportedDevices", "availableDevices", "biasModes", "task", "limits", "operations", "finiteChecks", "rawModelSemantics"])
        try requireKeys(container.nestedContainer(keyedBy: AnyKey.self, forKey: AnyKey("tensor")).allKeys, ["rank", "dtype", "layout", "noncontiguous", "minimumBatch", "preservesRows"])
        try requireKeys(container.nestedContainer(keyedBy: AnyKey.self, forKey: AnyKey("task")).allKeys, ["kind", "inputFeatures", "outputFeatures"])
        try requireKeys(container.nestedContainer(keyedBy: AnyKey.self, forKey: AnyKey("limits")).allKeys, ["nodes", "edges", "parameters", "configuredWidth", "intermediateWidth"])
        try requireKeys(container.nestedContainer(keyedBy: AnyKey.self, forKey: AnyKey("operations")).allKeys, ["inference", "training", "exactResume", "fineTune", "export"])
        schemaVersion = try container.decode(Int.self, forKey: AnyKey("schemaVersion"))
        adapterID = try container.decode(String.self, forKey: AnyKey("adapterID"))
        runtimeVersion = try container.decode(String.self, forKey: AnyKey("runtimeVersion"))
        graphSchemaVersions = try container.decode([Int].self, forKey: AnyKey("graphSchemaVersions"))
        tensor = try container.decode(Tensor.self, forKey: AnyKey("tensor"))
        supportedDevices = try container.decode([ContractDevice].self, forKey: AnyKey("supportedDevices"))
        availableDevices = try container.decode([ContractDevice].self, forKey: AnyKey("availableDevices"))
        biasModes = try container.decode([Bool].self, forKey: AnyKey("biasModes"))
        task = try container.decode(Task.self, forKey: AnyKey("task"))
        limits = try container.decode(Limits.self, forKey: AnyKey("limits"))
        operations = try container.decode(Operations.self, forKey: AnyKey("operations"))
        finiteChecks = try container.decode([String].self, forKey: AnyKey("finiteChecks"))
        rawModelSemantics = try container.decode(String.self, forKey: AnyKey("rawModelSemantics"))
    }

    init(schemaVersion: Int, adapterID: String, runtimeVersion: String, graphSchemaVersions: [Int], tensor: Tensor, supportedDevices: [ContractDevice], availableDevices: [ContractDevice], biasModes: [Bool], task: Task, limits: Limits, operations: Operations, finiteChecks: [String], rawModelSemantics: String) {
        self.schemaVersion = schemaVersion
        self.adapterID = adapterID
        self.runtimeVersion = runtimeVersion
        self.graphSchemaVersions = graphSchemaVersions
        self.tensor = tensor
        self.supportedDevices = supportedDevices
        self.availableDevices = availableDevices
        self.biasModes = biasModes
        self.task = task
        self.limits = limits
        self.operations = operations
        self.finiteChecks = finiteChecks
        self.rawModelSemantics = rawModelSemantics
    }

    /// A mismatching live worker must not establish execution readiness.
    public func validate(registry: ComponentRegistry, runtimeVersion: String,
                         availableDevices: [ContractDevice]) throws {
        let expected = try BuiltInCapabilityBridge.executionContract(
            registry: registry, runtimeVersion: runtimeVersion, availableDevices: availableDevices)
        guard self == expected else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry(
                "The worker execution contract differs from this application's supported tensor, task, lifecycle, device, or runtime contract.")
        }
    }
}

extension BuiltInCapabilityBridge {
    public static func executionContract(registry: ComponentRegistry, runtimeVersion: String,
                                         availableDevices: [ContractDevice]) throws -> BuiltInGraphExecutionContract {
        // Reuse registry and exact runtime validation rather than inventing a second catalog.
        _ = try package(registry: registry, runtimeVersion: runtimeVersion)
        guard Set(availableDevices).count == availableDevices.count,
              availableDevices.contains(.cpu),
              availableDevices.allSatisfy({ $0 == .cpu || $0 == .mps }) else {
            throw BuiltInCapabilityBridgeError.incompatibleRegistry(
                "Worker discovery must report CPU and only available supported devices, without duplicates.")
        }
        return BuiltInGraphExecutionContract(
            schemaVersion: 1, adapterID: "kwip.runner.product-execution.v1",
            runtimeVersion: runtimeVersion, graphSchemaVersions: [1, 2],
            tensor: .init(rank: 2, dtype: "float32", layout: "denseStrided", noncontiguous: true,
                          minimumBatch: 1, preservesRows: true),
            supportedDevices: [.cpu, .mps],
            availableDevices: [.cpu, .mps].filter { availableDevices.contains($0) },
            biasModes: [false, true],
            task: .init(kind: "binaryClassification", inputFeatures: 2, outputFeatures: 2),
            limits: .init(nodes: 128, edges: 256, parameters: 2_000_000, configuredWidth: 512,
                          intermediateWidth: 4096),
            operations: .init(inference: "boundedGraph", training: "experimentOwnedAdam",
                              exactResume: "cpuEpochBoundary", fineTune: "weightsOnlyFreshOptimizer",
                              export: "pythonSourceRawTorch"),
            finiteChecks: ["sourcePartitions", "initialParameters", "publicForwardInputsOutputs",
                           "inspectedStages", "validationLogitsAndLoss", "trainingLoss"],
            rawModelSemantics: "pytorch")
    }
}
