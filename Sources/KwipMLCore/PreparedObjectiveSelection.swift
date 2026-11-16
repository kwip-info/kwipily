import Foundation

/// Shared pins name immutable completed preparation evidence, not embedded tensors.
public struct PreparedDataReference: Codable, Equatable {
    public var manifestPath: String
    public var manifestSHA256: String
    public var graphDigest: String
    public var trainingNode: String
    public var trainingPort: String
    public var validationNode: String
    public var validationPort: String
    public init(manifestPath: String, manifestSHA256: String, graphDigest: String, trainingNode: String,
                trainingPort: String, validationNode: String, validationPort: String) {
        self.manifestPath = manifestPath; self.manifestSHA256 = manifestSHA256; self.graphDigest = graphDigest
        self.trainingNode = trainingNode; self.trainingPort = trainingPort
        self.validationNode = validationNode; self.validationPort = validationPort
    }
    public func validate() throws {
        func bounded(_ value: String, _ maximum: Int) -> Bool { !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && value.count <= maximum }
        func hash(_ value: String) -> Bool { value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) } }
        guard bounded(manifestPath, 16_384), hash(manifestSHA256), hash(graphDigest) else { throw DataFlowContractError("Prepared data requires a manifest path and two 64-character hexadecimal hashes.") }
        guard [trainingNode, trainingPort, validationNode, validationPort].allSatisfy({ bounded($0, 128) }) else { throw DataFlowContractError("Prepared training and validation endpoints must be explicitly named.") }
    }
}

/// Explicit objective transport keeps typed roles separate from the legacy scalar target.
public struct PreparedObjectiveSelection: Codable, Equatable {
    public var namedInputPlan:NamedInputPlan? = nil
    public var schemaVersion: Int = 1
    public var reference: PreparedDataReference
    public var objectiveRecipe: JSONValue
    public var targetMapping: LossTargetMapping
    public var scalarization: WorkflowObjectiveScalarization
    public init(reference: PreparedDataReference, objectiveRecipe: JSONValue, targetMapping: LossTargetMapping,
                scalarization: WorkflowObjectiveScalarization = .mean,namedInputPlan:NamedInputPlan?=nil) {
        self.namedInputPlan=namedInputPlan
        self.reference = reference; self.objectiveRecipe = objectiveRecipe
        self.targetMapping = targetMapping; self.scalarization = scalarization
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, reference, objectiveRecipe, targetMapping, scalarization, namedInputPlan }
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        namedInputPlan=try container.decodeIfPresent(NamedInputPlan.self,forKey:.namedInputPlan)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        reference = try container.decode(PreparedDataReference.self, forKey: .reference)
        let raw = try container.decode(JSONValue.self, forKey: .objectiveRecipe)
        let configuration = try LossConfiguration.decodeRecipe(raw)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard try encoder.encode(raw) == encoder.encode(configuration.recipeValue) else {
            throw DataFlowContractError("Prepared objectives require all canonical configuration fields.")
        }
        objectiveRecipe = configuration.recipeValue
        targetMapping = try container.decode(LossTargetMapping.self, forKey: .targetMapping)
        scalarization = try container.decode(WorkflowObjectiveScalarization.self, forKey: .scalarization)
        try validate()
    }
    public func validate() throws {
        guard schemaVersion == 1 else { throw DataFlowContractError("Unsupported prepared objective selection version.") }
        try namedInputPlan?.validate()
        try reference.validate()
        let configuration = try LossConfiguration.decodeRecipe(objectiveRecipe)
        guard configuration.recipeValue == objectiveRecipe else { throw DataFlowContractError("Prepared objectives require the complete canonical recipe, without implicit defaults.") }
        try configuration.validate(targetMapping: targetMapping)
    }
    public static func decode(_ bytes: Data) throws -> Self {
        guard bytes.count <= 524288 else { throw DataFlowContractError("Prepared objective selection exceeds its transport limit.") }
        let value = try JSONDecoder().decode(JSONValue.self, from: bytes)
        func fields(_ value: JSONValue?, _ names: Set<String>) throws -> [String: JSONValue] {
            guard case .object(let object) = value, Set(object.keys) == names else { throw DataFlowContractError("Prepared objective selection has unknown or incomplete fields.") }
            return object
        }
        let hasNamed:Bool={if case .object(let root)=value {return root["namedInputPlan"] != nil};return false}()
        let root = try fields(value, Set(["schemaVersion","reference","objectiveRecipe","targetMapping","scalarization"]).union(hasNamed ? ["namedInputPlan"]:[]))
        if hasNamed,case .object(let plan)=root["namedInputPlan"] {_=try NamedInputPlan(values:plan)}
        _ = try fields(root["reference"], ["manifestPath","manifestSHA256","graphDigest","trainingNode","trainingPort","validationNode","validationPort"])
        let mapping = try fields(root["targetMapping"], ["schemaVersion","componentID","inputColumns","roles","tripletSourceGroups"])
        guard case .array(let roles) = mapping["roles"] else { throw DataFlowContractError("Prepared objective roles must be an explicit array.") }
        for role in roles { _ = try fields(role, ["role","columns","sampleShape","dtype"]) }
        let decoded = try JSONDecoder().decode(Self.self, from: bytes)
        try decoded.validate()
        return decoded
    }
}
