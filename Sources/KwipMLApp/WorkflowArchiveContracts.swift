import Foundation
import CryptoKit
import KwipMLCore

struct WorkflowArchiveFile: Codable, Equatable {
    let path: String
    let bytes: Int
    let sha256: String
}
struct WorkflowArchiveSource: Codable, Equatable {
    let address: WorkflowNodeAddress
    let path: String
    let bytes: Int
    let sha256: String
    /// Nil is the historical scalar source. An empty array is a metadata-only collection.
    var collectionMembers: [WorkflowArchiveFile]? = nil
}
struct WorkflowArchiveOrigin: Codable, Equatable {
    let nodeID: String
    let origin: CompositeNodeOrigin
}
/// Canonical optimizer settings and qualified canvas origin, captured only by compiler 5.
struct WorkflowArchiveOptimizerRecipe: Codable, Equatable {
    let source: WorkflowNodeAddress
    let configuration: JSONValue

    init(_ recipe: OptimizerRecipe) throws {
        source = recipe.source
        configuration = recipe.configuration.recipeValue
        try validate()
    }
    private enum CodingKeys: String, CodingKey { case source, configuration }
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        source = try container.decode(WorkflowNodeAddress.self, forKey: .source)
        let raw = try container.decode(JSONValue.self, forKey: .configuration)
        let parsed = try OptimizerConfiguration.decodeRecipe(raw)
        guard case .object(let envelope) = raw, case .object(let controls) = envelope["config"],
              Set(controls.keys) == Set(parsed.values.keys) else {
            throw WorkflowArchiveError("Captured optimizer configuration requires every declared control.")
        }
        // JSON integers and decimal numbers can encode identically; retain canonical semantics.
        configuration = parsed.recipeValue
        try validate()
    }
    func validate() throws {
        guard source.domain == .control, !source.localID.isEmpty else {
            throw WorkflowArchiveError("Captured optimizer requires a qualified control source.")
        }
        _ = try OptimizerConfiguration.decodeRecipe(configuration)
    }
}
/// Canonical scheduler settings and qualified canvas origin, captured only by compiler 6.
struct WorkflowArchiveSchedulerRecipe: Codable, Equatable {
    let source: WorkflowNodeAddress
    let configuration: JSONValue

    init(_ recipe: SchedulerRecipe) throws {
        source = recipe.source
        configuration = recipe.configuration.recipeValue
        try validate()
    }
    private enum CodingKeys: String, CodingKey { case source, configuration }
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        source = try container.decode(WorkflowNodeAddress.self, forKey: .source)
        let raw = try container.decode(JSONValue.self, forKey: .configuration)
        let parsed = try SchedulerConfiguration.decodeRecipe(raw)
        guard case .object(let envelope) = raw, case .object(let controls) = envelope["config"],
              Set(controls.keys) == Set(parsed.values.keys) else {
            throw WorkflowArchiveError("Captured scheduler configuration requires every declared control.")
        }
        // JSON integers and decimal numbers can encode identically; retain canonical semantics.
        configuration = parsed.recipeValue
        try validate()
    }
    func validate() throws {
        guard source.domain == .control, !source.localID.isEmpty else {
            throw WorkflowArchiveError("Captured scheduler requires a qualified control source.")
        }
        _ = try SchedulerConfiguration.decodeRecipe(configuration)
    }
}
/// Canonical selected scoring recipe, qualified source and typed column roles; compiler 7 only.
struct WorkflowArchiveLossRecipe: Codable, Equatable {
    let source: WorkflowNodeAddress
    let configuration: JSONValue
    let targetMapping: LossTargetMapping

    init(_ recipe: LossRecipe) throws {
        source=recipe.source;configuration=recipe.configuration.recipeValue;targetMapping=recipe.targetMapping
        try validate()
    }
    private enum CodingKeys: String, CodingKey {case source,configuration,targetMapping}
    init(from decoder: Decoder) throws {
        let raw=try JSONValue(from:decoder)
        guard case .object(let object)=raw,Set(object.keys) == ["source","configuration","targetMapping"],
              case .object(let sourceObject)=object["source"],Set(sourceObject.keys) == ["domain","localID"],
              case .object(let mappingObject)=object["targetMapping"],Set(mappingObject.keys) == ["schemaVersion","componentID","inputColumns","roles","tripletSourceGroups"],
              case .array(let roles)=mappingObject["roles"],roles.allSatisfy({if case .object(let role)=$0 {return Set(role.keys) == ["role","columns","sampleShape","dtype"]};return false}),
              let rawConfiguration=object["configuration"] else {
            throw WorkflowArchiveError("Captured objective requires exact source, configuration and typed target mapping fields.")
        }
        let parsed=try LossConfiguration.decodeRecipe(rawConfiguration)
        guard case .object(let envelope)=rawConfiguration,case .object(let controls)=envelope["config"],Set(controls.keys) == Set(parsed.values.keys) else {
            throw WorkflowArchiveError("Captured objective configuration requires every declared control.")
        }
        source=try JSONDecoder().decode(WorkflowNodeAddress.self,from:WorkflowArchiveCodec.encode(object["source"]!))
        targetMapping=try JSONDecoder().decode(LossTargetMapping.self,from:WorkflowArchiveCodec.encode(object["targetMapping"]!))
        configuration=parsed.recipeValue
        try validate()
    }
    func validate() throws {
        guard source.domain == .control,!source.localID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,source.localID.utf8.count<=128 else {
            throw WorkflowArchiveError("Captured objective requires a bounded qualified control source.")
        }
        _ = try LossRecipe(configuration:LossConfiguration.decodeRecipe(configuration),source:source,targetMapping:targetMapping)
    }
}
/// Canonical starting-value policy and qualified source, captured only by compiler8.
struct WorkflowArchiveInitializationRecipe: Codable, Equatable {
    let source: WorkflowNodeAddress
    let configuration: JSONValue
    init(_ recipe: InitializationRecipe) throws {
        source = recipe.source
        configuration = recipe.configuration.recipeValue
        try validate()
    }
    init(from decoder: Decoder) throws {
        let raw = try JSONValue(from: decoder)
        guard case .object(let object) = raw, Set(object.keys) == ["source", "configuration"],
              case .object(let address) = object["source"], Set(address.keys) == ["domain", "localID"],
              let controls = object["configuration"] else {
            throw WorkflowArchiveError("Captured starting settings require exact source and configuration fields.")
        }
        let parsed = try InitializationConfiguration.decodeRecipe(controls)
        guard try WorkflowArchiveCodec.encode(controls) == WorkflowArchiveCodec.encode(parsed.recipeValue) else {
            throw WorkflowArchiveError("Captured starting settings must use canonical complete controls and owner order.")
        }
        source = try JSONDecoder().decode(WorkflowNodeAddress.self, from: WorkflowArchiveCodec.encode(object["source"]!))
        configuration = parsed.recipeValue
        try validate()
    }
    func validate() throws {
        guard source.domain == .control, !source.localID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              source.localID.utf8.count <= 128 else {
            throw WorkflowArchiveError("Captured starting settings require a bounded qualified control source.")
        }
        _ = try InitializationConfiguration.decodeRecipe(configuration)
    }
}
/// Derived immutable execution evidence; never an authoring store.
struct WorkflowArchivePlan: Codable, Equatable {
    let projectID: String
    let revision: UInt64
    let train: WorkflowNodeAddress
    let dataflow: DataFlowDocument
    let sourceAddresses: [WorkflowNodeAddress]
    let modelGraph: GraphDocument
    let training: DataFlowEndpoint
    let validation: DataFlowEndpoint
    let featureColumns: [String]
    let targetColumn: String?
    /// Compiler11 packet mapping; absent historical/dense projections retain their bytes.
    let namedInputPlan:NamedInputPlan?
    let bagInput: BagInputMapping?
    let settings: ExecutionSettings
    let modelSemanticSHA256: String
    let modelOrigins: [WorkflowArchiveOrigin]
    let modelProvenance: CompositeSourceSnapshot?
    let modelAuthoringRevision: UInt64?
    /// Omitted for compiler 1 to preserve its canonical archived bytes.
    let preparationCachePolicy: WorkflowPreparationCachePolicy?
    /// Present only in compiler3 captures; omitted from historical canonical bytes.
    let trainingMode: WorkflowTrainingMode?

    /// Omitted from historical captures and implicit optimizer runs.
    let optimizerRecipe: WorkflowArchiveOptimizerRecipe?
    /// Omitted before compiler6 and from all runs without a selected scheduler.
    let schedulerRecipe: WorkflowArchiveSchedulerRecipe?
    /// Omitted before compiler 7 and from implicit objective runs.
    let lossRecipe: WorkflowArchiveLossRecipe?
    let objectiveScalarization: WorkflowObjectiveScalarization?
    /// Omitted from historical and unselected-policy captures.
    let initializationRecipe: WorkflowArchiveInitializationRecipe?
    /// Compiler9 only; absent historical projections preserve their exact bytes.
    let gradientPolicyRecipe: WorkflowArchiveGradientPolicyRecipe?
    let trainabilityRecipe: WorkflowArchiveTrainabilityRecipe?
    let datasetRecipe:WorkflowDataLoadingArchiveRecipe?
    let loaderRecipe:WorkflowDataLoadingArchiveRecipe?
    let inlineDataLoadingRoles:[String]?
    let inlineTrainingRecipeRoles:[String]?

    init(_ plan: WorkflowExecutionPlan, compilerContractVersion: Int) throws {
        guard [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].contains(compilerContractVersion), (plan.inlineTrainingRecipeRoles.isEmpty || compilerContractVersion >= 12), (compilerContractVersion>=12 || (plan.datasetRecipe == nil && plan.loaderRecipe == nil)), (compilerContractVersion >= 11 || plan.bagInput == nil), (compilerContractVersion >= 9 || (plan.gradientPolicyRecipe == nil && plan.trainabilityRecipe == nil)), (compilerContractVersion >= 8 || plan.initializationRecipe == nil), (compilerContractVersion >= 7 || plan.lossRecipe == nil), (compilerContractVersion >= 6 || plan.schedulerRecipe == nil), (plan.schedulerRecipe == nil || plan.optimizerRecipe != nil), (compilerContractVersion >= 5 || plan.optimizerRecipe == nil), (compilerContractVersion >= 3 || plan.trainingMode == .fresh), compilerContractVersion != 1 || plan.preparationCachePolicy == .off else {
            throw WorkflowArchiveError("Unsupported compiler contract or cache policy for legacy archive projection.")
        }
        if let recipe=plan.lossRecipe {
            guard plan.targetColumn == nil,plan.featureColumns == recipe.targetMapping.inputColumns,
                  plan.inlineTrainingRecipeRoles.contains("objective") || recipe.source != plan.train else {
                throw WorkflowArchiveError("Selected objective requires its exact input mapping and distinct qualified source, without a dummy target column.")
            }
        } else {
            guard let target=plan.targetColumn,!target.isEmpty,target.utf8.count<=4096 else {throw WorkflowArchiveError("Implicit objectives require their historical target column.")}
        }
        inlineTrainingRecipeRoles=plan.inlineTrainingRecipeRoles.isEmpty ? nil:plan.inlineTrainingRecipeRoles.sorted()
        inlineDataLoadingRoles=plan.inlineDataLoadingRoles.isEmpty ? nil:plan.inlineDataLoadingRoles.sorted()
        datasetRecipe=try plan.datasetRecipe.map {try WorkflowDataLoadingArchiveRecipe($0)}
        loaderRecipe=try plan.loaderRecipe.map {try WorkflowDataLoadingArchiveRecipe($0)}
        gradientPolicyRecipe = try plan.gradientPolicyRecipe.map {try WorkflowArchiveGradientPolicyRecipe($0)}
        trainabilityRecipe = try plan.trainabilityRecipe.map {try WorkflowArchiveTrainabilityRecipe($0)}
        initializationRecipe = try plan.initializationRecipe.map { try WorkflowArchiveInitializationRecipe($0) }
        if let recipe = initializationRecipe {
            let configuration = try InitializationConfiguration.decodeRecipe(recipe.configuration)
            guard configuration.rules.allSatisfy({ owner in plan.model.graph.nodes.contains { $0.id == owner.owner.node } }) else {
                throw WorkflowArchiveError("Starting settings select an owner outside the captured model.")
            }
        }
        lossRecipe=try plan.lossRecipe.map {try WorkflowArchiveLossRecipe($0)}
        objectiveScalarization=plan.lossRecipe == nil ? nil:plan.objectiveScalarization
        optimizerRecipe = try plan.optimizerRecipe.map { try WorkflowArchiveOptimizerRecipe($0) }
        schedulerRecipe = try plan.schedulerRecipe.map { try WorkflowArchiveSchedulerRecipe($0) }
        preparationCachePolicy = compilerContractVersion == 1 ? nil : plan.preparationCachePolicy
        trainingMode = compilerContractVersion >= 3 ? plan.trainingMode : nil
        projectID = plan.projectID; revision = plan.revision; train = plan.train
        dataflow = plan.dataflow; sourceAddresses = plan.sourceAddresses.sorted { $0.localID < $1.localID }
        modelGraph = plan.model.graph; training = plan.training; validation = plan.validation
        namedInputPlan=plan.namedInputPlan
        featureColumns = plan.featureColumns; targetColumn = plan.targetColumn; bagInput = plan.bagInput; settings = plan.settings
        modelSemanticSHA256 = plan.model.semanticSHA256
        modelOrigins = plan.model.origins.keys.sorted().map { .init(nodeID: $0, origin: plan.model.origins[$0]!) }
        modelProvenance = plan.model.provenance; modelAuthoringRevision = plan.model.authoringRevision
        let otherSources = [Optional(train), optimizerRecipe?.source, schedulerRecipe?.source, lossRecipe?.source, initializationRecipe?.source].compactMap {$0}
        let reduction = try lossRecipe.map {try LossConfiguration.decodeRecipe($0.configuration).values["reduction"]!.stringValue!} ?? "mean"
        try GradientArchiveRecipes.validate(policy:gradientPolicyRecipe,trainability:trainabilityRecipe,graph:modelGraph,origins:modelOrigins,otherSources:otherSources,objectiveReduction:reduction,scalarization:objectiveScalarization?.rawValue ?? "mean", inlineTrain:train, inlineRoles:Set(inlineTrainingRecipeRoles ?? []))
        try validateBagMapping()
        try validateTrainingRecipeSources()
        try validateDataLoadingSources(train:plan.train)
    }
    func validateTrainingRecipeSources() throws {
        let sources:[String:WorkflowNodeAddress?] = ["optimizer":optimizerRecipe?.source,"scheduler":schedulerRecipe?.source,"objective":lossRecipe?.source,"initialization":initializationRecipe?.source,"gradientPolicy":gradientPolicyRecipe?.source,"trainability":trainabilityRecipe?.source]
        let inline=Set(inlineTrainingRecipeRoles ?? [])
        guard inlineTrainingRecipeRoles == nil || (!inline.isEmpty && inline.count == inlineTrainingRecipeRoles?.count),
              inline.isSubset(of:Set(sources.keys)) else {throw WorkflowArchiveError("Inline training roles must be explicit, supported and unique.")}
        var foreign=Set<WorkflowNodeAddress>()
        for (role,source) in sources {
            if inline.contains(role) {
                guard let source,source == train else {throw WorkflowArchiveError("Inline training controls must retain their exact captured Train origin.")}
            } else if let source {
                guard source != train,foreign.insert(source).inserted else {throw WorkflowArchiveError("Unmarked training recipes require distinct qualified sources outside Train.")}
            }
        }
    }
    func validateDataLoadingSources(train:WorkflowNodeAddress)throws {
        try datasetRecipe?.validate(componentID:DatasetConfiguration.componentID)
        try loaderRecipe?.validate(componentID:DataLoaderConfiguration.componentID)
        let selected=[datasetRecipe?.source,loaderRecipe?.source].compactMap {$0}
        let others=[train,optimizerRecipe?.source,schedulerRecipe?.source,lossRecipe?.source,initializationRecipe?.source,gradientPolicyRecipe?.source,trainabilityRecipe?.source].compactMap {$0}
        let inline=Set(inlineDataLoadingRoles ?? [])
        guard inlineDataLoadingRoles == nil || (!inline.isEmpty && inline.count == inlineDataLoadingRoles?.count) else {throw WorkflowArchiveError("Inline reader roles must be explicit and unique.")}
        try WorkflowDataLoadingVerifier.validateSources(dataset:datasetRecipe?.source,loader:loaderRecipe?.source,trainOrigin:train,inlineRoles:inline)
        guard selected.filter({$0 != train}).allSatisfy({!others.contains($0)}) else {throw WorkflowArchiveError("Reader policies require distinct sources from other selected recipes.")}
        if let loaderRecipe {let config=try DataLoaderConfiguration.decodeRecipe(loaderRecipe.configuration);guard settings.batchSize==config.batchSize else {throw WorkflowArchiveError("Captured batch size differs from its selected loader.")}}
    }
    func matches(_ plan: WorkflowExecutionPlan, compilerContractVersion: Int) throws -> Bool {
        let expected=try Self(plan,compilerContractVersion:compilerContractVersion)
        // Durable captured JSON may decode an integral floating value as integer.
        // Keep every source/plan field exact in its finite serialized representation.
        return try WorkflowArchiveCodec.encode(self)==WorkflowArchiveCodec.encode(expected)
    }
}
struct WorkflowArchiveCacheReference: Codable, Equatable {
    let projectID: String
    let directoryID: String
    let preparationManifestSHA256: String
}
extension WorkflowArchivePlan {
    /// Envelope validation can bind source controls and columns without borrowing
    /// today's registry. Full role/consumer admission uses the captured registry below.
    func validateBagMapping() throws {
        try namedInputPlan?.validate()
        guard namedInputPlan == nil || bagInput == nil else {throw WorkflowArchiveError("Named and bag mappings cannot share one training request.")}
        if let namedInputPlan {
            if let names=modelGraph.namedInputs {guard names==Dictionary(uniqueKeysWithValues:namedInputPlan.inputs.map {($0.name,$0.nodeID!)}) else {throw WorkflowArchiveError("Named mapping differs from captured model boundaries.")}}
        }
        let sources=modelGraph.nodes.filter {["input","index_input","bag_input"].contains($0.type)}
        guard let bagInput else {
            guard !sources.contains(where:{$0.type == "bag_input"}) else {throw WorkflowArchiveError("Captured bag source is missing its packet mapping.")}
            return
        }
        guard modelGraph.schemaVersion == 2,sources.count == 1,let source=sources.first,source.type == "bag_input",
              featureColumns == bagInput.indicesColumns else {throw WorkflowArchiveError("Packet mapping must match one actual schema2 bag source and its ordered item columns.")}
        try bagInput.validate(sourceConfiguration:BagSourceConfiguration(values:source.config),targetColumn:targetColumn,objectiveMapping:lossRecipe?.targetMapping)
    }
    /// Reproduce captured source against the captured contracts, never today's registry.
    func validateModelSnapshot(registry: ComponentRegistry, context: GraphValidationContext) throws {
        guard Set(modelOrigins.map(\.nodeID)).count == modelOrigins.count else { throw WorkflowArchiveError("Captured model origins must be unique.") }
        if let source = modelProvenance {
            let expansion = try source.verifiedExpansion(registry: registry, context: context)
            let sameGraph=try WorkflowArchiveCodec.encode(expansion.graph)==WorkflowArchiveCodec.encode(modelGraph)
            guard sameGraph,
                  expansion.nodeOrigins == Dictionary(uniqueKeysWithValues: modelOrigins.map { ($0.nodeID, $0.origin) }) else {
                throw WorkflowArchiveError("Captured model differs from its independently expanded source and ownership.")
            }
        } else if modelGraph.compositionPlan != nil {
            throw WorkflowArchiveError("Native composition requires its captured source snapshot.")
        }
        try validateBagMapping()
        if let bagInput {
            let requirement=try WorkflowMappingRequirements.bagInputRequirement(modelGraph,registry:registry,context:context)
            try bagInput.validate(sourceConfiguration:requirement.configuration,targetColumn:targetColumn,objectiveMapping:lossRecipe?.targetMapping)
        }
        guard try WorkflowArchiveCodec.hash(GraphSemantics.canonicalData(modelGraph, registry: registry)) == modelSemanticSHA256 else {
            throw WorkflowArchiveError("Captured model semantic identity differs from the captured contracts.")
        }
    }
}

struct WorkflowArchiveCapture: Codable, Equatable {
    var schemaVersion: Int = 1
    var compilerContractVersion: Int = 1
    let projectID: String
    let directoryID: String
    let runID: String
    let requestIdentity: String
    let revision: UInt64
    let train: WorkflowNodeAddress
    let workflow: WorkflowArchiveFile
    let modelRegistry: WorkflowArchiveFile
    let dataRegistry: WorkflowArchiveFile
    let modelRegistrySHA256: String
    let dataRegistrySHA256: String
    let runtimeEvidence: WorkflowArchiveFile
    let sources: [WorkflowArchiveSource]
    let plan: WorkflowArchivePlan
    var priorCache: WorkflowArchiveCacheReference? = nil
    var checkpointParent: WorkflowCheckpointReference? = nil
    var checkpointFiles: [WorkflowArchiveFile]? = nil
    /// Independent capture envelope version; authoring/compiler versions stay unchanged.
    /// Optional provenance is omitted for historical captures to preserve canonical bytes.
    var captureFormatVersion: Int? = nil
    var executionBudget: ResolvedExecutionBudget? = nil
    var sampleCapture: TrainingSampleCapturePolicy? = nil
    var preparationCacheLimitBytes: Int? = nil
    var derivativeCapture: DerivativeCapturePolicy? = nil
    var moduleCompositionDescriptor: JSONValue? = nil
}
enum WorkflowFailureCode: String, Codable { case wallTimeLimitExceeded, resourceBudgetFailed }
struct WorkflowArchiveOutcome: Codable, Equatable {
    var schemaVersion: Int = 2
    let projectID: String
    let directoryID: String
    let runID: String
    let requestIdentity: String
    let status: WorkflowExecutionPhase
    let terminalPhase: WorkflowExecutionPhase
    let startedAt: String
    let finishedAt: String
    let artifacts: [WorkflowArchiveFile]
    var capture: WorkflowArchiveFile? = nil
    var preparationManifestSHA256: String? = nil
    var modelManifestSHA256: String? = nil
    var workerRunID: String? = nil
    var error: String? = nil
    var failureCode: WorkflowFailureCode? = nil
    var failedGradientEvidence: WorkflowFailedGradientDescriptor? = nil
    var failedGradientNotice: String? = nil
}
struct WorkflowArchiveError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

enum WorkflowArchiveCodec {
    /// Authoring schema5 adds evaluation while retaining compiler4; schema6/7
    /// add optimizer/scheduler recipes under compiler5/6, and schema8 adds objectives under compiler7.
    /// Legacy archive pairs stay exact, including schema 1.
    /// Unknown versions remain unknown and are rejected by archive/compiler validation.
    static func trainingCompilerMatches(_ compiler:Int,schema:Int)->Bool {
        compiler == trainingCompilerContractVersion(forWorkflowSchema:schema) || (compiler == 16 && [13,14,15,16,17,18,19,20,21,22].contains(schema))
    }
    static func trainingCompilerContractVersion(forWorkflowSchema version:Int)->Int {
        // Schema14 adds authoring ownership only; its executable model contract
        // and captured reader recipes remain the reviewed compiler12 contract.
        if [17,18,19,20,21,22].contains(version) {return 15}
        if version == 16 {return 14}
        if version == 15 {return 13}
        if [13,14].contains(version) {return 12}
        return version == 12 ? 11 : (version == 11 ? 10 : (version == 10 ? 9 : (version == 9 ? 8 : (version == 8 ? 7 : (version == 7 ? 6 : (version == 6 ? 5 : (version == 5 ? 4 : version)))))))
    }
    static func timestamp(_ date: Date) -> String { formatter().string(from: date) }
    private static func formatter() -> ISO8601DateFormatter {
        let value = ISO8601DateFormatter()
        value.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        value.timeZone = TimeZone(secondsFromGMT: 0)
        return value
    }
    static func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }
    /// Existing request convention intentionally differs from exact archive file bytes.
    static func registryHash<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return hash(try encoder.encode(value))
    }
    static func encodeCapture(_ value: WorkflowArchiveCapture) throws -> Data {
        try validate(value); let bytes = try encode(value)
        try require(bytes.count <= 8_000_000, "Capture exceeds 8 MB."); return bytes
    }
    static func decodeCompositeSource(_ bytes: Data) throws -> CompositeSourceSnapshot {
        try decode(bytes, limit: 2_000_000)
    }
    static func decodeCapture(_ bytes: Data) throws -> WorkflowArchiveCapture {
        let value: WorkflowArchiveCapture = try decode(bytes, limit: 8_000_000)
        try validate(value); return value
    }
    static func encodeOutcome(_ value: WorkflowArchiveOutcome) throws -> Data {
        try validate(value); let bytes = try encode(value)
        try require(bytes.count <= 256_000, "Outcome exceeds 256 KB."); return bytes
    }
    static func decodeOutcome(_ bytes: Data) throws -> WorkflowArchiveOutcome {
        let value: WorkflowArchiveOutcome = try decode(bytes, limit: 256_000)
        try validate(value); return value
    }
    private static func decode<T: Codable>(_ bytes: Data, limit: Int) throws -> T {
        try require(!bytes.isEmpty && bytes.count <= limit, "Archive exceeds its byte limit.")
        let value = try JSONDecoder().decode(T.self, from: bytes)
        let original = try JSONSerialization.jsonObject(with: bytes)
        let normalized = try JSONSerialization.jsonObject(with: encode(value))
        try checkKeys(original, normalized, depth: 0)
        return value
    }
    /// Codable ignores unknown keys; recursively reject those rather than losing evidence.
    /// Optional nulls are rejected where the normalized schema omits the field.
    private static func checkKeys(_ original: Any, _ normalized: Any, depth: Int) throws {
        try require(depth <= 64, "Archive nesting exceeds its bound.")
        if let a = original as? [String: Any], let b = normalized as? [String: Any] {
            try require(Set(a.keys).isSubset(of: Set(b.keys)), "Archive contains unknown or null optional fields.")
            for (key, value) in a { if let other = b[key] { try checkKeys(value, other, depth: depth + 1) } }
        } else if let a = original as? [Any], let b = normalized as? [Any] {
            try require(a.count == b.count, "Archive array changed during decoding.")
            for (left, right) in zip(a, b) { try checkKeys(left, right, depth: depth + 1) }
        }
    }
    static func isHash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else { throw WorkflowArchiveError(message) }
    }
    private static func identity(_ project: String, _ directory: String, _ run: String, _ request: String) throws {
        try require([project, directory, run].allSatisfy { UUID(uuidString: $0) != nil } && isHash(request), "Invalid archive identity.")
    }
    static func pathParts(_ path: String) throws -> [String] {
        let parts = path.components(separatedBy: "/")
        try require(path.utf8.count <= 1024 && !path.contains("\0") && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }, "Unsafe archive path.")
        return parts
    }
    private static func file(_ value: WorkflowArchiveFile, path: String, limit: Int) throws {
        _ = try pathParts(value.path)
        try require(value.path == path && (0...limit).contains(value.bytes) && isHash(value.sha256), "Invalid archive file reference.")
    }
    /// History replay runs on bounded cooperative-worker stacks. Keep each logical
    /// validation section in a separate frame; no checks or archive authority change.
    @inline(never)
    static func validate(_ value: WorkflowArchiveCapture) throws {
        try validateCaptureFormat(value)
        try validateCaptureIdentityAndModel(value)
        try validateCaptureDataLoading(value)
        try validateCaptureGradientAndInitialization(value)
        try validateCaptureOptimizationAndObjective(value)
        try validateCaptureLifecycle(value)
        try validateCaptureCache(value)
        try validateCaptureFilesAndSources(value)
        try validateCapturePlanBounds(value)
    }
    @inline(never)
    private static func validateCaptureDataLoading(_ value:WorkflowArchiveCapture)throws {
        try require(value.compilerContractVersion>=12 || (value.plan.datasetRecipe == nil && value.plan.loaderRecipe == nil),"Historical captures cannot acquire reader policies.")
        try require(value.plan.inlineTrainingRecipeRoles == nil || (value.compilerContractVersion >= 12 && value.schemaVersion >= 13), "Inline training recipes require their current capture contract.")
        try value.plan.validateTrainingRecipeSources()
        try value.plan.validateDataLoadingSources(train:value.train)
    }
    @inline(never)
    private static func validateCaptureFormat(_ value: WorkflowArchiveCapture) throws {
        if let selected = value.plan.gradientPolicyRecipe {
            guard [5,6,7,8].contains(value.captureFormatVersion ?? 0), value.executionBudget != nil, let permission = value.derivativeCapture else {
                throw WorkflowArchiveError("Selected gradient policies require capture format5, an explicit derivative permission and an execution budget.")
            }
            try permission.validate(gradientPolicy:GradientPolicyConfiguration.decodeRecipe(selected.configuration))
        } else {
            try require(value.derivativeCapture == nil,"Unselected gradient policies cannot carry derivative permission.")
        }
        if value.captureFormatVersion != 5 && value.captureFormatVersion != 6 && value.captureFormatVersion != 7 && value.captureFormatVersion != 8 {
            try require(value.derivativeCapture == nil,"Older capture formats cannot carry derivative permission.")
        }
        if value.captureFormatVersion == 6 || value.captureFormatVersion == 7 || value.captureFormatVersion == 8 {
            guard let descriptor = value.moduleCompositionDescriptor else { throw WorkflowArchiveError("Capture6/7 requires its exact verified native module composition descriptor.") }
            try ModuleCompositionRuntimeCapability.verify(encode(descriptor))
        } else {
            try require(value.moduleCompositionDescriptor == nil, "Older captures cannot acquire native composition descriptor evidence.")
        }
        switch value.captureFormatVersion {
        case 6,7,8:
            try require(value.captureFormatVersion == 8 ? ([13,14,15,16,17,18,19,20,21,22].contains(value.schemaVersion) && trainingCompilerMatches(value.compilerContractVersion,schema:value.schemaVersion)) : value.captureFormatVersion == 7 ? (value.schemaVersion == 12 && value.compilerContractVersion == 11) : (value.schemaVersion == 11 && value.compilerContractVersion == 10),
                "Capture format6 requires workflow11/compiler10; format7 requires workflow12/compiler11.")
            try require(value.sampleCapture == nil || value.executionBudget != nil,
                "Explicit sample capture requires execution-budget provenance.")
            if let limit = value.preparationCacheLimitBytes {
                try require(ExecutionPreferences.preparationCacheByteBounds.contains(limit) &&
                    value.plan.preparationCachePolicy != nil && value.plan.preparationCachePolicy != .off,
                    "Invalid captured preparation cache allowance in format6/7.")
            }
        case 5:
            try require(value.executionBudget != nil && value.plan.gradientPolicyRecipe != nil && value.derivativeCapture != nil,
                "Capture format5 requires a selected gradient policy, execution budget and explicit derivative permission.")
            if let limit = value.preparationCacheLimitBytes {
                try require(ExecutionPreferences.preparationCacheByteBounds.contains(limit) &&
                    value.plan.preparationCachePolicy != nil && value.plan.preparationCachePolicy != .off,
                    "Invalid captured preparation cache allowance in format5.")
            }
        case 4:
            guard let limit = value.preparationCacheLimitBytes else { throw WorkflowArchiveError("Capture format 4 requires a preparation cache allowance.") }
            try require(ExecutionPreferences.preparationCacheByteBounds.contains(limit) &&
                value.plan.preparationCachePolicy != nil && value.plan.preparationCachePolicy != .off &&
                (value.sampleCapture == nil || value.executionBudget != nil),
                "Invalid captured preparation cache allowance or capture provenance.")
        case 3:
            try require(value.executionBudget != nil && value.sampleCapture != nil,
                "Capture format 3 requires execution-budget and training sample-capture provenance.")
        case 2:
            try require(value.executionBudget != nil && value.sampleCapture == nil,
                "Capture format 2 requires execution-budget provenance without a sample-capture policy.")
        case nil:
            try require(value.executionBudget == nil && value.sampleCapture == nil,
                "Historical captures cannot contain versioned execution or sample-capture provenance.")
        default: throw WorkflowArchiveError("Unsupported capture format.")
        }
        if value.captureFormatVersion != 4 && value.captureFormatVersion != 5 && value.captureFormatVersion != 6 && value.captureFormatVersion != 7 && value.captureFormatVersion != 8 {
            try require(value.preparationCacheLimitBytes == nil, "Historical capture formats cannot contain preparation cache allowances.")
        }
        if let budget = value.executionBudget {
            _ = try JSONDecoder().decode(ResolvedExecutionBudget.self, from: encode(budget))
        }
    }
    @inline(never)
    private static func validateCaptureIdentityAndModel(_ value: WorkflowArchiveCapture) throws {
        try identity(value.projectID, value.directoryID, value.runID, value.requestIdentity)
        try require([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22].contains(value.schemaVersion) && trainingCompilerMatches(value.compilerContractVersion,schema:value.schemaVersion), "Unsupported capture/compiler version pair.")
        try require((value.schemaVersion == 11) == (value.captureFormatVersion == 6), "Workflow11 requires capture format6; other workflows cannot acquire it.")
        try require([13,14,15,16,17,18,19,20,21,22].contains(value.schemaVersion) == (value.captureFormatVersion == 8),"Workflow13/14 require capture format8; other workflows cannot acquire it.")
        try require((value.schemaVersion == 12) == (value.captureFormatVersion == 7), "Workflow12 requires capture format7; other workflows cannot acquire it.")
        try require(value.plan.bagInput == nil || ((value.schemaVersion == 12 && value.compilerContractVersion == 11) || ([13,14,15,16,17,18,19,20,21,22].contains(value.schemaVersion) && trainingCompilerMatches(value.compilerContractVersion,schema:value.schemaVersion))),"Packet mapping requires workflow12/compiler11 or workflow13/14/compiler12.")
        try value.plan.validateBagMapping()
        try require(Set(value.plan.modelOrigins.map(\.nodeID)).count == value.plan.modelOrigins.count, "Captured model origins must be unique.")
        if let source = value.plan.modelProvenance {
            if value.schemaVersion >= 14 {
                // Ownership-only migration may retain source2/native1. The exact
                // per-model selection is authenticated by History's recompile of
                // the pinned canonical workflow, including modelSourceVersions.
                try require((source.sourceCompositeDocument.schemaVersion == 2 && source.nativeExpansionContractVersion == 1) ||
                            (source.sourceCompositeDocument.schemaVersion == 3 && source.nativeExpansionContractVersion == 2),
                            "Captured Block source uses a different expansion version.")
            } else {
                let expected = value.schemaVersion >= 11 ? 2 : 1
                try require(source.nativeExpansionContractVersion == expected && (expected == 2 ? source.sourceCompositeDocument.schemaVersion == 3 : [1,2].contains(source.sourceCompositeDocument.schemaVersion)), "Captured Block source uses a different expansion version.")
            }
            try require(source.expandedGraph == value.plan.modelGraph && source.nodeOrigins == Dictionary(uniqueKeysWithValues:value.plan.modelOrigins.map {($0.nodeID,$0.origin)}), "Captured Block graph or origins differ from the execution plan.")
        }
        if value.plan.modelGraph.compositionPlan != nil {
            try require([11,12,13,14,15,16,17,18,19,20,21,22].contains(value.schemaVersion) && value.plan.modelProvenance != nil, "Composition requires workflow11–14 and captured source provenance.")
            try require(value.plan.modelProvenance?.nativeExpansionContractVersion == 2, "Native composition requires source3 expansion provenance.")
        }
    }
    @inline(never)
    private static func validateCaptureGradientAndInitialization(_ value: WorkflowArchiveCapture) throws {
        if value.plan.gradientPolicyRecipe != nil || value.plan.trainabilityRecipe != nil {
            try require((value.compilerContractVersion == 9 && value.schemaVersion == 10) || (value.compilerContractVersion == 10 && value.schemaVersion == 11) || (value.compilerContractVersion == 11 && value.schemaVersion == 12) || (trainingCompilerMatches(value.compilerContractVersion,schema:value.schemaVersion) && [13,14,15,16,17,18,19,20,21,22].contains(value.schemaVersion)),"Gradient/trainability captures require schema10/compiler9, schema11/compiler10, schema12/compiler11 or schema13/14/compiler12.")
            let otherSources = [Optional(value.plan.train), value.plan.optimizerRecipe?.source, value.plan.schedulerRecipe?.source, value.plan.lossRecipe?.source, value.plan.initializationRecipe?.source].compactMap {$0}
            let reduction = try value.plan.lossRecipe.map {try LossConfiguration.decodeRecipe($0.configuration).values["reduction"]!.stringValue!} ?? "mean"
            try GradientArchiveRecipes.validate(policy:value.plan.gradientPolicyRecipe,trainability:value.plan.trainabilityRecipe,graph:value.plan.modelGraph,origins:value.plan.modelOrigins,otherSources:otherSources,objectiveReduction:reduction,scalarization:value.plan.objectiveScalarization?.rawValue ?? "mean", inlineTrain:value.plan.train, inlineRoles:Set(value.plan.inlineTrainingRecipeRoles ?? []))
        }
        if let recipe = value.plan.initializationRecipe {
            try require(value.compilerContractVersion >= 8 && value.schemaVersion >= 9, "Starting settings require schema9/compiler8 or newer reviewed contracts.")
            try recipe.validate()
            let configuration = try InitializationConfiguration.decodeRecipe(recipe.configuration)
            try require(configuration.rules.allSatisfy { rule in value.plan.modelGraph.nodes.contains { $0.id == rule.owner.node } },
                        "Captured starting settings select an owner outside the model.")
        }
    }
    @inline(never)
    private static func validateCaptureOptimizationAndObjective(_ value: WorkflowArchiveCapture) throws {
        if let recipe = value.plan.optimizerRecipe {
            try require(value.compilerContractVersion >= 5, "Historical captures cannot contain optimizer recipes.")
            try recipe.validate()
            let configuration = try OptimizerConfiguration.decodeRecipe(recipe.configuration)
            try require(value.plan.settings.learningRate == configuration.learningRate,
                "Captured learning rate differs from its selected optimizer recipe.")
        }
        if let recipe = value.plan.schedulerRecipe {
            try require(value.compilerContractVersion >= 6, "Historical captures cannot contain scheduler recipes.")
            try recipe.validate()
            guard let optimizer = value.plan.optimizerRecipe else {
                throw WorkflowArchiveError("Captured scheduler requires its explicit optimizer recipe.")
            }
            _ = optimizer // Source distinction is checked with explicit inline-role proof.
        }
        if let recipe=value.plan.lossRecipe {
            try require(value.compilerContractVersion >= 7 && value.plan.targetColumn == nil && value.plan.objectiveScalarization != nil,
                "Selected objective requires compiler 7, explicit score scalarization and no dummy target column.")
            try recipe.validate()
            try require(value.plan.featureColumns == recipe.targetMapping.inputColumns,
                "Captured objective input mapping or qualified source differs from its selected control recipe.")
        } else {
            try require(value.plan.objectiveScalarization == nil,"Implicit objective captures cannot contain explicit scalarization.")
            guard let target=value.plan.targetColumn else {throw WorkflowArchiveError("Historical objective requires its selected target column.")}
            try require(!target.isEmpty && target.utf8.count<=4096,"Invalid historical target column.")
        }
    }
    @inline(never)
    private static func validateCaptureLifecycle(_ value: WorkflowArchiveCapture) throws {
        if value.compilerContractVersion >= 3 {
            guard let mode=value.plan.trainingMode else {throw WorkflowArchiveError("Compiler3 capture requires explicit training mode.")}
            try require((mode != .fresh) == (value.checkpointParent != nil), "Continuation requires one checkpoint parent; fresh training forbids it.")
            try require((mode != .fresh) == (value.checkpointFiles != nil), "Continuation requires a pinned parent file inventory; fresh training forbids it.")
            if let parent=value.checkpointParent {
                guard let files=value.checkpointFiles else {throw WorkflowArchiveError("Missing checkpoint inventory.")}
                let names=Set(files.map {String($0.path.dropFirst("continuation-parent/".count))})
                let expected=try WorkflowCheckpointInventory.structuralLimits(names:names)
                try require(files.count == expected.count && Set(files.map(\.path)) == Set(expected.keys.map {"continuation-parent/"+$0}),
                    "Checkpoint inventory must contain the base files and only a dependent schedule trace.")
                for entry in files {
                    let name=String(entry.path.dropFirst("continuation-parent/".count))
                    try file(entry,path:"continuation-parent/"+name,limit:expected[name]!)
                    try require(entry.bytes>0,"Checkpoint parent artifact cannot be empty.")
                    if name == "manifest.json" {try require(entry.sha256 == parent.modelManifestSHA256,"Checkpoint parent manifest inventory differs from its selected pin.")}
                }
                try identity(parent.projectID,parent.directoryID,parent.runID,parent.requestIdentity)
                try require(parent.projectID == value.projectID && parent.hasValidPins
                    && UUID(uuidString:parent.directoryID) != UUID(uuidString:value.directoryID)
                    && UUID(uuidString:parent.runID) != UUID(uuidString:value.runID),
                    "Checkpoint parent must be a different run in this project with valid manifest and request pins.")
            }
        } else {
            try require(value.plan.trainingMode == nil && value.checkpointParent == nil && value.checkpointFiles == nil,
                "Historical capture cannot contain lifecycle mode or checkpoint parent fields.")
        }
    }
    @inline(never)
    private static func validateCaptureCache(_ value: WorkflowArchiveCapture) throws {
        if value.compilerContractVersion == 1 {
            try require(value.plan.preparationCachePolicy == nil && value.priorCache == nil, "Legacy capture cannot contain cache policy or reference fields.")
        } else {
            guard let policy = value.plan.preparationCachePolicy else { throw WorkflowArchiveError("Compiler 2 capture requires explicit preparation cache policy.") }
            try require((policy == .reuse) == (value.priorCache != nil), "Cache reuse requires exactly one managed prior reference; other policies forbid it.")
            if let reference = value.priorCache {
                try require(UUID(uuidString: reference.projectID) == UUID(uuidString: value.projectID) && UUID(uuidString: reference.directoryID) != nil && UUID(uuidString: reference.directoryID) != UUID(uuidString: value.directoryID) && isHash(reference.preparationManifestSHA256), "Invalid managed prior cache identity or manifest pin.")
            }
        }
    }
    @inline(never)
    private static func validateCaptureFilesAndSources(_ value: WorkflowArchiveCapture) throws {
        try file(value.workflow, path: "workflow.json", limit: 2_000_000)
        try file(value.modelRegistry, path: "model-registry.json", limit: 2_000_000)
        try file(value.dataRegistry, path: "data-registry.json", limit: 2_000_000)
        try file(value.runtimeEvidence, path: "runtime-evidence.json", limit: 256_000)
        try require(isHash(value.modelRegistrySHA256) && isHash(value.dataRegistrySHA256), "Invalid registry hash.")
        let plan = value.plan
        try require(plan.projectID == value.projectID && plan.revision == value.revision && plan.train == value.train && value.train.domain == .control, "Capture plan identity differs.")
        try require(!value.train.localID.isEmpty && value.train.localID.utf8.count <= 128, "Invalid Train address.")
        try require((1...128).contains(value.sources.count), "Invalid source count.")
        var addresses = Set<WorkflowNodeAddress>(), paths = Set<String>(), total = 0
        let aggregateLimit = WorkflowCapturedSourceArchive.aggregateLimit(hasCollections: value.sources.contains { $0.collectionMembers != nil })
        for source in value.sources {
            try require(source.address.domain == .data && !source.address.localID.isEmpty && source.address.localID.utf8.count <= 128 && addresses.insert(source.address).inserted, "Invalid or duplicate source address.")
            let sourcePaths = try WorkflowCapturedSourceArchive.validate(source, prefix: "inputs/")
            try require((source.collectionMembers != nil) == (plan.dataflow.nodes.first { $0.id == source.address.localID }?.operatorID == "kwip.data.sourceFiles"), "Source closure kind differs from the captured source operator.")
            try require(sourcePaths.isDisjoint(with: paths), "Duplicate captured source paths.")
            paths.formUnion(sourcePaths)
            let count = source.bytes + (source.collectionMembers ?? []).reduce(0) { $0 + $1.bytes }
            try require(count <= aggregateLimit - total, "Captured source closure exceeds its byte budget.")
            total += count
        }
        try require(total <= aggregateLimit && addresses == Set(plan.sourceAddresses) && addresses.count == plan.sourceAddresses.count, "Captured sources differ from plan or exceed budget.")
    }
    @inline(never)
    private static func validateCapturePlanBounds(_ value: WorkflowArchiveCapture) throws {
        let plan = value.plan
        try require(isHash(plan.modelSemanticSHA256) && plan.modelOrigins.count <= 128 && Set(plan.modelOrigins.map(\.nodeID)).count == plan.modelOrigins.count, "Invalid model origin or semantic identity.")
        try require(Set(plan.modelOrigins.map(\.nodeID)) == Set(plan.modelGraph.nodes.map(\.id)), "Model origins differ from graph.")
        try portable(plan.settings)
        if let settings = plan.modelGraph.execution { try portable(settings) }
        try require((1...128).contains(plan.featureColumns.count) && Set(plan.featureColumns).count == plan.featureColumns.count && plan.featureColumns.allSatisfy { !$0.isEmpty && $0.utf8.count <= 4096 }, "Invalid feature mapping.")
        try require(plan.dataflow.nodes.count <= 128 && plan.dataflow.edges.count <= 256 && plan.modelGraph.nodes.count <= 128 && plan.modelGraph.edges.count <= 256, "Plan exceeds graph bounds.")
    }
    private static func portable(_ settings: ExecutionSettings) throws {
        try require(settings.datasetCSV == nil && settings.preparedTraining == nil && settings.compositeSource == nil, "Portable capture cannot contain private execution paths.")
        try require((1...10_000).contains(settings.epochs) && settings.learningRate.isFinite && settings.learningRate > 0 && settings.batchSize > 0, "Invalid captured execution settings.")
        try settings.gradientCapture?.validate()
    }
    static func validate(_ value: WorkflowArchiveOutcome) throws {
        try identity(value.projectID, value.directoryID, value.runID, value.requestIdentity)
        if value.failureCode != nil { try require(value.status == .failed, "Failure code requires failed outcome.") }
        try require([2,3].contains(value.schemaVersion), "Unsupported outcome version.")
        let hasDiagnostic = value.failedGradientEvidence != nil || value.failedGradientNotice != nil
        try require(value.schemaVersion == 3 ? hasDiagnostic : !hasDiagnostic, "Failed-gradient fields require outcome version 3.")
        if hasDiagnostic {
            try require(value.status == .failed && value.terminalPhase == .training && value.capture != nil && value.preparationManifestSHA256 != nil, "Failed-gradient evidence requires captured failed training.")
        }
        if let notice=value.failedGradientNotice {try require(!notice.isEmpty && notice.utf8.count<=2000 && value.failedGradientEvidence==nil,"Invalid failed-gradient unavailable notice.")}
        if let diagnostic=value.failedGradientEvidence {
            try require(diagnostic.schemaVersion==1 && UUID(uuidString:diagnostic.workerRunID) != nil,"Invalid failed-gradient descriptor identity.")
            try file(diagnostic.file,path:WorkflowFailedGradientEvidence.path,limit:FailedGradientCaptureCodec.maximumArtifactBytes)
        }
        try require([WorkflowExecutionPhase.completed, .failed, .cancelled].contains(value.status) && [.idle, .preparing, .training].contains(value.terminalPhase), "Invalid terminal status or phase.")
        let clock = formatter()
        guard let start = clock.date(from: value.startedAt), let finish = clock.date(from: value.finishedAt), clock.string(from: start) == value.startedAt, clock.string(from: finish) == value.finishedAt, finish >= start else { throw WorkflowArchiveError("Invalid outcome timestamps.") }
        if let error = value.error { try require(!error.isEmpty && error.utf8.count <= 16_384, "Invalid terminal error text.") }
        try require(value.status != .failed || value.error != nil, "Failure requires original error text.")
        if let capture = value.capture { try file(capture, path: "capture.json", limit: 8_000_000) }
        try require(value.artifacts.count <= 320, "Artifact inventory exceeds its count bound.")
        var files: [String: WorkflowArchiveFile] = [:], preparationBytes = 0, preparationCount = 0
        for artifact in value.artifacts {
            let parts = try pathParts(artifact.path)
            try require(files[artifact.path] == nil && isHash(artifact.sha256) && artifact.bytes >= 0, "Invalid or duplicate artifact.")
            let limit: Int
            if parts.count == 3 && parts[0] == "preparation" && parts[1] == "artifacts" {
                limit = parts[2] == "manifest.json" ? 2_000_000 : 50_000_000
                preparationCount += 1
                try require(artifact.bytes <= limit, "Preparation artifact too large.")
                preparationBytes += artifact.bytes
            } else if artifact.path == WorkflowFailedGradientEvidence.path {
                limit=FailedGradientCaptureCodec.maximumArtifactBytes
                try require(value.schemaVersion==3 && value.status == .failed && value.failedGradientEvidence?.file==artifact,"Unpinned failed-gradient artifact.")
            } else {
                let limits = ["training/artifacts/routing.json":2_000_000,"training/artifacts/routingSample.json":2_000_000,TrainingProgressArchive.path:TrainingProgressArchive.maximumBytes,"training/artifacts/prediction-table.json":50_000_000,"training/artifacts/performance.json":131072,"training/artifacts/checkpoint.pt":100_000_000,"training/artifacts/manifest.json":20_000_000,"training/artifacts/metrics.json":8_000_000,"training/artifacts/model.py":2_000_000,"training/artifacts/data_loading_recipe.py":131072,"training/artifacts/objective_recipe.py":131072,"training/artifacts/initialization_recipe.py":131072,"training/artifacts/gradient_policy.py":131072,"training/artifacts/trainability_recipe.py":131072,"training/artifacts/derivative-results.json":8_388_608,"training/artifacts/training_recipe.py":8192,"training/artifacts/rate-trace.json":4_000_000,"training/artifacts/graph.json":2_000_000,"training/artifacts/composite/source.json":2_000_000,"training/artifacts/gradient-summary.json":8_000_000,"training/artifacts/predictions.json":1_000_000,"training/artifacts/activations.json":2_000_000]
                guard let maximum = limits[artifact.path] else { throw WorkflowArchiveError("Unrecognized phase artifact path.") }; limit = maximum
                try require(value.status == .completed, "Interrupted run cannot publish completed model artifacts.")
            }
            try require(artifact.bytes <= limit, "Artifact exceeds byte limit."); files[artifact.path] = artifact
        }
        try require(preparationCount <= 301 && preparationBytes <= 202_000_000, "Preparation inventory exceeds aggregate budget.")
        for (pin, path) in [(value.preparationManifestSHA256,"preparation/artifacts/manifest.json"),(value.modelManifestSHA256,"training/artifacts/manifest.json")] {
            if let pin { try require(isHash(pin) && files[path]?.sha256 == pin, "Manifest pin differs from inventory.") }
            else { try require(files[path] == nil, "Unpinned manifest.") }
        }
        if let diagnostic=value.failedGradientEvidence {try require(files[diagnostic.file.path]==diagnostic.file,"Failed-gradient descriptor differs from inventory.")}
        if preparationCount > 0 { try require(value.preparationManifestSHA256 != nil && value.capture != nil, "Preparation artifacts require pinned capture and manifest.") }
        if value.status == .completed {
            try require(value.capture != nil && value.preparationManifestSHA256 != nil && value.modelManifestSHA256 != nil && value.workerRunID.flatMap(UUID.init(uuidString:)) != nil && value.terminalPhase == .training && value.error == nil, "Completed outcome lacks required evidence.")
        } else { try require(value.modelManifestSHA256 == nil && value.workerRunID == nil, "Interrupted outcome cannot claim model completion.") }
    }
}
