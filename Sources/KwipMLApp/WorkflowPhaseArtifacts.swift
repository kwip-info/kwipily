import Foundation
import Darwin
import CryptoKit
import KwipMLCore

struct WorkflowPhaseError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

struct WorkflowPhaseExpectation {
    let plan: WorkflowExecutionPlan
    let modelRegistrySHA256: String
    let dataRegistrySHA256: String
    let priorCacheManifestSHA256: String?
    let requiresCacheReport: Bool
    let requiresLifecycleEvidence: Bool
    let lifecycleParent: WorkflowLifecycleParentEvidence?
    let sampleCapture: TrainingSampleCapturePolicy?
    let preparationCacheLimitBytes: Int?
    let expectedCPUThreads: Int?
    let capturedModelRegistry: ComponentRegistry?
    let derivativeCapture: DerivativeCapturePolicy?
    let workload: ExecutionWorkloadPolicy?
    init(plan: WorkflowExecutionPlan, modelRegistrySHA256: String, dataRegistrySHA256: String, priorCacheManifestSHA256: String? = nil, compilerContractVersion: Int = 1, requiresCacheReport: Bool = false, lifecycleParent: WorkflowLifecycleParentEvidence? = nil, expectedCPUThreads: Int? = nil, sampleCapture: TrainingSampleCapturePolicy? = nil, preparationCacheLimitBytes: Int? = nil, capturedModelRegistry: ComponentRegistry? = nil, derivativeCapture: DerivativeCapturePolicy? = nil, workload: ExecutionWorkloadPolicy? = nil) {
        self.requiresLifecycleEvidence = compilerContractVersion >= 3
        self.lifecycleParent = lifecycleParent
        self.expectedCPUThreads = expectedCPUThreads
        self.capturedModelRegistry = capturedModelRegistry
        self.derivativeCapture = derivativeCapture
        self.workload = workload
        self.sampleCapture = sampleCapture
        self.preparationCacheLimitBytes = preparationCacheLimitBytes
        self.requiresCacheReport = requiresCacheReport || compilerContractVersion >= 2
        self.priorCacheManifestSHA256 = priorCacheManifestSHA256
        self.plan = plan; self.modelRegistrySHA256 = modelRegistrySHA256; self.dataRegistrySHA256 = dataRegistrySHA256
    }
    init(request: WorkflowExecutionRequest, lifecycleParent: WorkflowLifecycleParentEvidence? = nil) {
        self.init(plan: request.plan, modelRegistrySHA256: request.modelRegistrySHA256, dataRegistrySHA256: request.dataRegistrySHA256, priorCacheManifestSHA256: request.priorCache?.reference.preparationManifestSHA256, compilerContractVersion: request.archiveCompilerContractVersion, requiresCacheReport: true, lifecycleParent: lifecycleParent, expectedCPUThreads: request.executionBudget?.resources?.cpuThreads, sampleCapture: request.sampleCapture, preparationCacheLimitBytes: request.preparationCacheLimitBytes, capturedModelRegistry: request.capturedModelRegistry, derivativeCapture: request.derivativeCapture, workload: request.executionBudget?.resources?.workloadPolicy)
    }
}

/// Bounded native inventory verification; semantic table/partition validation remains
/// the existing prepared-data worker's responsibility before tensor allocation.
enum WorkflowPhaseArtifacts {
    static func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }
    static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let bytes = try encode(value)
        guard bytes.count <= 8_000_000 else { throw WorkflowPhaseError("Workflow request or receipt exceeds its size limit.") }
        try bytes.write(to: url, options: .atomic)
    }
    static func preparedArguments(_ selection: PreparedTrainingSelection) throws -> [String] {
        try selection.validate()
        return ["--prepared-dataflow", selection.manifestPath, "--prepared-manifest-sha256", selection.manifestSHA256,
            "--prepared-graph-digest", selection.graphDigest,
            "--prepared-training-node", selection.trainingNode, "--prepared-training-port", selection.trainingPort,
            "--prepared-validation-node", selection.validationNode, "--prepared-validation-port", selection.validationPort,
            "--feature-columns-json", String(decoding: try encode(selection.featureColumns), as: UTF8.self), "--target-column", selection.targetColumn]+(try selection.namedInputPlan.map {["--named-input-plan-json",String(decoding:try encode($0),as:UTF8.self)]} ?? [])
    }
    static func preparedArguments(_ selection: WorkflowPreparedSelection) throws -> [String] {
        switch selection {
        case .legacy(let value): return try preparedArguments(value)
        case .objective(let value):
            try value.validate()
            let reference = value.reference
            let recipe = try encode(value.objectiveRecipe), mapping = try encode(value.targetMapping)
            guard recipe.count <= LossConfiguration.maximumRecipeBytes, mapping.count <= 524288 else {
                throw WorkflowPhaseError("Prepared objective recipe or target mapping exceeds its launch limit.")
            }
            return ["--prepared-dataflow", reference.manifestPath, "--prepared-manifest-sha256", reference.manifestSHA256,
                "--prepared-graph-digest", reference.graphDigest,
                "--prepared-training-node", reference.trainingNode, "--prepared-training-port", reference.trainingPort,
                "--prepared-validation-node", reference.validationNode, "--prepared-validation-port", reference.validationPort,
                "--objective-recipe-json", String(decoding: recipe, as: UTF8.self),
                "--objective-target-mapping-json", String(decoding: mapping, as: UTF8.self),
                "--objective-scalarization", value.scalarization.rawValue]+(try value.namedInputPlan.map {["--named-input-plan-json",String(decoding:try encode($0),as:UTF8.self)]} ?? [])
        }
    }
    struct Prepared {
        let cacheReport: DataFlowCacheReport?
        let selection: WorkflowPreparedSelection
        let hashes: [String: String]
        let artifactFiles: [WorkflowArchiveFile]
    }
    private struct Inventory: Decodable {
        let cache: JSONValue?
        let schemaVersion: Int
        let kind: String
        let status: String
        let graphDigest: String
        let artifacts: [Entry]
    }
    private struct Entry: Decodable {
        let path: String
        let bytes: Int
        let sha256: String
        let kind: String
        let nodeID: String?
        let port: String?
    }
    static func preparation(request: WorkflowExecutionRequest, root: URL, completion: [String: Any],modelInputNodes:[String]?=nil) throws -> Prepared {
        let output = root.appendingPathComponent("preparation/artifacts")
        guard matchesOutputPath(completion["outputDir"], expected: output),
              matchesOutputPath(completion["manifestPath"], expected: output.appendingPathComponent("manifest.json")) else {
            throw WorkflowPhaseError("Preparation completion does not identify this request's artifact directory.")
        }
        let verified = try preparation(expectation: .init(request: request), root: root)
        if let modelInputNodes {
            try WorkflowCompletedPreparationInventory.verifyModelInputs(data:DiagnosticFileIO.read(root:root,parts:["preparation","artifacts","manifest.json"],limit:2_000_000),expected:modelInputNodes)
        }
        guard completion["graphDigest"] as? String == verified.selection.graphDigest else { throw WorkflowPhaseError("Preparation event differs from its verified manifest.") }
        return verified
    }
    static func preparation(expectation: WorkflowPhaseExpectation, root: URL, storageRoot: URL? = nil, storageParts: [String] = []) throws -> Prepared {
        let parts = storageParts + ["preparation", "artifacts"]
        let readRoot = storageRoot ?? root
        guard let resolvedRoot = realpath(readRoot.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let trustedRoot = URL(fileURLWithPath: String(cString: resolvedRoot), isDirectory: true)
        free(resolvedRoot)
        let output = root.appendingPathComponent("preparation/artifacts")
        let bytes = try DiagnosticFileIO.read(root: readRoot, parts: parts + ["manifest.json"], limit: 2_000_000)
        let manifest = try JSONDecoder().decode(Inventory.self, from: bytes)
        let artifactPolicy = try PreparationArtifactPolicy(manifest: bytes)
        let retention=try JSONDecoder().decode([String:JSONValue].self,from:bytes)["observationRetention"]
        guard retention == (expectation.sampleCapture?.savesComponentOutputs == false ? .string("requiredOnly"):nil) else {throw WorkflowPhaseError("Preparation retention differs from the captured run.")}
        guard manifest.schemaVersion == 1, manifest.kind == "dataflowExecution", manifest.status == "completed",
              isHash(manifest.graphDigest),
              (1...artifactPolicy.inventoryCount).contains(manifest.artifacts.count) else { throw WorkflowPhaseError("Preparation manifest is incomplete or unsupported.") }
        if expectation.sampleCapture?.savesComponentOutputs == false {
            let raw=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            guard raw["requiredOutputNodes"] == .array(expectation.plan.requiredPreparationNodes.map(JSONValue.string)) else {throw WorkflowPhaseError("Required preparation retention differs from the selected training partitions.")}
        }
        var names = Set<String>(), total = 0, hashes: [String: String] = [:]
        var files: [WorkflowArchiveFile] = []
        var graph: DataFlowDocument?, registryHash: String?
        var endpoints = Set<DataFlowEndpoint>()
        for entry in manifest.artifacts {
            guard validName(entry.path), entry.path != "manifest.json", names.insert(entry.path).inserted,
                  (0...artifactPolicy.fileBytes(entry.path)).contains(entry.bytes), isHash(entry.sha256) else { throw WorkflowPhaseError("Preparation inventory contains an invalid artifact.") }
            guard entry.bytes <= artifactPolicy.totalBytes - total else { throw WorkflowPhaseError("Preparation inventory exceeds its total byte limit.") }
            total += entry.bytes
            // Resolve only the explicitly trusted root. Descendants remain subject
            // to descriptor-based no-follow traversal and streaming hash checks.
            do {
                try WorkflowPreviewToolContract.verifyFile(
                    file: .init(path: (parts + [entry.path]).joined(separator: "/"), bytes: entry.bytes, sha256: entry.sha256),
                    directory: trustedRoot, limit: artifactPolicy.fileBytes(entry.path))
            } catch { throw WorkflowPhaseError("Preparation artifact verification failed for \(entry.path): \(error.localizedDescription)") }
            hashes["preparation/artifacts/" + entry.path] = entry.sha256
            files.append(.init(path: "preparation/artifacts/" + entry.path, bytes: entry.bytes, sha256: entry.sha256))
            if entry.path == "dataflow.json" || entry.path == "dataflow-operators.json" {
                let content = try DiagnosticFileIO.read(root: readRoot, parts: parts + [entry.path], limit: min(entry.bytes, 2_000_000))
                guard content.count == entry.bytes, hash(content) == entry.sha256 else { throw WorkflowPhaseError("Preparation metadata hash or size changed.") }
                if entry.path == "dataflow.json" { graph = try DataFlowCodec.decodeGraph(content) }
                else { registryHash = try WorkflowArchiveCodec.registryHash(JSONDecoder().decode(DataFlowOperatorRegistry.self, from: content)) }
            }
            if entry.kind == "table", let node = entry.nodeID, let port = entry.port {
                guard endpoints.insert(.init(node: node, port: port)).inserted else { throw WorkflowPhaseError("Preparation contains duplicate table endpoints.") }
            }
        }
        try FileImageArtifactInventory.verify(manifest:bytes,files:files,prefix:"preparation/artifacts/",directory:readRoot,readPrefix:parts)
        guard graph == expectation.plan.dataflow, registryHash == expectation.dataRegistrySHA256,
              endpoints.contains(expectation.plan.training), endpoints.contains(expectation.plan.validation) else {
            throw WorkflowPhaseError("Preparation graph, registry or selected table endpoints differ from the captured plan.")
        }
        // Old captures may predate cache reporting. When present, verify the
        // complete report from these same pinned bytes, never a later file read.
        guard manifest.cache != nil || (!expectation.requiresCacheReport && expectation.plan.preparationCachePolicy == .off) else {
            throw WorkflowPhaseError("Preparation omitted the requested cache evidence.")
        }
        let cacheReport = try manifest.cache.map { _ in
            try DataFlowCacheReportCodec.decodeManifest(bytes, graph: expectation.plan.dataflow,
                expectation: .init(policy: expectation.plan.preparationCachePolicy, sourceManifestSHA256: expectation.priorCacheManifestSHA256, materializedBytesLimit: expectation.preparationCacheLimitBytes, workload: expectation.workload))
        }
        let pin = hash(bytes); hashes["preparation/artifacts/manifest.json"] = pin
        let reference = PreparedDataReference(manifestPath: output.appendingPathComponent("manifest.json").path,
            manifestSHA256: pin, graphDigest: manifest.graphDigest,
            trainingNode: expectation.plan.training.node, trainingPort: expectation.plan.training.port,
            validationNode: expectation.plan.validation.node, validationPort: expectation.plan.validation.port)
        let selection: WorkflowPreparedSelection
        if let recipe = expectation.plan.lossRecipe {
            guard expectation.plan.targetColumn == nil, expectation.plan.featureColumns == recipe.targetMapping.inputColumns else {
                throw WorkflowPhaseError("Prepared objective mapping differs from the captured typed plan.")
            }
            selection = .objective(.init(reference: reference, objectiveRecipe: recipe.configuration.recipeValue,
                targetMapping: recipe.targetMapping, scalarization: expectation.plan.objectiveScalarization,namedInputPlan:expectation.plan.namedInputPlan))
        } else {
            guard let legacyTarget = expectation.plan.targetColumn else { throw WorkflowPhaseError("Legacy training requires its captured scalar target.") }
            selection = .legacy(.init(manifestPath: reference.manifestPath, manifestSHA256: pin, graphDigest: reference.graphDigest,
                trainingNode: reference.trainingNode, trainingPort: reference.trainingPort,
                validationNode: reference.validationNode, validationPort: reference.validationPort,
                featureColumns: expectation.plan.featureColumns, targetColumn: legacyTarget,namedInputPlan:expectation.plan.namedInputPlan))
        }
        try selection.validate()
        files.append(.init(path: "preparation/artifacts/manifest.json", bytes: bytes.count, sha256: pin))
        return Prepared(cacheReport: cacheReport, selection: selection, hashes: hashes, artifactFiles: files.sorted { $0.path < $1.path })
    }
    struct PreparedGradientEvidence {
        let rows:WorkflowFailedGradientEvidence.PreparedRowCounts
        let manifest:Data
        let graphDigest:String
    }
    private static func gradientPreparation(_ expectation:WorkflowPhaseExpectation,selection:WorkflowPreparedSelection,root:URL,parts:[String])throws->PreparedGradientEvidence? {
        guard expectation.plan.gradientPolicyRecipe != nil || expectation.plan.trainabilityRecipe != nil || expectation.plan.datasetRecipe != nil || expectation.plan.loaderRecipe != nil else {return nil}
        let rows=try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,parts:parts,selection:selection)
        let bytes=try DiagnosticFileIO.read(root:root,parts:parts+["preparation","artifacts","manifest.json"],limit:2_000_000)
        guard hash(bytes)==selection.manifestSHA256 else {throw WorkflowPhaseError("Selected derivative preparation changed.")}
        return .init(rows:rows,manifest:bytes,graphDigest:selection.graphDigest)
    }
    struct Trained {
        var routingTables:[String:WorkflowPredictionTableEvidence]=[:]
        var predictionTable:WorkflowPredictionTableEvidence? = nil
        let partitionEvidence: WorkflowPartitionEvidence?
        let manifestSHA256: String
        let hashes: [String: String]
        let artifactFiles: [WorkflowArchiveFile]
        let workerRunID: String
        let modelCode: String
        let predictionPreview: PredictionPreview?
        let activationPreviews: JSONValue?
        let gradientArtifact: GradientCaptureArtifact?
        let metrics: [EpochMetric]
        var performanceCapture: WorkflowPerformanceCapture? = nil
        var derivativeResults: DerivativeResults? = nil
        var objectiveContext: WorkflowObjectiveArtifactContext? = nil
        var objectivePredictionPreview: ObjectivePredictionPreview? = nil
        var objectiveEvaluation: ObjectiveEvaluation? = nil
        var objectiveModelCode: String? = nil
        var bagContext: WorkflowBagArtifactContext? = nil
        var bagPredictionPreview: BagPredictionPreview? = nil
        var objectiveEvaluationContext: WorkflowObjectiveEvaluationContext? = nil
        var dataLoadingIdentity:WorkflowDataLoadingIdentity? = nil
        var dataLoadingModelCode:String? = nil
        var trainingControl:TrainingControlReport? = nil
    }
    private struct TrainingManifest: Decodable {
        let schemaVersion: Int
        let status: String
        let runID: String
        let graph: GraphDocument
        let seed: Int
        let checkpoint: Checkpoint
        let settings: Settings
        let provenance: Provenance
        struct Checkpoint: Decodable {
            let sha256: String
            let bytes: Int
            let completedEpoch: Int
            let metricsSHA256: String
        }
        struct Settings: Decodable {
            let epochs: Int
            let partitionMode: String
            let requestedDevice: String
            let requestedBatchSize: Int
            let learningRate: Double
        }
        struct Provenance: Decodable {
            let featureColumns: [String]?
            let targetColumn: String?
            let preparation: Preparation?
        }
        struct Preparation: Decodable {
            let manifestSHA256: String
            let graphDigest: String
            let trainingNode: String
            let trainingPort: String
            let validationNode: String
            let validationPort: String
        }
    }
    static func training(request: WorkflowExecutionRequest, root: URL, preparation: WorkflowPreparedSelection,
                         completion: [String: Any], lifecycleParent: WorkflowLifecycleParentEvidence? = nil) throws -> Trained {
        let output = root.appendingPathComponent("training/artifacts")
        guard matchesOutputPath(completion["outputDir"], expected: output) else { throw WorkflowPhaseError("Training completion identifies a different output directory.") }
        let verified = try training(expectation: .init(request:request,lifecycleParent:lifecycleParent),root:root,preparation:preparation)
        guard completion["runID"] as? String == verified.workerRunID else {throw WorkflowPhaseError("Training event run identity differs from manifest.")}
        let event = try JSONDecoder().decode([String:JSONValue].self,from:JSONSerialization.data(withJSONObject:completion))
        let expected = try request.sampleCapture.map {try JSONDecoder().decode(JSONValue.self,from:encode($0))}
        guard event["sampleCapture"] == expected else {throw WorkflowPhaseError("Training completion capture policy differs from the accepted request.")}
        let performanceManifestBytes = try DiagnosticFileIO.read(root: root, parts: ["training", "artifacts", "manifest.json"], limit: 20_000_000)
        let performanceManifest = try JSONDecoder().decode([String:JSONValue].self, from: performanceManifestBytes)
        guard hash(performanceManifestBytes) == verified.manifestSHA256,
              event["performanceCapture"] == performanceManifest["performanceCapture"] else {
            throw WorkflowPhaseError("Live performance capture differs from its verified saved descriptor.")
        }
        if request.sampleCapture?.savesComponentOutputs == false {
            guard event["previews"]==nil,event["predictionRows"]==nil,event["predictions"]==nil,event["predictionInput"]==nil,
                  case .object(let evaluation)=performanceManifest["evaluation"],case .object(let runtime)=performanceManifest["runtime"],
                  event["evaluationCount"]==evaluation["count"],event["validationLoss"]==evaluation["validationLoss"],event["accuracy"]==evaluation["accuracy"],event["device"]==runtime["device"] else {throw WorkflowPhaseError("Capture-off completion differs from its required model evidence or contains optional outputs.")}
        } else if verified.bagContext != nil {
            try verifyBagCompletion(event:event,trained:verified,graph:request.plan.model.graph,
                registry:request.capturedModelRegistry,sampleCapture:request.sampleCapture,root:root)
        } else if verified.objectiveContext != nil {
            let manifestBytes=try DiagnosticFileIO.read(root:root,parts:["training","artifacts","manifest.json"],limit:20_000_000)
            let predictionsBytes=try DiagnosticFileIO.read(root:root,parts:["training","artifacts","predictions.json"],limit:1_000_000)
            guard hash(manifestBytes) == verified.manifestSHA256,hash(predictionsBytes) == verified.hashes["training/artifacts/predictions.json"] else {throw WorkflowPhaseError("Objective artifacts changed before completion publication.")}
            let manifest=try JSONDecoder().decode([String:JSONValue].self,from:manifestBytes)
            let predictions=try JSONDecoder().decode([String:JSONValue].self,from:predictionsBytes)
            guard case .object(let evaluation)=manifest["evaluation"],case .object(let runtime)=manifest["runtime"],event["predictions"] == .array([]),event["predictionRows"] == predictions["predictionRows"],event["predictionInput"] == predictions["predictionInput"],event["objective"] == evaluation["objective"],event["accuracy"] == evaluation["accuracy"],event["validationLoss"] == evaluation["validationLoss"],event["evaluationCount"] == evaluation["count"],event["device"] == runtime["device"],event["gradientCapture"] == manifest["gradientCapture"],event["performanceCapture"] == manifest["performanceCapture"] else {throw WorkflowPhaseError("Live objective completion differs from its verified saved evidence.")}
        } else if request.sampleCapture != nil {
            try WorkflowModelTrainingExecution.verifyPresentation(event:event,graph:request.plan.model.graph,sampleCapture:request.sampleCapture)
        }
        if let context=verified.objectiveContext {try WorkflowModelTrainingExecution.verifyPresentation(event:event,graph:request.plan.model.graph,sampleCapture:request.sampleCapture,objectiveContext:context,capturedRegistry:request.capturedModelRegistry)}
        if let saved=verified.activationPreviews,saved != event["previews"] {throw WorkflowPhaseError("Live activation samples differ from the verified saved artifact.")}
        return verified
    }
    static func training(expectation: WorkflowPhaseExpectation, root: URL, preparation: WorkflowPreparedSelection,
                         storageRoot: URL? = nil, storageParts: [String] = []) throws -> Trained {
        if expectation.plan.bagInput != nil {
            return try bagTrainingArtifacts(expectation: expectation, root: root, selection: preparation,
                storageRoot: storageRoot, storageParts: storageParts)
        }
        switch preparation {
        case .legacy(let selection):
            guard expectation.plan.lossRecipe == nil else {throw WorkflowPhaseError("Selected objective requires its typed prepared mapping.")}
            return try training(expectation:expectation,root:root,preparation:selection,storageRoot:storageRoot,storageParts:storageParts)
        case .objective(let selection):
            guard let recipe=expectation.plan.lossRecipe,selection.scalarization == expectation.plan.objectiveScalarization else {throw WorkflowPhaseError("Prepared objective differs from its captured workflow.")}
            return try trainingArtifacts(graph:expectation.plan.model.graph,settings:expectation.plan.settings,preparation:nil,compositeSource:expectation.plan.model.provenance,trainingMode:expectation.plan.trainingMode,requiresLifecycleEvidence:expectation.requiresLifecycleEvidence,lifecycleParent:expectation.lifecycleParent,readRoot:storageRoot ?? root,parts:storageParts+["training","artifacts"],artifactPrefix:"training/artifacts/",expectedCPUThreads:expectation.expectedCPUThreads,sampleCapture:expectation.sampleCapture,optimizerRecipe:expectation.plan.optimizerRecipe,schedulerRecipe:expectation.plan.schedulerRecipe,initializationRecipe:expectation.plan.initializationRecipe?.configuration,objectiveSelection:selection,objectiveRecipe:recipe,capturedModelRegistry:expectation.capturedModelRegistry,capturedModelRegistrySHA256:expectation.modelRegistrySHA256,gradientPolicy:expectation.plan.gradientPolicyRecipe?.configuration,trainability:expectation.plan.trainabilityRecipe?.configuration,derivativeCapture:expectation.derivativeCapture,preparedGradientEvidence:gradientPreparation(expectation,selection:preparation,root:storageRoot ?? root,parts:storageParts),datasetRecipe:expectation.plan.datasetRecipe,loaderRecipe:expectation.plan.loaderRecipe,trainOrigin:expectation.plan.train,inlineRoles:expectation.plan.inlineDataLoadingRoles)
        }
    }
    static func training(request: WorkflowExecutionRequest, root: URL, preparation: PreparedTrainingSelection,
                         completion: [String: Any], lifecycleParent: WorkflowLifecycleParentEvidence? = nil) throws -> Trained {
        try training(request:request,root:root,preparation:.legacy(preparation),completion:completion,lifecycleParent:lifecycleParent)
    }
    static func training(expectation: WorkflowPhaseExpectation, root: URL, preparation: PreparedTrainingSelection, storageRoot: URL? = nil, storageParts: [String] = []) throws -> Trained {
        if expectation.plan.bagInput != nil {
            return try bagTrainingArtifacts(expectation: expectation, root: root, selection: .legacy(preparation),
                storageRoot: storageRoot, storageParts: storageParts)
        }
        return try trainingArtifacts(graph: expectation.plan.model.graph, settings: expectation.plan.settings,
            preparation: preparation, compositeSource: expectation.plan.model.provenance,
            trainingMode: expectation.plan.trainingMode, requiresLifecycleEvidence: expectation.requiresLifecycleEvidence,
            lifecycleParent: expectation.lifecycleParent, readRoot: storageRoot ?? root,
            parts: storageParts + ["training", "artifacts"], artifactPrefix: "training/artifacts/", expectedCPUThreads: expectation.expectedCPUThreads, sampleCapture: expectation.sampleCapture, optimizerRecipe: expectation.plan.optimizerRecipe, schedulerRecipe: expectation.plan.schedulerRecipe, initializationRecipe: expectation.plan.initializationRecipe?.configuration,capturedModelRegistry:expectation.capturedModelRegistry,gradientPolicy:expectation.plan.gradientPolicyRecipe?.configuration,trainability:expectation.plan.trainabilityRecipe?.configuration,derivativeCapture:expectation.derivativeCapture,preparedGradientEvidence:gradientPreparation(expectation,selection:.legacy(preparation),root:storageRoot ?? root,parts:storageParts),datasetRecipe:expectation.plan.datasetRecipe,loaderRecipe:expectation.plan.loaderRecipe,trainOrigin:expectation.plan.train,inlineRoles:expectation.plan.inlineDataLoadingRoles)
    }
    static func validateUnselectedGradientMetadata(_ manifest:[String:JSONValue],derivativeCapture:DerivativeCapturePolicy?)throws {
        func fields(_ value:JSONValue?)->[String:JSONValue] {if case .object(let map)=value {return map};return [:]}
        let storedSettings=fields(manifest["settings"]),provenance=fields(manifest["provenance"]),contract=fields(fields(manifest["checkpoint"])["contract"]),runtime=fields(manifest["runtime"])
        guard derivativeCapture == nil,storedSettings["microbatchesPerEpoch"] == nil,storedSettings["optimizerUpdatesPerEpoch"] == nil,runtime["gradientSummaryContext"] == nil,storedSettings["gradientPolicyRecipe"] == nil,storedSettings["trainabilityRecipe"] == nil,provenance["gradientPolicy"] == nil,provenance["trainability"] == nil,contract["gradientPolicy"] == nil,contract["trainability"] == nil,runtime["updateCadence"] == nil,manifest["derivativeCapture"] == nil,manifest["derivativeResults"] == nil else {throw WorkflowPhaseError("Unselected gradient controls cannot acquire policy metadata or derivative permission.")}
    }
    /// Compare the decoded receipt with the exact JSON sent to the worker. JSONValue's
    /// integer-first decoder legitimately reads authored number endpoints 0.0/1.0 as
    /// integers; source types, fields and all other structure still compare exactly.
    static func matchesSerializedGraph(_ actual:GraphDocument,expected:GraphDocument)throws->Bool {
        actual == (try JSONDecoder().decode(GraphDocument.self,from:encode(expected)))
    }
    /// Shared bounded training verifier. The universal wrapper above preserves its prepared-only contract.
    static func trainingArtifacts(graph: GraphDocument, settings: ExecutionSettings,
                                  preparation: PreparedTrainingSelection?, compositeSource: CompositeSourceSnapshot?,
                                  trainingMode: WorkflowTrainingMode, requiresLifecycleEvidence: Bool,
                                  lifecycleParent: WorkflowLifecycleParentEvidence?, readRoot: URL, parts: [String],
                                  artifactPrefix: String, expectedCPUThreads: Int? = nil, sampleCapture: TrainingSampleCapturePolicy? = nil, optimizerRecipe: OptimizerRecipe? = nil, optimizerConfiguration: OptimizerConfiguration? = nil, schedulerRecipe: SchedulerRecipe? = nil, schedulerConfiguration: SchedulerConfiguration? = nil, initializationRecipe: InitializationConfiguration? = nil, objectiveSelection: PreparedObjectiveSelection? = nil, objectiveRecipe: LossRecipe? = nil, capturedModelRegistry: ComponentRegistry? = nil, capturedModelRegistrySHA256: String? = nil, gradientPolicy:GradientPolicyConfiguration? = nil, trainability:TrainabilityConfiguration? = nil, derivativeCapture:DerivativeCapturePolicy? = nil, preparedGradientEvidence:PreparedGradientEvidence? = nil, bagContext:WorkflowBagArtifactContext? = nil,datasetRecipe:DatasetRecipe? = nil,loaderRecipe:DataLoaderRecipe? = nil,trainOrigin:WorkflowNodeAddress? = nil,inlineRoles:Set<String> = []) throws -> Trained {
        // Do not retain the large inventory frame while resolving graph contracts
        // on the bounded cooperative worker used by History.
        let canonicalGraph=try JSONDecoder().decode(GraphDocument.self,from:encode(graph))
        // Resolve graph contracts before entering the large receipt-inventory frame.
        // History also invokes this path on 512 KB cooperative worker stacks.
        try verifyBagGraphContext(graph:canonicalGraph,registry:capturedModelRegistry,
            registryHash:capturedModelRegistrySHA256,context:objectiveSelection == nil ? .preparedWorkflow:.explicitObjective,
            bag:bagContext,sampleCapture:sampleCapture)
        let trained=try verifiedTrainingInventory(graph: canonicalGraph, settings: settings, preparation: preparation, compositeSource: compositeSource, trainingMode: trainingMode, requiresLifecycleEvidence: requiresLifecycleEvidence, lifecycleParent: lifecycleParent, readRoot: readRoot, parts: parts, artifactPrefix: artifactPrefix, expectedCPUThreads: expectedCPUThreads, sampleCapture: sampleCapture, optimizerRecipe: optimizerRecipe, optimizerConfiguration: optimizerConfiguration, schedulerRecipe: schedulerRecipe, schedulerConfiguration: schedulerConfiguration, initializationRecipe: initializationRecipe, objectiveSelection: objectiveSelection, objectiveRecipe: objectiveRecipe, capturedModelRegistry: capturedModelRegistry, capturedModelRegistrySHA256: capturedModelRegistrySHA256, gradientPolicy: gradientPolicy, trainability: trainability, derivativeCapture: derivativeCapture, preparedGradientEvidence: preparedGradientEvidence, bagContext: bagContext, datasetRecipe: datasetRecipe, loaderRecipe: loaderRecipe, trainOrigin: trainOrigin, inlineRoles: inlineRoles)
        if let value=trained.activationPreviews {
            guard case .object(let previews)=value else {throw WorkflowFileLeaseError.unsafePath}
            try WorkflowModelTrainingExecution.verifyActivationPreviews(previews,graph:canonicalGraph,sampleCapture:sampleCapture,
                objectiveContext:trained.objectiveContext,capturedRegistry:capturedModelRegistry,bagContext:bagContext,maximumEpoch:trained.trainingControl?.completedEpoch ?? settings.epochs)
        }
        return trained
    }

    @inline(never)
    private static func verifyBagGraphContext(graph:GraphDocument,registry:ComponentRegistry?,registryHash:String?,
        context:GraphValidationContext,bag:WorkflowBagArtifactContext?,sampleCapture:TrainingSampleCapturePolicy?)throws {
        guard let bag else {return}
        guard let registry,let registryHash,
              try WorkflowArchiveCodec.registryHash(registry)==registryHash,
              try WorkflowMappingRequirements.bagInputRequirement(graph,registry:registry,context:context)==bag.requirement,
              try WorkflowMappingRequirements.outputContract(graph,registry:registry)==bag.outputContract,
              sampleCapture == (bag.explicitSampleCapture ? bag.policy:nil) else {
            throw WorkflowPhaseError("Bag training requires its captured registry, actual source/output contracts and exact capture policy.")
        }
    }

    @inline(never)
    private static func verifiedTrainingInventory(graph: GraphDocument, settings: ExecutionSettings,
                                  preparation: PreparedTrainingSelection?, compositeSource: CompositeSourceSnapshot?,
                                  trainingMode: WorkflowTrainingMode, requiresLifecycleEvidence: Bool,
                                  lifecycleParent: WorkflowLifecycleParentEvidence?, readRoot: URL, parts: [String],
                                  artifactPrefix: String, expectedCPUThreads: Int? = nil, sampleCapture: TrainingSampleCapturePolicy? = nil, optimizerRecipe: OptimizerRecipe? = nil, optimizerConfiguration: OptimizerConfiguration? = nil, schedulerRecipe: SchedulerRecipe? = nil, schedulerConfiguration: SchedulerConfiguration? = nil, initializationRecipe: InitializationConfiguration? = nil, objectiveSelection: PreparedObjectiveSelection? = nil, objectiveRecipe: LossRecipe? = nil, capturedModelRegistry: ComponentRegistry? = nil, capturedModelRegistrySHA256: String? = nil, gradientPolicy:GradientPolicyConfiguration? = nil, trainability:TrainabilityConfiguration? = nil, derivativeCapture:DerivativeCapturePolicy? = nil, preparedGradientEvidence:PreparedGradientEvidence? = nil, bagContext:WorkflowBagArtifactContext? = nil,datasetRecipe:DatasetRecipe? = nil,loaderRecipe:DataLoaderRecipe? = nil,trainOrigin:WorkflowNodeAddress? = nil,inlineRoles:Set<String> = []) throws -> Trained {
        // The sole caller already canonicalizes the exact worker JSON. Do not
        // encode/decode it again while this large verification frame is live.
        // All receipt, composition and archived-graph comparisons remain intact.
        let bytes = try DiagnosticFileIO.read(root: readRoot, parts: parts + ["manifest.json"], limit: 20_000_000)
        let manifest = try JSONDecoder().decode(TrainingManifest.self, from: bytes)
        let optimizerObject = try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        let requestedSettings=settings
        let controlReport=try TrainingControlReport.verify(optimizerObject["trainingControl"],policy:settings.trainingControl,epochLimit:settings.epochs)
        var settings=settings
        if let controlReport {
            guard case .object(let saved)=optimizerObject["settings"],saved["epochLimit"] == .integer(Int64(settings.epochs)),case .object(let checkpoint)=optimizerObject["checkpoint"],case .object(let contract)=checkpoint["contract"],case .object(let recorded)=contract["trainingControl"],Set(recorded.keys)==["schemaVersion","policy"],recorded["schemaVersion"] == .integer(1),recorded["policy"] == (try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(controlReport.policy))) else {throw WorkflowPhaseError("Model selection contract differs from the accepted run.")}
            settings.epochs=controlReport.completedEpoch
        } else if case .object(let saved)=optimizerObject["settings"],saved["epochLimit"] != nil {throw WorkflowPhaseError("Undeclared epoch limit provenance.")}

        let validationContext:GraphValidationContext = objectiveSelection == nil ? .preparedWorkflow:.explicitObjective
        let dataLoadingIdentity:WorkflowDataLoadingIdentity?
        if datasetRecipe != nil || loaderRecipe != nil {
            guard let evidence=preparedGradientEvidence else {throw WorkflowPhaseError("Data loading requires independently verified original prepared rows.")}
            dataLoadingIdentity=try WorkflowDataLoadingVerifier.validate(manifest:optimizerObject,datasetRecipe:datasetRecipe,loaderRecipe:loaderRecipe,prepared:evidence.rows,capturedBatchSize:settings.batchSize,expectedEpochs:settings.epochs,trainOrigin:trainOrigin,inlineRoles:inlineRoles)
        } else {
            try WorkflowDataLoadingVerifier.validateAttachment(manifest:optimizerObject,identity:nil)
            dataLoadingIdentity=nil
        }

        if let bagContext {
            switch bagContext.target {
            case .binary:
                guard let preparation,bagContext.preparation.selection == .legacy(preparation),objectiveSelection == nil,objectiveRecipe == nil else {throw WorkflowPhaseError("Bag binary preparation differs from its selected partitions.")}
            case .objective(let selected,let recipe):
                guard preparation == nil,objectiveSelection == selected,objectiveRecipe == recipe else {throw WorkflowPhaseError("Bag objective differs from its selected partitions and recipe.")}
            }
            try WorkflowBagArtifacts.validateIdentity(manifest:optimizerObject,context:bagContext)
        } else if graph.nodes.contains(where:{$0.type == "bag_input"}) {
            throw WorkflowPhaseError("Bag training requires independently verified prepared packet evidence.")
        }
        let objectiveContext:WorkflowObjectiveArtifactContext?
        if bagContext != nil,objectiveSelection != nil {
            objectiveContext=nil
        } else if let objectiveSelection {
            guard preparation == nil,let objectiveRecipe,let capturedModelRegistry,let capturedModelRegistrySHA256,
                  try WorkflowArchiveCodec.registryHash(capturedModelRegistry) == capturedModelRegistrySHA256 else {throw WorkflowPhaseError("Objective verification requires its exact captured registry and typed preparation.")}
            let input=try WorkflowMappingRequirements.inputContract(graph,registry:capturedModelRegistry,context:.explicitObjective)
            let output=try WorkflowMappingRequirements.outputContract(graph,registry:capturedModelRegistry)
            objectiveContext=try WorkflowObjectiveIdentity.validate(optimizerObject,selection:objectiveSelection,recipe:objectiveRecipe,inputContract:input,outputContract:output,expectedEpochs:settings.epochs,policy:sampleCapture ?? .factory,explicitSampleCapture:sampleCapture != nil,dataLoading:dataLoadingIdentity,usesNamedValues:graph.namedInputs != nil)
        } else {
            guard objectiveRecipe == nil,case .object(let savedSettings)=optimizerObject["settings"],case .object(let savedProvenance)=optimizerObject["provenance"],case .object(let savedCheckpoint)=optimizerObject["checkpoint"],savedSettings["objectiveRecipe"] == nil,savedSettings["targetMapping"] == nil,savedSettings["objectiveScalarization"] == nil,savedProvenance["objective"] == nil,savedProvenance["objectiveRecipe"] == nil,savedProvenance["objectiveMapping"] == nil else {throw WorkflowPhaseError("Legacy training cannot acquire an unrequested objective.")}
            if case .object(let contract)=savedCheckpoint["contract"],contract["objective"] != nil {throw WorkflowPhaseError("Legacy checkpoint cannot acquire an unrequested objective.")}
            objectiveContext=nil
        }
        let objectiveEvaluationContext:WorkflowObjectiveEvaluationContext?
        if let bagContext,case .objective(let selected,let recipe)=bagContext.target {
            let common=try WorkflowObjectiveIdentity.validateEvaluationIdentity(optimizerObject,selection:selected,recipe:recipe,outputContract:bagContext.outputContract,expectedEpochs:settings.epochs,policy:bagContext.policy,explicitSampleCapture:bagContext.explicitSampleCapture,dataLoading:dataLoadingIdentity)
            guard common.runID == bagContext.runID,common.device == bagContext.device,
                  common.trainingCount == (dataLoadingIdentity?.retainedRows ?? bagContext.preparation.training.count),common.evaluationCount == bagContext.preparation.validation.count else {throw WorkflowPhaseError("Bag objective runtime or counts differ from verified preparation.")}
            objectiveEvaluationContext=common
        } else {objectiveEvaluationContext=try objectiveContext?.evaluationContext()}
        let gradientIdentity:WorkflowGradientPolicyIdentity?
        if gradientPolicy != nil || trainability != nil {
            guard let evidence=preparedGradientEvidence,let capturedModelRegistry else {throw WorkflowPhaseError("Gradient controls require captured model contracts and verified prepared rows.")}
            gradientIdentity=try WorkflowGradientPolicyVerifier.validate(manifest:optimizerObject,gradientPolicy:gradientPolicy,trainability:trainability,derivativeCapture:derivativeCapture,graph:graph,trainingRows:evidence.rows.trainingRows,batchSize:settings.batchSize,context:validationContext,capturedRegistry:capturedModelRegistry,dataLoading:dataLoadingIdentity)
        } else {
            try validateUnselectedGradientMetadata(optimizerObject,derivativeCapture:derivativeCapture)
            gradientIdentity=nil
        }
        if graph.compositionPlan != nil {
            guard let capturedModelRegistry else { throw WorkflowPhaseError("Composition verification requires the captured model registry.") }
            try WorkflowLifecycleVerifier.validateComposition(optimizerObject, graph: graph, registry: capturedModelRegistry, context: validationContext)
        } else {
            // Absence is also verified: a legacy model cannot acquire container state.
            try WorkflowLifecycleVerifier.validateComposition(optimizerObject, graph: graph, registry: capturedModelRegistry ?? ComponentRegistry.builtIn(), context: validationContext)
        }
        try WorkflowInitializationVerifier.validate(optimizerObject,expected:initializationRecipe,graph:graph,trainingMode:trainingMode,context:validationContext,capturedRegistry:capturedModelRegistry)
        try WorkflowLifecycleVerifier.validateOptimizer(optimizerObject,expected:optimizerRecipe,graph:graph,expectedConfiguration:optimizerConfiguration,context:validationContext,capturedRegistry:capturedModelRegistry,gradientIdentity:gradientIdentity)
        if case .object(let runtime)=optimizerObject["runtime"] {try WorkflowLifecycleVerifier.validateAttentionExecution(runtime,graph:graph,context:validationContext,capturedRegistry:capturedModelRegistry)}
        if let expectedCPUThreads {
            let object = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
            guard ExecutionResourcePolicy.cpuThreadBounds.contains(expectedCPUThreads),
                  case .object(let checkpoint) = object["checkpoint"],
                  case .object(let contract) = checkpoint["contract"], case .object(let schedule) = contract["schedule"],
                  schedule["numThreads"] == .integer(Int64(expectedCPUThreads)) else {
                throw WorkflowPhaseError("Checkpoint CPU thread schedule differs from the captured resource policy.")
            }
        }
        if requiresLifecycleEvidence {
            try WorkflowLifecycleVerifier.validate(JSONDecoder().decode([String:JSONValue].self,from:bytes),
                mode:trainingMode,targetEpochs:settings.epochs,parent:lifecycleParent,graph:graph,hasScheduler:schedulerRecipe != nil || schedulerConfiguration != nil,hasObjective:objectiveEvaluationContext != nil,hasGradientPolicy:gradientPolicy != nil,hasTrainability:trainability != nil,gradientIdentity:gradientIdentity,dataLoadingIdentity:dataLoadingIdentity)
        }
        guard manifest.schemaVersion == 1, manifest.status == "completed", UUID(uuidString: manifest.runID) != nil,
              try matchesSerializedGraph(manifest.graph,expected:graph),
              manifest.seed == settings.seed, manifest.settings.epochs == settings.epochs,
              manifest.settings.requestedDevice == settings.device, manifest.settings.requestedBatchSize == settings.batchSize,
              manifest.settings.learningRate == settings.learningRate,
              manifest.checkpoint.completedEpoch == settings.epochs, (1...100_000_000).contains(manifest.checkpoint.bytes),
              isHash(manifest.checkpoint.sha256), isHash(manifest.checkpoint.metricsSHA256) else {
            throw WorkflowPhaseError("Training manifest differs from the captured model, settings or prepared partitions.")
        }
        if let plan=preparation?.namedInputPlan {
            guard case .object(let provenance)=optimizerObject["provenance"],provenance["namedInputPlan"] == .object(plan.values) else {throw WorkflowPhaseError("Named input provenance differs from the captured mapping.")}
        }
        if objectiveEvaluationContext != nil || bagContext != nil {
            // Complete typed preparation identity was checked above; no legacy target is fabricated.
        } else if let preparation {
            guard let prepared = manifest.provenance.preparation, manifest.settings.partitionMode == "prepared",
                  prepared.manifestSHA256 == preparation.manifestSHA256, prepared.graphDigest == preparation.graphDigest,
                  prepared.trainingNode == preparation.trainingNode, prepared.trainingPort == preparation.trainingPort,
                  prepared.validationNode == preparation.validationNode, prepared.validationPort == preparation.validationPort,
                  manifest.provenance.featureColumns == preparation.featureColumns, manifest.provenance.targetColumn == preparation.targetColumn else {
                throw WorkflowPhaseError("Training manifest differs from the captured prepared partitions.")
            }
        } else {
            guard manifest.settings.partitionMode == "generatedSplit", manifest.provenance.preparation == nil else {
                throw WorkflowPhaseError("Training unexpectedly changed its partition mode.")
            }
        }
        var hashes: [String: String] = [:]
        var files: [WorkflowArchiveFile] = []
        var modelCode: String?
        var metrics: [EpochMetric] = []
        var schedulerMetrics: [[String:JSONValue]] = []
        for (name, limit) in [("checkpoint.pt", manifest.checkpoint.bytes), ("metrics.json", 8_000_000), ("model.py", 2_000_000), ("graph.json", 2_000_000)] {
            let content = try DiagnosticFileIO.read(root: readRoot, parts: parts + [name], limit: limit)
            guard !content.isEmpty else { throw WorkflowPhaseError("Training artifact is empty.") }
            let pin = hash(content)
            if name == "checkpoint.pt", (content.count != manifest.checkpoint.bytes || pin != manifest.checkpoint.sha256) { throw WorkflowPhaseError("Checkpoint hash or size differs from manifest.") }
            if name == "metrics.json", pin != manifest.checkpoint.metricsSHA256 { throw WorkflowPhaseError("Metrics hash differs from manifest.") }
            if name == "metrics.json" {
                schedulerMetrics = try JSONDecoder().decode([[String:JSONValue]].self, from: content)
                if let objectiveEvaluationContext {
                    metrics=try WorkflowObjectiveArtifacts.metrics(content,context:objectiveEvaluationContext).map {EpochMetric(epoch:$0.epoch,trainLoss:$0.trainLoss,validationLoss:$0.validationLoss,accuracy:$0.accuracy)}
                } else {
                    struct Metric: Decodable {let epoch:Int;let trainLoss:Double;let validationLoss:Double;let accuracy:Double}
                    let rows=try JSONDecoder().decode([Metric].self,from:content)
                    guard rows.count == settings.epochs,rows.enumerated().allSatisfy({index,row in row.epoch == index+1 && row.trainLoss.isFinite && row.validationLoss.isFinite && row.accuracy.isFinite && (0...1).contains(row.accuracy)}) else {throw WorkflowPhaseError("Saved metrics do not contain the requested finite epoch observations.")}
                    metrics=rows.map {EpochMetric(epoch:$0.epoch,trainLoss:$0.trainLoss,validationLoss:$0.validationLoss,accuracy:$0.accuracy)}
                }
            }
            if name == "model.py" {
                guard let text = String(data: content, encoding: .utf8) else { throw WorkflowPhaseError("Generated model code is not valid UTF-8.") }
                modelCode = text
            }
            if name == "graph.json", try JSONDecoder().decode(GraphDocument.self, from: content) != graph { throw WorkflowPhaseError("Archived graph differs from captured model.") }
            hashes[artifactPrefix + name] = pin
            files.append(.init(path: artifactPrefix + name, bytes: content.count, sha256: pin))
        }
        _=try TrainingControlReport.verify(optimizerObject["trainingControl"],policy:requestedSettings.trainingControl,epochLimit:requestedSettings.epochs,losses:metrics.map(\.validationLoss))
        let traceBytes = try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+["rate-trace.json"],limit:SchedulerConfiguration.maximumTraceBytes)
        let trace = try traceBytes.map {try JSONDecoder().decode(JSONValue.self,from:$0)}
        try WorkflowLifecycleVerifier.validateScheduler(optimizerObject,expected:schedulerRecipe,optimizer:optimizerRecipe,graph:graph,trace:trace,metrics:schedulerMetrics,expectedConfiguration:schedulerConfiguration,optimizerConfiguration:optimizerConfiguration,context:validationContext,capturedRegistry:capturedModelRegistry,gradientIdentity:gradientIdentity)
        if let traceBytes {
            guard case .object(let metadata)=optimizerObject["rateTrace"],metadata["bytes"] == .integer(Int64(traceBytes.count)),metadata["sha256"] == .string(hash(traceBytes)) else {throw WorkflowPhaseError("Scheduler rate trace hash or size differs from the manifest.")}
            hashes[artifactPrefix+"rate-trace.json"]=hash(traceBytes)
            files.append(.init(path:artifactPrefix+"rate-trace.json",bytes:traceBytes.count,sha256:hash(traceBytes)))
        }
        if optimizerRecipe != nil || optimizerConfiguration != nil {
            let recipeBytes=try DiagnosticFileIO.read(root:readRoot,parts:parts+["training_recipe.py"],limit:8192)
            guard !recipeBytes.isEmpty,String(data:recipeBytes,encoding:.utf8) != nil else {throw WorkflowPhaseError("Training recipe export must be nonempty UTF-8 within its 8192-byte limit.")}
            hashes[artifactPrefix+"training_recipe.py"]=hash(recipeBytes)
            files.append(.init(path:artifactPrefix+"training_recipe.py",bytes:recipeBytes.count,sha256:hash(recipeBytes)))
        }
        let initializationHelper=try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+["initialization_recipe.py"],limit:131072)
        if initializationRecipe != nil {
            guard let helper=initializationHelper,!helper.isEmpty,String(data:helper,encoding:.utf8) != nil else {throw WorkflowPhaseError("Starting-settings helper must be nonempty UTF-8 within its 131,072-byte limit.")}
            hashes[artifactPrefix+"initialization_recipe.py"]=hash(helper)
            files.append(.init(path:artifactPrefix+"initialization_recipe.py",bytes:helper.count,sha256:hash(helper)))
        } else if initializationHelper != nil {throw WorkflowPhaseError("A run without starting settings cannot acquire an unrequested initialization helper.")}
        for (name,selected) in [("gradient_policy.py",gradientPolicy != nil),("trainability_recipe.py",trainability != nil)] {
            let helper=try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+[name],limit:131072)
            if selected {
                guard let helper,!helper.isEmpty,String(data:helper,encoding:.utf8) != nil else {throw WorkflowPhaseError("Selected gradient control helper is missing, oversized or invalid UTF-8.")}
                hashes[artifactPrefix+name]=hash(helper)
                files.append(.init(path:artifactPrefix+name,bytes:helper.count,sha256:hash(helper)))
            } else if helper != nil {throw WorkflowPhaseError("Unselected gradient control cannot acquire a helper export.")}
        }
        var dataLoadingModelCode:String?
        let dataLoadingHelper=try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+["data_loading_recipe.py"],limit:131072)
        if dataLoadingIdentity != nil {
            guard let helper=dataLoadingHelper,!helper.isEmpty,let source=String(data:helper,encoding:.utf8) else {throw WorkflowPhaseError("Selected data loading helper is missing, oversized or invalid UTF-8.")}
            dataLoadingModelCode=source
            hashes[artifactPrefix+"data_loading_recipe.py"]=hash(helper)
            files.append(.init(path:artifactPrefix+"data_loading_recipe.py",bytes:helper.count,sha256:hash(helper)))
        } else if dataLoadingHelper != nil {throw WorkflowPhaseError("Unselected data loading cannot acquire a helper export.")}
        if let source = compositeSource {
            let content = try DiagnosticFileIO.read(root: readRoot, parts: parts + ["composite", "source.json"], limit: 2_000_000)
            guard try WorkflowArchiveCodec.decodeCompositeSource(content) == source else { throw WorkflowPhaseError("Archived Block source differs from capture.") }
            if source.nativeExpansionContractVersion == 2 {
                guard let capturedModelRegistry else { throw WorkflowPhaseError("Source3 requires its captured model registry for independent expansion.") }
                let expansion = try source.verifiedExpansion(registry: capturedModelRegistry, context: validationContext)
                guard try GraphSemantics.canonicalData(expansion.graph, registry: capturedModelRegistry) == GraphSemantics.canonicalData(graph, registry: capturedModelRegistry),
                      case .object(let provenance) = optimizerObject["provenance"], case .object(let composite) = provenance["composite"],
                      Set(composite.keys) == ["path","sha256","bytes","nativeExpansionContractVersion","expandedSemanticDigest","consumedDefinitions"],
                      composite["path"] == .string("composite/source.json"), composite["sha256"] == .string(hash(content)),
                      composite["bytes"] == .integer(Int64(content.count)), composite["nativeExpansionContractVersion"] == .integer(2),
                      composite["consumedDefinitions"] == (try JSONDecoder().decode(JSONValue.self, from: encode(expansion.consumedDefinitions))),
                      let digest = composite["expandedSemanticDigest"]?.stringValue, WorkflowArchiveCodec.isHash(digest),
                      case .object(let checkpoint) = optimizerObject["checkpoint"], case .object(let contract) = checkpoint["contract"],
                      contract["semanticGraphDigest"] == .string(digest) else { throw WorkflowPhaseError("Saved source3 provenance differs from independent expansion, source bytes or checkpoint architecture.") }
            }
            hashes[artifactPrefix + "composite/source.json"] = hash(content)
            files.append(.init(path: artifactPrefix + "composite/source.json", bytes: content.count, sha256: hash(content)))
        }
        let object = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        let partitionEvidence = preparation == nil && objectiveEvaluationContext == nil && bagContext == nil ? nil : try object["provenance"].map { try WorkflowPartitionEvidence.decode($0) } ?? nil
        var derivativeResults:DerivativeResults?
        let derivativeBytes=try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+["derivative-results.json"],limit:8_388_608)
        if let gradientPolicy,!gradientPolicy.requests.isEmpty,sampleCapture?.savesDiagnosticDetails != false {
            guard let derivativeCapture,let content=derivativeBytes,let metadata=object["derivativeResults"],let evidence=preparedGradientEvidence,let identity=gradientIdentity,let registry=capturedModelRegistry,case .object(let runtime)=object["runtime"],let deviceName=runtime["device"]?.stringValue,let device=ContractDevice(rawValue:deviceName),let version=runtime["torch"]?.stringValue else {throw WorkflowPhaseError("Selected derivative values require captured permission, actual prepared rows and saved runtime evidence.")}
            let checkpoint=try DiagnosticFileIO.read(root:readRoot,parts:parts+["checkpoint.pt"],limit:100_000_000)
            let graphBytes=try DiagnosticFileIO.read(root:readRoot,parts:parts+["graph.json"],limit:2_000_000)
            guard hash(checkpoint)==hashes[artifactPrefix+"checkpoint.pt"],hash(graphBytes)==hashes[artifactPrefix+"graph.json"] else {throw WorkflowPhaseError("Derivative source changed during artifact verification.")}
            guard case .object(let provenance)=object["provenance"],provenance["trainingRowIDs"] == .array(evidence.rows.trainingRowIDs.map(JSONValue.string)),provenance["validationRowIDs"] == .array(evidence.rows.validationRowIDs.map(JSONValue.string)) else {throw WorkflowPhaseError("Derivative partition order differs from the actual prepared tables.")}
            let bagDerivatives = try bagContext.map {try WorkflowBagDerivativeContext(requirement:$0.requirement,preparation:$0.preparation)}
            let geometry=try WorkflowDerivativeGeometry.resolve(policy:gradientPolicy,graph:graph,registry:registry,context:validationContext,identity:identity,bag:bagDerivatives)
            derivativeResults=try DerivativeResults.verify(content,metadata:metadata,context:.init(policy:gradientPolicy,capture:derivativeCapture,runID:manifest.runID,device:device,torchVersion:version,graph:graphBytes,checkpoint:checkpoint,preparationManifest:evidence.manifest,preparationGraphDigest:evidence.graphDigest,partitionRowIDs:["training":evidence.rows.trainingRowIDs,"validation":evidence.rows.validationRowIDs],geometry:geometry,bag:bagDerivatives))
            hashes[artifactPrefix+"derivative-results.json"]=hash(content)
            files.append(.init(path:artifactPrefix+"derivative-results.json",bytes:content.count,sha256:hash(content)))
        } else if derivativeBytes != nil || object["derivativeResults"] != nil {throw WorkflowPhaseError("A run without selected derivative requests cannot acquire derivative results.")}
        var gradientArtifact: GradientCaptureArtifact?
        if let configuration = settings.gradientCapture,sampleCapture?.savesDiagnosticDetails != false {
            guard let raw = object["gradientCapture"], case .object(let runtime) = object["runtime"],
                  case .string(let device) = runtime["device"] else { throw WorkflowPhaseError("Requested gradient evidence is missing.") }
            let metadata = try GradientCaptureCodec.decodeMetadata(encode(raw))
            let content = try DiagnosticFileIO.read(root: readRoot, parts: parts + ["gradient-summary.json"], limit: 8_000_000)
            let artifact = try GradientCaptureCodec.decodeArtifact(content, expectedRunID: manifest.runID, expectedDevice: device, metadata: metadata)
            gradientArtifact = artifact
            guard artifact.config == configuration else { throw WorkflowPhaseError("Gradient capture configuration differs from the request.") }
            hashes[artifactPrefix + "gradient-summary.json"] = hash(content)
            files.append(.init(path: artifactPrefix + "gradient-summary.json", bytes: content.count, sha256: hash(content)))
        } else if object["gradientCapture"] != nil { throw WorkflowPhaseError("Unrequested gradient evidence appeared in the manifest.") }
        if let sampleCapture {
            let expected = try JSONDecoder().decode(JSONValue.self, from: encode(sampleCapture))
            guard object["sampleCapture"] == expected else { throw WorkflowPhaseError("Saved training capture policy differs from the accepted request.") }
        }
        if sampleCapture?.savesComponentOutputs == false {
            for (field,file) in [("activationPreview","activations.json"),("predictionPreview","predictions.json")] {
                guard object[field]==nil,try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+[file],limit:2_000_000)==nil else {throw WorkflowPhaseError("Disabled History output capture retained optional samples.")}
            }
        }
        if sampleCapture?.savesDiagnosticDetails == false {
            for (field,file) in [("gradientCapture","gradient-summary.json"),("derivativeResults","derivative-results.json"),("performanceCapture","performance.json")] {
                guard object[field]==nil,try DiagnosticFileIO.readIfPresent(root:readRoot,parts:parts+[file],limit:8_388_608)==nil else {throw WorkflowPhaseError("Disabled History diagnostic capture retained optional observations.")}
            }
        }
        var routingTables:[String:WorkflowPredictionTableEvidence]=[:]
        if let raw=object["routingTables"] {
            guard sampleCapture?.savesDiagnosticDetails != false,graph.nodes.contains(where:{$0.type=="mixture_experts"}),case .object(let metadata)=raw,!metadata.isEmpty,Set(metadata.keys).isSubset(of:["routing","routingSample"]),let graphHash=object["graphHash"]?.stringValue else {throw WorkflowPhaseError("Undeclared routing capture or disabled diagnostic policy.")}
            for (port,value) in metadata {
                guard case .object(let pin)=value,Set(pin.keys)==["schemaVersion","file","bytes","sha256","rowCount"],pin["schemaVersion"]==1,pin["file"] == .string(port+".json"),let size=pin["bytes"]?.integerValue,let digest=pin["sha256"]?.stringValue,isHash(digest) else {throw WorkflowPhaseError("Routing table identity is invalid.")}
                let content=try DiagnosticFileIO.read(root:readRoot,parts:parts+[port+".json"],limit:2_000_000)
                let file=WorkflowArchiveFile(path:artifactPrefix+port+".json",bytes:size,sha256:digest)
                let table=try WorkflowRoutingTableEvidence.verify(content,file:file,port:port,runID:manifest.runID,graphHash:graphHash)
                guard pin["rowCount"] == .integer(Int64(table.rowCount)) else {throw WorkflowPhaseError("Routing row count differs from the captured table.")}
                routingTables[port]=table;hashes[file.path]=digest;files.append(file)
            }
        }
        var completePredictionTable:WorkflowPredictionTableEvidence?
        if let raw=object["predictionTable"] {
            guard sampleCapture?.predictionRows==true,case .object(let metadata)=raw,
                  Set(metadata.keys)==["schemaVersion","file","bytes","sha256","rowCount","partition"],metadata["schemaVersion"]==1,
                  metadata["file"] == .string("prediction-table.json"),metadata["partition"] == .string("validation"),
                  let sourceGraphHash=object["graphHash"]?.stringValue,isHash(sourceGraphHash),
                  let size=metadata["bytes"]?.integerValue,let pin=metadata["sha256"]?.stringValue,
                  case .object(let provenance)=object["provenance"],case .array(let ids)=provenance["validationRowIDs"],
                  ids.allSatisfy({$0.stringValue != nil}),metadata["rowCount"] == .integer(Int64(ids.count)),
                  case .object(let evaluation)=object["evaluation"],evaluation["count"] == .integer(Int64(ids.count)) else {throw WorkflowPhaseError("Complete prediction table identity or capture policy differs.")}
            let content=try DiagnosticFileIO.read(root:readRoot,parts:parts+["prediction-table.json"],limit:50_000_000)
            let file=WorkflowArchiveFile(path:artifactPrefix+"prediction-table.json",bytes:size,sha256:pin)
            completePredictionTable=try WorkflowPredictionTableEvidence.verify(content,file:file,partition:"validation",observationIDs:ids.compactMap(\.stringValue),sourceDigest:hash(Data((manifest.runID+"\n"+manifest.checkpoint.sha256+"\n"+sourceGraphHash).utf8)))
            hashes[file.path]=pin;files.append(file)
        }
        var predictionPreview: PredictionPreview?
        var objectivePredictionPreview: ObjectivePredictionPreview?
        var bagPredictionPreview: BagPredictionPreview?
        var objectiveEvaluation: ObjectiveEvaluation?
        var objectiveModelCode: String?
        if let metadata = object["predictionPreview"] {
            let content = try DiagnosticFileIO.read(root: readRoot, parts: parts + ["predictions.json"], limit: 1_000_000)
            guard case .object(let evaluationObject) = object["evaluation"] else { throw WorkflowPhaseError("Saved evaluation count is missing.") }
            let evaluation = evaluationObject["count"]
            if let bagContext {
                bagPredictionPreview=try WorkflowBagArtifacts.predictions(data:content,metadata:metadata,context:bagContext)
            } else if let objectiveContext {
                objectivePredictionPreview=try WorkflowObjectiveArtifacts.predictions(content,metadata:metadata,context:objectiveContext)
            } else {
                predictionPreview = try WorkflowPredictionArtifact.decode(content, metadata: metadata, runID: manifest.runID,
                    graph: graph, evaluationCount: evaluation)
            }
            hashes[artifactPrefix + "predictions.json"] = hash(content)
            files.append(.init(path: artifactPrefix + "predictions.json", bytes: content.count, sha256: hash(content)))
        }
        if let objectiveEvaluationContext {
            guard (sampleCapture?.savesComponentOutputs == false || objectivePredictionPreview != nil || bagPredictionPreview != nil),case .object(let evaluation)=object["evaluation"],let raw=evaluation["objective"] else {throw WorkflowPhaseError("Completed objective predictions or evaluation are missing.")}
            let result=try WorkflowObjectiveArtifacts.evaluation(raw,context:objectiveEvaluationContext)
            func near(_ a:Double?,_ b:Double)->Bool {guard let a,a.isFinite else {return false};return abs(a-b)<=max(1e-6,max(abs(a),abs(b))*1e-5)}
            guard near(evaluation["validationLoss"]?.doubleValue,result.effectiveScalar),near(controlReport.map {metrics[$0.selectedEpoch-1].validationLoss} ?? metrics.last?.validationLoss,result.effectiveScalar),evaluation["accuracy"]?.doubleValue == result.accuracy,evaluation["accuracy"] == nil || result.accuracy != nil else {throw WorkflowPhaseError("Final objective evaluation differs from its completed epoch observations.")}
            objectiveEvaluation=result
            let helper=try DiagnosticFileIO.read(root:readRoot,parts:parts+["objective_recipe.py"],limit:131072)
            guard !helper.isEmpty,let source=String(data:helper,encoding:.utf8) else {throw WorkflowPhaseError("Objective helper export is missing, oversized or invalid UTF-8.")}
            objectiveModelCode=source
            hashes[artifactPrefix+"objective_recipe.py"]=hash(helper)
            files.append(.init(path:artifactPrefix+"objective_recipe.py",bytes:helper.count,sha256:hash(helper)))
        }
        var activationPreviews: JSONValue?
        if let metadata = object["activationPreview"] {
            guard case .string(let graphHash) = object["graphHash"] else { throw WorkflowPhaseError("Saved activation graph identity is missing.") }
            let content = try DiagnosticFileIO.read(root:readRoot,parts:parts+["activations.json"],limit:2_000_000)
            activationPreviews = try WorkflowActivationArtifact.decodeIdentity(content,metadata:metadata,runID:manifest.runID,
                graphHash:graphHash,sampleCapture:sampleCapture)
            hashes[artifactPrefix+"activations.json"] = hash(content)
            files.append(.init(path:artifactPrefix+"activations.json",bytes:content.count,sha256:hash(content)))
        }
        var performanceCapture: WorkflowPerformanceCapture?
        if sampleCapture?.performanceMeasurements == true {
            guard let metadata = object["performanceCapture"], let policy = sampleCapture,
                  let graphHash = object["graphHash"]?.stringValue, case .object(let runtime) = object["runtime"],
                  let device = runtime["device"]?.stringValue else { throw WorkflowPhaseError("Requested performance observations are missing their captured identity.") }
            let content = try DiagnosticFileIO.read(root: readRoot, parts: parts+["performance.json"], limit: WorkflowPerformanceCapture.maximumBytes)
            performanceCapture = try WorkflowPerformanceCapture.decode(content, metadata: metadata, runID: manifest.runID,
                graphHash: graphHash, device: device, graph: graph, policy: policy)
            hashes[artifactPrefix+"performance.json"] = hash(content)
            files.append(.init(path: artifactPrefix+"performance.json", bytes: content.count, sha256: hash(content)))
        } else {
            guard object["performanceCapture"] == nil,
                  try DiagnosticFileIO.readIfPresent(root: readRoot, parts: parts+["performance.json"], limit: WorkflowPerformanceCapture.maximumBytes) == nil else {
                throw WorkflowPhaseError("Unrequested performance capture cannot be published.")
            }
        }
        let pin = hash(bytes); hashes[artifactPrefix + "manifest.json"] = pin
        files.append(.init(path: artifactPrefix + "manifest.json", bytes: bytes.count, sha256: pin))
        guard let modelCode else { throw WorkflowPhaseError("Verified model code is missing.") }
        if let sampleCapture,sampleCapture.savesComponentOutputs {
            let captured:Bool
            if bagContext != nil {captured=bagPredictionPreview?.captureStatus == (sampleCapture.predictionRows ? .captured:.omittedByPolicy)}
            else {captured = objectiveContext == nil ? predictionPreview?.captureStatus == (sampleCapture.predictionRows ? .captured : .omittedByPolicy) : objectivePredictionPreview?.captureStatus == (sampleCapture.predictionRows ? .captured : .omittedByPolicy)}
            guard captured else {
                throw WorkflowPhaseError("Saved predictions differ from the accepted sample capture policy.")
            }
        }
        var trained=Trained(partitionEvidence: partitionEvidence, manifestSHA256: pin, hashes: hashes, artifactFiles: files.sorted { $0.path < $1.path }, workerRunID: manifest.runID, modelCode: modelCode, predictionPreview: predictionPreview, activationPreviews: activationPreviews, gradientArtifact: gradientArtifact, metrics: metrics, performanceCapture: performanceCapture, derivativeResults: derivativeResults, objectiveContext: objectiveContext, objectivePredictionPreview: objectivePredictionPreview, objectiveEvaluation: objectiveEvaluation, objectiveModelCode: objectiveModelCode,bagContext:bagContext,bagPredictionPreview:bagPredictionPreview,objectiveEvaluationContext:objectiveEvaluationContext,dataLoadingIdentity:dataLoadingIdentity,dataLoadingModelCode:dataLoadingModelCode)
        trained.trainingControl=controlReport
        trained.routingTables=routingTables
        trained.predictionTable=completePredictionTable
        return trained
    }
    /// Python resolves macOS /var aliases. Compare physical locations, but never use
    /// the reported path to choose files: all reads remain beneath the owned root.
    static func matchesOutputPath(_ value: Any?, expected: URL) -> Bool {
        guard let path = value as? String, path.hasPrefix("/"), !path.contains("\0"), path.utf8.count <= 16_384 else { return false }
        return URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
            == expected.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private static func validName(_ name: String) -> Bool { !name.isEmpty && name != "." && name != ".." && !name.contains("/") && name.utf8.count <= 512 }
    private static func isHash(_ value: String) -> Bool { value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
}
