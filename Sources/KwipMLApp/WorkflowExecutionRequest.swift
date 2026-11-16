import Foundation
import CoreFoundation
import CryptoKit
import KwipMLCore

/// An in-process request, not a portable project or a cache key. Local invocation URLs
/// are intentionally retained here; exported provenance must redact them separately.
struct WorkflowExecutionRequest: Equatable {
    let lease: WorkflowEditorLease
    let document: WorkflowDocument
    let authoringDocument: WorkflowDocument
    let experiment: WorkflowExperimentRequestIdentity?
    let experimentInputs: WorkflowExperimentInputs?
    let train: WorkflowNodeAddress
    let plan: WorkflowExecutionPlan
    let modelRegistrySHA256: String
    let dataRegistrySHA256: String
    let runtime: RuntimeLocation
    let runtimeEvidence: WorkflowRuntimeEvidence
    let requestIdentity: String
    let priorCache: WorkflowCapturedCache?
    let gradientArguments: [String]
    let checkpointParent: WorkflowCheckpointReference?
    let executionBudget: ResolvedExecutionBudget?
    let sampleCapture: TrainingSampleCapturePolicy?
    let preparationCacheLimitBytes: Int?
    var outputDestinationPaths:[String:String] = [:]
    var capturedModelRegistry:ComponentRegistry? = nil
    var derivativeCapture:DerivativeCapturePolicy? = nil
    var moduleCompositionDescriptor:JSONValue? = nil
    var inspectionViewsIncluded=false
    var archiveCompilerContractVersion:Int {inspectionViewsIncluded ? 16:WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:document.schemaVersion)}

    @MainActor static func capture(from owner: WorkflowAuthoringStore, train: WorkflowNodeAddress,
                                  runtime: RuntimeLocation, priorCache: WorkflowCapturedCache? = nil, supportedGradientDevices: [String] = [], checkpointParent: WorkflowCheckpointReference? = nil, executionBudget: ResolvedExecutionBudget? = nil, sampleCapture: TrainingSampleCapturePolicy? = nil, preparationCacheLimitBytes: Int? = nil, derivativeCapture: DerivativeCapturePolicy? = nil, moduleCompositionEvidence: ModuleCompositionDiscoveryEvidence? = nil, outputDestinationPaths:[String:String] = [:]) throws -> Self {
        guard sampleCapture == nil || executionBudget != nil else {
            throw WorkflowRequestError("Explicit training sample capture requires a captured execution budget.")
        }
        let lease = owner.lease()
        guard owner.isCurrent(lease), !owner.isBusy else {
            throw WorkflowRequestError("Stop execution and use the active workflow editor before capturing a request.")
        }
        guard !owner.hasPendingDataEdits else {
            throw WorkflowRequestError("Resolve pending numeric edits before capturing workflow execution.")
        }
        let document = owner.document
        guard document.drafts.isEmpty else { throw WorkflowRequestError("Apply or discard the open Block draft before execution.") }
        // No suspension between the owner lease, source and immutable registry captures.
        let modelRegistry = owner.modelRegistry, dataRegistry = owner.dataRegistry
        let plan = try WorkflowCompiler.compile(document, train: train, modelRegistry: modelRegistry, dataRegistry: dataRegistry,
            contractVersion: document.schemaVersion>=13 ? 16:max(2,WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:document.schemaVersion)))
        guard !plan.dataflow.nodes.contains(where: {$0.operatorID == OCRConfiguration.operatorID}) else {
            throw WorkflowRequestError("Extract text is supported for data workflows and document classification. For neural-network training, first export its Documents text through Table Output, then use that captured table as input. Direct OCR pack capture for this training engine is not yet qualified.")
        }
        let sinks=Set(plan.dataflow.nodes.filter { $0.operatorID == "kwip.data.outputDelimited" && !APITableSettings.isExternal($0.config)}.map(\.id))
        guard Set(outputDestinationPaths.keys)==sinks,outputDestinationPaths.values.allSatisfy({$0.hasPrefix("/") && !$0.contains("\0") && $0.utf8.count<=4096}) else {
            throw WorkflowRequestError("Choose a private destination folder for every connected table output before running.")
        }
        let requiresPredictions=document.edges.contains {$0.from.address==train && $0.from.port=="predictions"}
        let inheritedCapture=sampleCapture ?? (owner.runHistoryPolicy == .all ? nil:.factory)
        let sampleCapture=inheritedCapture.map {$0.applyingHistory(outputs:owner.runHistoryPolicy.componentOutputs,diagnostics:owner.runHistoryPolicy.metricsAndDiagnostics,requiresPredictionTable:requiresPredictions)}
        // Capture permission with the selected policy, never infer it from activation
        // or gradient-summary capture. Unselected settings retain the old identity.
        let capturedDerivative: DerivativeCapturePolicy?
        if let recipe = plan.gradientPolicyRecipe {
            guard executionBudget != nil else {
                throw WorkflowRequestError("Gradient policy execution requires a captured execution budget and derivative permission.")
            }
            let permission = derivativeCapture ?? .factory
            try permission.validate(gradientPolicy: recipe.configuration)
            capturedDerivative = permission
        } else { capturedDerivative = nil }
        if let limit = preparationCacheLimitBytes, !ExecutionPreferences.preparationCacheByteBounds(workload: executionBudget?.resources?.workloadPolicy).contains(limit) {
            throw WorkflowRequestError("Preparation cache allowance must fit the captured retained-data allowance.")
        }
        let capturedCacheLimit = plan.preparationCachePolicy == .off ? nil : preparationCacheLimitBytes
        guard (plan.trainingMode != .fresh) == (checkpointParent != nil) else {
            throw WorkflowRequestError("Continuation requires a selected checkpoint parent; fresh training cannot consume one.")
        }
        if let parent=checkpointParent {
            guard parent.projectID == document.projectID, UUID(uuidString:parent.projectID) != nil,
                  UUID(uuidString:parent.directoryID) != nil,parent.hasValidPins else {
                throw WorkflowRequestError("Select a checkpoint from this project with valid run, request and model-manifest identities.")
            }
        }
        guard (plan.preparationCachePolicy == .reuse) == (priorCache != nil) else {
            throw WorkflowRequestError("Cache reuse requires a selected prior preparation result; off and materialize cannot consume one.")
        }
        if let priorCache {
            guard UUID(uuidString: priorCache.reference.projectID) == UUID(uuidString: document.projectID) else {
                throw WorkflowRequestError("Select a prior preparation result from this project.")
            }
            try priorCache.revalidate()
        }
        let gradientArguments = try GradientRunArguments.arguments(config: sampleCapture?.savesDiagnosticDetails == false ? nil:plan.settings.gradientCapture, supportedDevices: supportedGradientDevices, device: plan.settings.device)
        let modelHash = try workflowDigest(modelRegistry), dataHash = try workflowDigest(dataRegistry)
        let evidence = try WorkflowRuntimeEvidence.capture(runtime)
        let compositionDescriptor: JSONValue?
        if [11,12,13,14,15,16,17,18,19,20,21,22].contains(document.schemaVersion) {
            guard let moduleCompositionEvidence else { throw WorkflowRequestError("Refresh the selected runtime to verify native module composition before capturing this workflow.") }
            compositionDescriptor = try moduleCompositionEvidence.verifiedDescriptor(for: runtime, evidence: evidence)
        } else { compositionDescriptor = nil }
        // Compare decoded contracts, not JSON whitespace, while pinning these same
        // bytes to the source inventory observed above.
        let modelParts = runtime.kind == .bundled ? ["components.json"] : ["Sources", "KwipMLCore", "Resources", "components.json"]
        let modelBytes = try DiagnosticFileIO.read(root: runtime.workingDirectoryURL, parts: modelParts, limit: 2_000_000)
        let dataBytes = try DiagnosticFileIO.read(root: runtime.workingDirectoryURL, parts: ["runner", "dataflow-operators.json"], limit: 2_000_000)
        guard workflowHash(modelBytes) == evidence.workerFiles["components.json"],
              workflowHash(dataBytes) == evidence.workerFiles["runner/dataflow-operators.json"],
              try workflowDigest(JSONDecoder().decode(ComponentRegistry.self, from: modelBytes)) == modelHash,
              try workflowDigest(JSONDecoder().decode(DataFlowOperatorRegistry.self, from: dataBytes)) == dataHash else {
            throw WorkflowRequestError("The workflow editor registry differs from the captured worker registry. Reload a matching runtime and editor before execution.")
        }
        struct Identity: Encodable {
            let schemaVersion: Int
            let priorCache: WorkflowArchiveCacheReference?
            let checkpointParent: WorkflowCheckpointReference?
            let document: WorkflowDocument
            let train: WorkflowNodeAddress
            let modelRegistrySHA256: String
            let dataRegistrySHA256: String
            let runtimeEvidence: WorkflowRuntimeEvidence
            let moduleCompositionDescriptor: JSONValue?
            let executable: String
            let arguments: [String]
            let registry: String
            let workingDirectory: String
            let executionBudget: ResolvedExecutionBudget?
            let sampleCapture: TrainingSampleCapturePolicy?
            let derivativeCapture: DerivativeCapturePolicy?
            let preparationCacheLimitBytes: Int?
            let outputDestinationPaths:[String:String]?
        }
        let identity = try workflowDigest(Identity(schemaVersion: document.schemaVersion, priorCache: priorCache?.reference, checkpointParent: checkpointParent, document: document, train: train,
            modelRegistrySHA256: modelHash, dataRegistrySHA256: dataHash, runtimeEvidence: evidence, moduleCompositionDescriptor: compositionDescriptor,
            executable: runtime.executableURL.path, arguments: runtime.argumentPrefix,
            registry: runtime.registryURL.path, workingDirectory: runtime.workingDirectoryURL.path, executionBudget: executionBudget, sampleCapture: sampleCapture, derivativeCapture: capturedDerivative, preparationCacheLimitBytes: capturedCacheLimit, outputDestinationPaths:outputDestinationPaths.isEmpty ? nil:outputDestinationPaths))
        return Self(lease: lease, document: document, authoringDocument: document, experiment: nil, experimentInputs: nil, train: train, plan: plan,
            modelRegistrySHA256: modelHash, dataRegistrySHA256: dataHash, runtime: runtime,
            runtimeEvidence: evidence, requestIdentity: identity, priorCache: priorCache, gradientArguments: gradientArguments, checkpointParent: checkpointParent, executionBudget: executionBudget, sampleCapture: sampleCapture, preparationCacheLimitBytes: capturedCacheLimit, outputDestinationPaths:outputDestinationPaths, capturedModelRegistry:modelRegistry, derivativeCapture:capturedDerivative, moduleCompositionDescriptor:compositionDescriptor,inspectionViewsIncluded:document.schemaVersion>=13)
    }

    /// Derive only a declared variant of this captured source. The live canvas is
    /// neither replaced nor temporarily edited to satisfy a worker request.
    @MainActor func experimentVariant(ordinal:Int,inputs:WorkflowExperimentInputs,
                                      owner:WorkflowAuthoringStore,supportedGradientDevices:[String]=[])throws->Self {
        guard (1...32).contains(ordinal),let result=try experimentVariants(inputs:inputs,owner:owner,
            supportedGradientDevices:supportedGradientDevices).first(where:{$0.experiment?.ordinal == ordinal}) else {
            throw WorkflowRequestError("Choose a declared experiment variant.")
        }
        return result
    }

    /// Capture a whole group in one pass: shared input/runtime verification occurs
    /// once here; sessions still revalidate before each worker phase.
    @MainActor func experimentVariants(inputs:WorkflowExperimentInputs,owner:WorkflowAuthoringStore,
                                       supportedGradientDevices:[String]=[])throws->[Self] {
        try validateExperimentOwner(owner)
        return try deriveExperimentVariants(inputs:inputs,modelRegistry:owner.modelRegistry,
            dataRegistry:owner.dataRegistry,supportedGradientDevices:supportedGradientDevices)
    }

    @MainActor func experimentVariantsInBackground(inputs:WorkflowExperimentInputs,owner:WorkflowAuthoringStore,
                                                    supportedGradientDevices:[String]=[]) async throws->[Self] {
        try Task.checkCancellation()
        try validateExperimentOwner(owner)
        let modelRegistry=owner.modelRegistry,dataRegistry=owner.dataRegistry
        let worker=Task.detached(priority:.userInitiated) {
            try deriveExperimentVariants(inputs:inputs,modelRegistry:modelRegistry,
                dataRegistry:dataRegistry,supportedGradientDevices:supportedGradientDevices)
        }
        let requests=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        try Task.checkCancellation()
        try validateExperimentOwner(owner)
        return requests
    }

    @MainActor private func validateExperimentOwner(_ owner:WorkflowAuthoringStore)throws {
        guard experiment == nil,isCurrent(in:owner),!owner.isBusy,!owner.hasPendingInspectorEdits,
              document.drafts.isEmpty,checkpointParent == nil else {
            throw WorkflowRequestError("Capture experiments from the unchanged, idle authoring source.")
        }
    }

    private func deriveExperimentVariants(inputs:WorkflowExperimentInputs,modelRegistry:ComponentRegistry,
                                           dataRegistry:DataFlowOperatorRegistry,supportedGradientDevices:[String])throws->[Self] {
        try Task.checkCancellation()
        guard outputDestinationPaths.isEmpty, !plan.dataflow.nodes.contains(where:{$0.operatorID == "kwip.data.outputDelimited" && APITableSettings.isExternal($0.config)}) else {throw WorkflowRequestError("Table file outputs are not supported in experiment groups yet. Run a single flow, or remove the output before comparing variants.")}
        let runAuthority=try WorkflowTrainRunSetup.resolve(in:document,train:train)
        guard let value=runAuthority.configuration["experiment"] else {
            throw WorkflowRequestError("Save experiment combinations in Train first.")
        }
        let specification=try WorkflowExperimentSpecification.decode(value)
        let grid=try WorkflowExperimentPlan.build(document,train:train,axes:specification.axes,budget:specification.budget,
            modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        guard Set(inputs.sources.keys) == Set(plan.sourceAddresses),
              grid.variants.allSatisfy({$0.execution.sourceAddresses == plan.sourceAddresses}) else {
            throw WorkflowRequestError("Choose a declared variant and its exact captured sources.")
        }
        try Task.checkCancellation()
        try inputs.verify()
        guard try runtimeEvidence.matchesCurrent(runtime) else {
            throw WorkflowRequestError("The runtime changed after the experiment source was captured.")
        }
        return try grid.variants.map {variant in
            try Task.checkCancellation()
            let gradientArguments=try GradientRunArguments.arguments(config:variant.execution.settings.gradientCapture,
                supportedDevices:supportedGradientDevices,device:variant.execution.settings.device)
            let binding=WorkflowExperimentRequestIdentity(groupID:inputs.directory.lastPathComponent,
                inputsManifestSHA256:inputs.manifestSHA256,sourceRequestIdentity:requestIdentity,ordinal:variant.ordinal)
            let identity=try Self.experimentIdentity(binding:binding,document:variant.document)
            return Self(lease:lease,document:variant.document,authoringDocument:authoringDocument,experiment:binding,experimentInputs:inputs,
                train:train,plan:inspectionViewsIncluded ? try WorkflowCompiler.compile(variant.document,train:train,modelRegistry:modelRegistry,dataRegistry:dataRegistry,contractVersion:16):variant.execution,modelRegistrySHA256:modelRegistrySHA256,dataRegistrySHA256:dataRegistrySHA256,
                runtime:runtime,runtimeEvidence:runtimeEvidence,requestIdentity:identity,priorCache:priorCache,
                gradientArguments:gradientArguments,checkpointParent:nil,executionBudget:executionBudget,sampleCapture:sampleCapture,preparationCacheLimitBytes:preparationCacheLimitBytes,capturedModelRegistry:capturedModelRegistry,derivativeCapture:derivativeCapture,moduleCompositionDescriptor:moduleCompositionDescriptor,inspectionViewsIncluded:inspectionViewsIncluded)
        }
    }

    static func experimentIdentity(binding:WorkflowExperimentRequestIdentity,document:WorkflowDocument)throws->String {
        struct Identity:Encodable {
            let experiment:WorkflowExperimentRequestIdentity
            let document:WorkflowDocument
        }
        return try workflowDigest(Identity(experiment:binding,document:document))
    }

    @MainActor func isCurrent(in owner: WorkflowAuthoringStore?) -> Bool {
        owner?.isCurrent(lease) == true && owner?.document == authoringDocument
    }
}

/// Persist this binding with the group record; per-run workflow archives still
/// contain their exact derived configuration and immutable input copies.
struct WorkflowExperimentRequestIdentity:Codable,Equatable {
    let groupID:String
    let inputsManifestSHA256:String
    let sourceRequestIdentity:String
    let ordinal:Int
}

/// Source-content observation only. Neither a signature nor a full environment hash:
/// the interpreter, installed native packages, device and process environment are not
/// hashed. Package versions are manifest claims, not independently probed versions.
/// Revalidate before each execution phase; capture alone cannot lock the filesystem.
struct WorkflowRuntimeEvidence: Codable, Equatable {
    let schemaVersion: Int
    let kind: String
    let packageVersions: [String: String]
    let manifestSHA256: String?
    let workerFiles: [String: String]
    let workerInventorySHA256: String
    let coverage: String

    private static let maximumEntries = 512
    private static let maximumFileBytes = 10_000_000
    private static let maximumTotalBytes = 32_000_000

    func matchesCurrent(_ runtime: RuntimeLocation) throws -> Bool { try Self.capture(runtime) == self }

    static func capture(_ runtime: RuntimeLocation) throws -> Self {
        let root = runtime.workingDirectoryURL
        let bundled = runtime.kind == .bundled
        // Same inventory boundary as scripts/runtime_launch.py; development exclusions
        // mirror shutil.ignore_patterns in scripts/runtime_package.py, at every depth.
        var paths: [(name: String, parts: [String])] = bundled
            ? [("launch.py", ["launch.py"]), ("components.json", ["components.json"])]
            : [("launch.py", ["scripts", "runtime_launch.py"]),
               ("components.json", ["Sources", "KwipMLCore", "Resources", "components.json"])]
        var pending = [["runner"]], entryCount = 0
        while let parts = pending.popLast() {
            let directory = parts.reduce(root) { $0.appendingPathComponent($1) }
            let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else {
                throw WorkflowRequestError("Worker inventory directories must be real directories.")
            }
            var enumerationError: Error?
            guard let entries = FileManager.default.enumerator(at: directory,
                includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey],
                options: [.skipsSubdirectoryDescendants], errorHandler: { _, error in
                    enumerationError = error; return false
                }) else { throw WorkflowRequestError("Worker directory cannot be enumerated.") }
            var enumerated = 0
            while let entry = entries.nextObject() as? URL {
                enumerated += 1
                guard enumerated <= maximumEntries else { throw WorkflowRequestError("Worker inventory exceeds its entry limit.") }
                let name = entry.lastPathComponent
                if !bundled && (["tests", "__pycache__", ".pytest_cache"].contains(name) || name.hasSuffix(".pyc")) { continue }
                entryCount += 1
                guard entryCount <= maximumEntries else { throw WorkflowRequestError("Worker inventory exceeds its entry limit.") }
                let info = try entry.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
                guard info.isSymbolicLink != true else { throw WorkflowRequestError("Worker symbolic links are not accepted.") }
                let child = parts + [name]
                if info.isDirectory == true { pending.append(child) }
                else if info.isRegularFile == true { paths.append((child.joined(separator: "/"), child)) }
                else { throw WorkflowRequestError("Worker inventory contains a nonregular file.") }
            }
            if let enumerationError { throw enumerationError }
        }
        var hashes: [String: String] = [:], total = 0
        for path in paths.sorted(by: { $0.name < $1.name }) {
            let bytes = try DiagnosticFileIO.read(root: root, parts: path.parts, limit: maximumFileBytes)
            total += bytes.count
            guard total <= maximumTotalBytes else { throw WorkflowRequestError("Worker source exceeds its total byte limit.") }
            hashes[path.name] = workflowHash(bytes)
        }
        guard hashes["runner/main.py"] != nil else { throw WorkflowRequestError("Worker entry point is missing.") }
        var manifestHash: String?
        if bundled {
            let bytes = try DiagnosticFileIO.read(root: root, parts: ["runtime-manifest.json"], limit: 65_536)
            let manifest = try JSONDecoder().decode(Manifest.self, from: bytes)
            guard manifest.schemaVersion == 1, manifest.workerIntegrityVersion == 1,
                  manifest.workerSourceHashes.count >= 3, manifest.workerSourceHashes.count <= maximumEntries + 2,
                  manifest.workerSourceHashes == hashes,
                  manifest.entryHash == nil || manifest.entryHash == hashes["launch.py"],
                  manifest.registryHash == nil || manifest.registryHash == hashes["components.json"],
                  manifest.packages == runtime.packageVersions,
                  manifest.entryPoint == "launch.py", manifest.registry == "components.json",
                  root.appendingPathComponent(manifest.pythonExecutable).standardizedFileURL == runtime.executableURL.standardizedFileURL,
                  runtime.entryURL.standardizedFileURL == root.appendingPathComponent("launch.py").standardizedFileURL,
                  runtime.registryURL.standardizedFileURL == root.appendingPathComponent("components.json").standardizedFileURL else {
                throw WorkflowRequestError("Bundled worker inventory or invocation differs from its integrity manifest. Rebuild or reinstall the runtime.")
            }
            manifestHash = workflowHash(bytes)
        } else {
            guard runtime.entryURL.standardizedFileURL == root.appendingPathComponent("runner/main.py").standardizedFileURL,
                  runtime.registryURL.standardizedFileURL == root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json").standardizedFileURL else {
                throw WorkflowRequestError("Development invocation does not match the observed worker source.")
            }
        }
        if !bundled {
            let nativeParts=[".local","native-watchdog"]
            let libraryParts=nativeParts+["lifetime_watchdog.dylib"],manifestParts=nativeParts+["lifetime_watchdog.json"]
            func present(_ parts:[String])->Bool {
                let path=parts.reduce(root) {$0.appendingPathComponent($1)}
                // attributesOfItem observes dangling symlinks too; a malformed
                // staged pair must not silently become the legacy absent case.
                return (try? FileManager.default.attributesOfItem(atPath:path.path)) != nil
            }
            if present(libraryParts) || present(manifestParts) {
                let manifest=try DiagnosticFileIO.read(root:root,parts:manifestParts,limit:16384)
                guard let object=try JSONSerialization.jsonObject(with:manifest) as? [String:Any],
                      Set(object.keys) == ["schemaVersion","sourceSHA256","librarySHA256"],
                      let version=object["schemaVersion"] as? NSNumber,
                      CFGetTypeID(version) != CFBooleanGetTypeID(),
                      !["f","d"].contains(String(cString:version.objCType)),version == 1,
                      let sourceHash=object["sourceSHA256"] as? String,
                      let libraryHash=object["librarySHA256"] as? String,
                      WorkflowArchiveCodec.isHash(sourceHash),WorkflowArchiveCodec.isHash(libraryHash),
                      sourceHash == hashes["runner/native/lifetime_watchdog.c"] else {
                    throw WorkflowRequestError("Development lifetime guard attestation differs from the captured source. Rebuild the development app.")
                }
                let library=try DiagnosticFileIO.read(root:root,parts:libraryParts,limit:maximumFileBytes)
                guard workflowHash(library) == libraryHash else {
                    throw WorkflowRequestError("Development lifetime guard library differs from its attestation. Rebuild the development app.")
                }
                total += manifest.count+library.count
                guard total <= maximumTotalBytes,hashes.count+2 <= maximumEntries+2 else {
                    throw WorkflowRequestError("Development lifetime guard exceeds runtime evidence bounds.")
                }
                hashes[libraryParts.joined(separator:"/")]=libraryHash
                hashes[manifestParts.joined(separator:"/")]=workflowHash(manifest)
            }
        }
        return Self(schemaVersion: 1, kind: runtime.kind.rawValue, packageVersions: runtime.packageVersions,
            manifestSHA256: manifestHash, workerFiles: hashes, workerInventorySHA256: try workflowDigest(hashes),
            coverage: bundled ? "manifestVerifiedWorkerSourceOnly" : "developmentObservedWorkerSourceOnly")
    }

    private struct Manifest: Decodable {
        let schemaVersion: Int
        let workerIntegrityVersion: Int
        let workerSourceHashes: [String: String]
        let entryHash: String?
        let registryHash: String?
        let packages: [String: String]
        let pythonExecutable: String
        let entryPoint: String
        let registry: String

        private enum CodingKeys: String, CodingKey {
            case schemaVersion, workerIntegrityVersion, workerSourceHashes, entryHash, registryHash
            case packages, pythonExecutable, entryPoint, registry
        }
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
            workerIntegrityVersion = try values.decode(Int.self, forKey: .workerIntegrityVersion)
            workerSourceHashes = try values.decode([String: String].self, forKey: .workerSourceHashes)
            // Python's launcher checks a present key, including null. decodeIfPresent
            // would incorrectly collapse null and absence into the same contract.
            entryHash = values.contains(.entryHash) ? try values.decode(String.self, forKey: .entryHash) : nil
            registryHash = values.contains(.registryHash) ? try values.decode(String.self, forKey: .registryHash) : nil
            packages = try values.decode([String: String].self, forKey: .packages)
            pythonExecutable = try values.decode(String.self, forKey: .pythonExecutable)
            entryPoint = try values.decode(String.self, forKey: .entryPoint)
            registry = try values.decode(String.self, forKey: .registry)
        }
    }
}

private func workflowDigest<T: Encodable>(_ value: T) throws -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    return workflowHash(try encoder.encode(value))
}
private func workflowHash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
private struct WorkflowRequestError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
