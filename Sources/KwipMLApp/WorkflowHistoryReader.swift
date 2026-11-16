import Foundation
import Darwin
import KwipMLCore

struct WorkflowHistoryReference: Hashable {
    let projectID: String
    let directoryID: UUID
}
enum WorkflowHistoryVerification: Equatable { case recordedV2, legacyV1, incomplete, unavailable }
struct WorkflowHistoryEntry {
    let reference: WorkflowHistoryReference
    let status: WorkflowExecutionPhase?
    let runID: String?
    let finishedAt: String?
    let verification: WorkflowHistoryVerification
    let message: String?
}
struct WorkflowHistoryListing {
    let entries: [WorkflowHistoryEntry]
    let issues: [String]
    let truncated: Bool
}
struct WorkflowLegacyOutcome: Codable, Equatable {
    let schemaVersion: Int
    let runID: String
    let projectID: String
    let requestIdentity: String
    let status: WorkflowExecutionPhase
    let failedPhase: WorkflowExecutionPhase?
    let preparationManifestSHA256: String?
    let modelManifestSHA256: String?
    let artifactHashes: [String: String]
    let sourceHashes: [String: String]
    let error: String?
}
struct WorkflowVerifiedModelContent {
    let reference: WorkflowHistoryReference
    let outcome: WorkflowArchiveOutcome
    let capture: WorkflowArchiveCapture
    let code: String
    let metrics: [EpochMetric]
    var trainingControl:TrainingControlReport? = nil
    var trainingProgress:TrainingProgressArchive? = nil
    var partitionEvidence: WorkflowPartitionEvidence? = nil
    var predictionPreview: PredictionPreview? = nil
    var activationPreviews: JSONValue? = nil
    var gradientArtifact: GradientCaptureArtifact? = nil
    var performanceCapture: WorkflowPerformanceCapture? = nil
    var derivativeResults: DerivativeResults? = nil
    var objectiveContext: WorkflowObjectiveArtifactContext? = nil
    var objectivePredictionPreview: ObjectivePredictionPreview? = nil
    var objectiveEvaluation: ObjectiveEvaluation? = nil
    var objectiveModelCode: String? = nil
    var dataLoadingIdentity: WorkflowDataLoadingIdentity? = nil
    var dataLoadingModelCode: String? = nil
    var bagContext: WorkflowBagArtifactContext? = nil
    var bagPredictionPreview: BagPredictionPreview? = nil
    var objectiveEvaluationContext: WorkflowObjectiveEvaluationContext? = nil
    var predictionTable:WorkflowPredictionTableEvidence? = nil
    var routingTables:[String:WorkflowPredictionTableEvidence]=[:]
}
/// Immutable verified proof. Keep the large plan/content on the heap rather
/// than copying them through nested History/frozen staging worker frames.
final class VerifiedWorkflowOutcome {
    var gradientArtifact: GradientCaptureArtifact? { modelContent?.gradientArtifact }
    var derivativeResults: DerivativeResults? { modelContent?.derivativeResults }
    let preparationCacheReport: DataFlowCacheReport?
    let failedGradientEvidence: WorkflowFailedGradientEvidence?
    let reference: WorkflowHistoryReference
    let outcome: WorkflowArchiveOutcome
    let capture: WorkflowArchiveCapture
    let plan: WorkflowExecutionPlan
    let modelContent: WorkflowVerifiedModelContent?
    init(preparationCacheReport:DataFlowCacheReport?=nil,failedGradientEvidence:WorkflowFailedGradientEvidence?=nil,reference:WorkflowHistoryReference,outcome:WorkflowArchiveOutcome,capture:WorkflowArchiveCapture,plan:WorkflowExecutionPlan,modelContent:WorkflowVerifiedModelContent?) {
        self.preparationCacheReport=preparationCacheReport;self.failedGradientEvidence=failedGradientEvidence
        self.reference=reference;self.outcome=outcome;self.capture=capture;self.plan=plan;self.modelContent=modelContent
    }
}
enum WorkflowHistoryLoad {
    case verified(VerifiedWorkflowOutcome)
    case legacyUnverifiable(WorkflowLegacyOutcome, reason: String)
    case incomplete(String)
    case unavailable(String)
}

/// Files are reopened beneath the configured private workspace; no stored absolute
/// path, live authoring owner, or installed runtime is needed for archive replay.
enum WorkflowHistoryReader {
    private enum Receipt { case current(WorkflowArchiveOutcome), legacy(WorkflowLegacyOutcome) }
    static func list(root: URL, projectID: String, limit: Int = 1_000) throws -> WorkflowHistoryListing {
        try require(UUID(uuidString: projectID) != nil && (1...10_000).contains(limit), "Invalid project or listing budget.")
        let scan = try scanDirectories(root: root, projectID: projectID, limit: limit)
        var entries: [WorkflowHistoryEntry] = [], issues = scan.issues
        for directory in scan.directories {
            let reference = WorkflowHistoryReference(projectID: projectID, directoryID: directory)
            do {
                guard let bytes = try DiagnosticFileIO.readIfPresent(root: root, parts: parts(reference) + ["outcome.json"], limit: 256_000) else {
                    entries.append(.init(reference: reference, status: nil, runID: nil, finishedAt: nil, verification: .incomplete, message: "Terminal outcome is missing; the archive may be interrupted.")); continue
                }
                switch try receipt(bytes, reference: reference) {
                case .current(let value):
                    entries.append(.init(reference: reference, status: value.status, runID: value.runID, finishedAt: value.finishedAt, verification: .recordedV2, message: value.error))
                case .legacy(let value):
                    entries.append(.init(reference: reference, status: value.status, runID: value.runID, finishedAt: nil, verification: .legacyV1, message: value.error ?? legacyReason))
                }
            } catch {
                entries.append(.init(reference: reference, status: nil, runID: nil, finishedAt: nil, verification: .unavailable, message: error.localizedDescription))
            }
        }
        if scan.truncated { issues.append("The directory scan reached its entry budget. This is not a complete newest-first history.") }
        entries.sort { left, right in
            if left.finishedAt != right.finishedAt { return (left.finishedAt ?? "") > (right.finishedAt ?? "") }
            return left.reference.directoryID.uuidString < right.reference.directoryID.uuidString
        }
        return .init(entries: entries, issues: issues, truncated: scan.truncated)
    }
    /// Revalidates the archive at access time; a previous listing or verified value
    /// never authorizes later filesystem bytes. No generated code is executed.
    static func loadModelContent(_ reference: WorkflowHistoryReference, root: URL) throws -> WorkflowVerifiedModelContent? {
        switch load(reference, root: root) {
        case .verified(let value): return value.modelContent
        case .legacyUnverifiable(_, let reason): throw WorkflowArchiveError(reason)
        case .incomplete(let reason), .unavailable(let reason): throw WorkflowArchiveError(reason)
        }
    }
    static func load(_ reference: WorkflowHistoryReference, root: URL) -> WorkflowHistoryLoad {
        var recordedError: String?
        do {
            try require(UUID(uuidString: reference.projectID) != nil, "Invalid project identity.")
            guard let bytes = try DiagnosticFileIO.readIfPresent(root: root, parts: parts(reference) + ["outcome.json"], limit: 256_000) else {
                return .incomplete("Terminal outcome is missing; no terminal state can be inferred.")
            }
            let outcome: WorkflowArchiveOutcome
            switch try receipt(bytes, reference: reference) {
            case .legacy(let value): return .legacyUnverifiable(value, reason: legacyReason)
            case .current(let value): outcome = value; recordedError = value.error
            }
            guard let captureFile = outcome.capture else {
                return .incomplete("Recorded \(outcome.status.rawValue) before a verifiable capture was committed." + errorSuffix(recordedError))
            }
            let capture = try WorkflowArchiveCodec.decodeCapture(read(captureFile, reference: reference, root: root))
            try require(sameUUID(capture.projectID, outcome.projectID) && sameUUID(capture.directoryID, outcome.directoryID) && sameUUID(capture.runID, outcome.runID) && capture.requestIdentity == outcome.requestIdentity, "Capture differs from outcome identity.")
            var parentContents:[String:Data]=[:]
            for parentFile in capture.checkpointFiles ?? [] {
                let name=try WorkflowArchiveCodec.pathParts(parentFile.path).last!
                parentContents[name] = try read(parentFile, reference: reference, root: root)
            }
            let parentRecord = capture.checkpointParent == nil ? nil : try WorkflowLifecycleParentRecord.decode(parentContents)
            let workflow = try WorkflowCodec.decode(read(capture.workflow, reference: reference, root: root))
            try require(sameUUID(workflow.projectID, outcome.projectID) && workflow.revision == capture.revision, "Workflow source identity differs from capture.")
            try require(workflow.schemaVersion == capture.schemaVersion, "Workflow source version differs from its capture schema.")
            let model = try JSONDecoder().decode(ComponentRegistry.self, from: read(capture.modelRegistry, reference: reference, root: root))
            let data = try JSONDecoder().decode(DataFlowOperatorRegistry.self, from: read(capture.dataRegistry, reference: reference, root: root))
            try require(WorkflowArchiveCodec.registryHash(model) == capture.modelRegistrySHA256 && WorkflowArchiveCodec.registryHash(data) == capture.dataRegistrySHA256, "Registry contract hashes differ from capture.")
            try runtimeEvidence(read(capture.runtimeEvidence, reference: reference, root: root))
            for source in capture.sources {
                _ = try WorkflowCapturedSourceArchive.open(source, root: root, prefix: "inputs/", baseParts: parts(reference))
            }
            let plan = try WorkflowCompiler.compileHistoricalEvidence(workflow, train: capture.train, modelRegistry: model, dataRegistry: data, contractVersion: max(2,capture.compilerContractVersion))
            try require(capture.plan.matches(plan, compilerContractVersion: capture.compilerContractVersion), "Archived execution plan does not reproduce from its captured source and registries.")
            try capture.plan.validateModelSnapshot(registry: model, context: plan.lossRecipe == nil ? .preparedWorkflow : .explicitObjective)
            let expectation = WorkflowPhaseExpectation(plan: plan, modelRegistrySHA256: capture.modelRegistrySHA256, dataRegistrySHA256: capture.dataRegistrySHA256, priorCacheManifestSHA256: capture.priorCache?.preparationManifestSHA256, compilerContractVersion: capture.compilerContractVersion, lifecycleParent: parentRecord?.evidence, expectedCPUThreads: capture.executionBudget?.resources?.cpuThreads, sampleCapture: capture.sampleCapture, preparationCacheLimitBytes: capture.preparationCacheLimitBytes, capturedModelRegistry: model, derivativeCapture: capture.derivativeCapture)
            let runRoot = parts(reference).reduce(root) { $0.appendingPathComponent($1) }
            var verified: [WorkflowArchiveFile] = []
            var modelContent: WorkflowVerifiedModelContent?
            var cacheReport: DataFlowCacheReport?
            var failedGradients: WorkflowFailedGradientEvidence?
            if let pin = outcome.preparationManifestSHA256 {
                let prepared = try WorkflowPhaseArtifacts.preparation(expectation: expectation, root: runRoot, storageRoot: root, storageParts: parts(reference))
                try require(prepared.selection.manifestSHA256 == pin, "Preparation manifest differs from outcome pin.")
                verified += prepared.artifactFiles
                cacheReport = prepared.cacheReport
                if let descriptor=outcome.failedGradientEvidence {
                    let bytes=try read(descriptor.file,reference:reference,root:root)
                    let rows=try WorkflowFailedGradientEvidence.trainingRows(root:root,parts:parts(reference),selection:prepared.selection)
                    let value=try WorkflowFailedGradientEvidence.decode(bytes,plan:plan,workflowRunID:outcome.runID,requestSHA256:capture.requestIdentity,preparationSHA256:pin,trainingRows:WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:rows,datasetRecipe:plan.datasetRecipe,loaderRecipe:plan.loaderRecipe,capturedBatchSize:plan.settings.batchSize,trainOrigin:plan.train,inlineRoles:plan.inlineDataLoadingRoles),initialEpoch:plan.trainingMode == .exactResume ? (parentRecord?.evidence.completedEpoch ?? 0):0)
                    try require(value.context.workerRunID==descriptor.workerRunID,"Failed-gradient descriptor worker differs.")
                    failedGradients=value;verified.append(descriptor.file)
                }
                if outcome.status == .completed {
                    let trained = try WorkflowPhaseArtifacts.training(expectation: expectation, root: runRoot, preparation: prepared.selection, storageRoot: root, storageParts: parts(reference))
                    try require(trained.manifestSHA256 == outcome.modelManifestSHA256 && sameUUID(trained.workerRunID, outcome.workerRunID ?? ""), "Training result differs from outcome identity.")
                    verified += trained.artifactFiles
                    modelContent = .init(reference: reference, outcome: outcome, capture: capture, code: trained.modelCode, metrics: trained.metrics, trainingControl:trained.trainingControl,partitionEvidence: trained.partitionEvidence, predictionPreview: trained.predictionPreview, activationPreviews: trained.activationPreviews, gradientArtifact: trained.gradientArtifact, performanceCapture: trained.performanceCapture, derivativeResults: trained.derivativeResults, objectiveContext: trained.objectiveContext, objectivePredictionPreview: trained.objectivePredictionPreview, objectiveEvaluation: trained.objectiveEvaluation, objectiveModelCode: trained.objectiveModelCode, dataLoadingIdentity:trained.dataLoadingIdentity,dataLoadingModelCode:trained.dataLoadingModelCode, bagContext: trained.bagContext, bagPredictionPreview: trained.bagPredictionPreview, objectiveEvaluationContext: trained.objectiveEvaluationContext,predictionTable:trained.predictionTable,routingTables:trained.routingTables)
                }
            }
            if let file=outcome.artifacts.first(where:{$0.path==TrainingProgressArchive.path}) {
                guard capture.sampleCapture?.savesDiagnosticDetails != false,let worker=outcome.workerRunID,let content=modelContent else {throw WorkflowArchiveError("Progress retained despite disabled capture or absent model.")}
                let archive=try TrainingProgressArchive.decode(read(file,reference:reference,root:root))
                try archive.validate(request:capture.requestIdentity,worker:worker,metrics:content.metrics)
                modelContent?.trainingProgress=archive;verified.append(file)
            }
            try require(verified.sorted { $0.path < $1.path } == outcome.artifacts.sorted { $0.path < $1.path }, "Verified phase inventory differs from outcome inventory.")
            return .verified(.init(preparationCacheReport: cacheReport, failedGradientEvidence:failedGradients, reference: reference, outcome: outcome, capture: capture, plan: plan, modelContent: modelContent))
        } catch { return .unavailable(error.localizedDescription + errorSuffix(recordedError)) }
    }
    /// Re-read pinned bytes through anchored I/O, then give the worker an owned
    /// snapshot. The worker never follows a mutable archive ancestor or source path.
    static func stagePreparationInspection(_ reference: WorkflowHistoryReference, root: URL, accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil, workload: ExecutionWorkloadPolicy? = nil,
                                           checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> WorkflowPreparationInspectionInput {
        try checkCancellation()
        let value: VerifiedWorkflowOutcome
        switch load(reference, root: root) {
        case .verified(let verified): value = verified
        case .legacyUnverifiable(_, let reason), .incomplete(let reason), .unavailable(let reason):
            throw WorkflowArchiveError(reason)
        }
        try checkCancellation()
        guard let pin = value.outcome.preparationManifestSHA256, let report = value.preparationCacheReport else {
            throw WorkflowArchiveError("This archive has no supported completed preparation to inspect.")
        }
        let workflow = try WorkflowCodec.decode(read(value.capture.workflow, reference: reference, root: root))
        let registry = try JSONDecoder().decode(DataFlowOperatorRegistry.self, from: read(value.capture.dataRegistry, reference: reference, root: root))
        let nodes = try value.plan.dataflow.nodes.map { node -> WorkflowPreparationInspectionInput.Node in
            let address = WorkflowNodeAddress(domain: .data, localID: node.id)
            guard let source = workflow.nodes.first(where: { $0.address == address }),
                  let operation = registry.operatorDefinition(id: node.operatorID)?.configuredForExecution(node.config,
                    connectedInputs: Set(value.plan.dataflow.edges.filter { $0.to.node == node.id }.map { $0.to.port })) else {
                throw WorkflowArchiveError("Captured preparation component identity is unavailable.")
            }
            return .init(address: address, name: source.name, operatorID: node.operatorID, operatorTitle: operation.title, portKinds: Dictionary(uniqueKeysWithValues: operation.outputs.map { ($0.id, $0.kind) }))
        }
        let files = value.outcome.artifacts.filter { $0.path.hasPrefix("preparation/artifacts/") }
        guard !files.isEmpty, files.count <= WorkflowPreviewCapture.Purpose.preparationInspection.captureCount(workload) else {
            throw WorkflowArchiveError("Preparation snapshot exceeds its inventory allowance.")
        }
        var total = 0, paths = Set<String>()
        for file in files {
            try checkCancellation()
            let parts = try WorkflowArchiveCodec.pathParts(file.path)
            try require(parts.count == 3, "Preparation snapshot contains an unsupported artifact path.")
            let relative = "input/preparation/" + parts[2]
            let limit = try WorkflowPreviewCapture.Purpose.preparationInspection.fileLimit(relative, workload: workload)
            guard paths.insert(relative).inserted, (0...limit).contains(file.bytes),
                  file.bytes <= WorkflowPreviewCapture.Purpose.preparationInspection.captureBytes(workload) - total else {
                throw WorkflowArchiveError("Preparation snapshot exceeds its capture allowance.")
            }
            total += file.bytes
        }
        let snapshotID = UUID()
        let reservation: WorkflowCaptureStorageReservation?
        if let storageLimitBytes {
            reservation = try WorkflowCaptureStorageReservation(workspace: root, accountingRoot: accountingRoot, projectID: reference.projectID,
                purpose: .preparationInspection, maximumCaptureBytes: total,
                additionalScopes: [try WorkflowStorageAdmission.Scope(namespace: .preparationInspectionStaging, components: [snapshotID.uuidString])],
                additionalBytes: Int64(total), limitBytes: storageLimitBytes, workload: workload)
        } else { reservation = nil }
        var retained = false, capture: WorkflowPreviewCapture?
        defer { if !retained { capture?.remove(); reservation?.close() } }
        // Stream pinned archive artifacts through descriptor-relative reads; only the small manifest is decoded.
        var inputs: [WorkflowPreviewCapture.Input] = []
        var graphDigest: String?, registryDigest: String?, numpyVersion: String?
        for file in files {
            try checkCancellation()
            let parts = try WorkflowArchiveCodec.pathParts(file.path)
            try require(parts.count == 3, "Preparation snapshot contains an unsupported artifact path.")
            if parts.last == "manifest.json" {
                let bytes = try read(file, reference: reference, root: root)
                try require(WorkflowArchiveCodec.hash(bytes) == pin, "Preparation snapshot manifest changed.")
                let manifest = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
                if case .string(let hash) = manifest["graphDigest"] { graphDigest = hash }
                if case .string(let hash) = manifest["registryDigest"] { registryDigest = hash }
                if case .string(let version) = manifest["numpyVersion"] { numpyVersion = version }
            }
            try checkCancellation()
            inputs.append(.anchoredFile(path: "input/preparation/" + parts[2], root: root,
                parts: Self.parts(reference) + parts, expectedSHA256: file.sha256))
        }
        guard let graphDigest, let registryDigest, let numpyVersion, !numpyVersion.isEmpty, numpyVersion.utf8.count <= 128, WorkflowArchiveCodec.isHash(graphDigest), WorkflowArchiveCodec.isHash(registryDigest) else {
            throw WorkflowArchiveError("Preparation snapshot is missing its manifest identities.")
        }
        try checkCancellation()
        let snapshot = try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs,
            purpose: .preparationInspection, workload: workload, stageID: snapshotID, maximumBytes: total, checkCancellation: checkCancellation)
        capture = snapshot
        try checkCancellation()
        let ownership = WorkflowPreparationInspectionInput.CaptureOwnership(capture: snapshot, reservation: reservation)
        retained = true
        return .init(reference: reference, runID: value.outcome.runID, requestIdentity: value.outcome.requestIdentity,
                     terminalStatus: value.outcome.status, capturedRevision: value.capture.revision,
                     directory: snapshot.directory.appendingPathComponent("input/preparation"), manifestSHA256: pin,
                     graphDigest: graphDigest, registryDigest: registryDigest, numpyVersion: numpyVersion,
                     graph: value.plan.dataflow, nodes: nodes, cacheReport: report, captureOwnership: ownership)
    }
    /// Internal continuation input, not a share/export package. The caller owns cleanup.
    static func stageCheckpoint(_ reference: WorkflowHistoryReference, root: URL, accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil) throws -> WorkflowCheckpointSnapshot {
        try stageCheckpoint(reference,root:root,expected:nil,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
    }

    /// Stage the exact parent selected on Train, not merely whatever completed run
    /// now occupies the managed directory. Validate pins before writing private files.
    static func stageCheckpoint(_ selected: WorkflowCheckpointReference, root: URL, accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil) throws -> WorkflowCheckpointSnapshot {
        guard UUID(uuidString:selected.projectID) != nil,
              let directoryID=UUID(uuidString:selected.directoryID),selected.hasValidPins else {
            throw WorkflowArchiveError("Checkpoint selection requires valid project/run identities and lowercase request/manifest pins.")
        }
        return try stageCheckpoint(.init(projectID:selected.projectID,directoryID:directoryID),root:root,expected:selected,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
    }

    private static func stageCheckpoint(_ reference: WorkflowHistoryReference, root: URL,
                                        expected:WorkflowCheckpointReference?, accountingRoot: URL? = nil, storageLimitBytes: Int64?) throws -> WorkflowCheckpointSnapshot {
        try Task.checkCancellation()
        guard case .verified(let value) = load(reference, root: root),
              value.outcome.status == .completed, value.modelContent != nil,
              let manifestPin = value.outcome.modelManifestSHA256 else {
            throw WorkflowArchiveError("Select a completed run with verified model artifacts before continuing training.")
        }
        if let expected {
            guard value.outcome.runID == expected.runID,
                  value.outcome.requestIdentity == expected.requestIdentity,
                  manifestPin == expected.modelManifestSHA256 else {
                throw WorkflowArchiveError("Selected checkpoint parent no longer matches its run, request or model-manifest identity. Select the parent again.")
            }
        }
        var contents: [String: Data] = [:]
        for (name, maximum) in WorkflowCheckpointSnapshot.artifactLimits.sorted(by:{$0.key<$1.key}) {
            guard let file = value.outcome.artifacts.first(where: { $0.path == "training/artifacts/" + name }),
                  file.bytes > 0, file.bytes <= maximum else {
                throw WorkflowArchiveError("Continuation requires a bounded saved artifact: " + name)
            }
            try Task.checkCancellation()
            contents[name] = try read(file, reference: reference, root: root)
        }
        guard let manifest = contents["manifest.json"], WorkflowArchiveCodec.hash(manifest) == manifestPin else {
            throw WorkflowArchiveError("The checkpoint manifest changed after verification.")
        }
        if try WorkflowCheckpointInventory.limits(manifest:manifest)[WorkflowCheckpointInventory.traceName] != nil {
            let name=WorkflowCheckpointInventory.traceName
            guard let file=value.outcome.artifacts.first(where:{$0.path == "training/artifacts/"+name}),file.bytes>0,file.bytes<=WorkflowCheckpointInventory.traceLimit else {throw WorkflowArchiveError("Scheduled parent trace is missing or exceeds its bound.")}
            contents[name]=try read(file,reference:reference,root:root)
        }
        try WorkflowCheckpointInventory.verify(contents)
        var peak: Int64 = 0
        for data in contents.values {
            let (sum, overflow) = peak.addingReportingOverflow(Int64(data.count))
            guard !overflow else { throw WorkflowStorageAdmission.Failure.invalidScope }
            peak = sum
        }
        try Task.checkCancellation()
        let stageID = UUID()
        let reservation = try storageLimitBytes.map { limit in
            try WorkflowStorageAdmission(workspace: root, accountingRoot: accountingRoot).reserve(scopes: [
                .init(namespace: .checkpointStaging, components: [stageID.uuidString])
            ], peakBytes: peak, limitBytes: limit)
        }
        var lifetime: WorkflowExportStorageLifetime?, retained = false
        defer {
            if !retained {
                if let lifetime { lifetime.remove() }
                else { reservation?.close() }
            }
        }
        guard let canonical = realpath(root.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let canonicalRoot = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        free(canonical)
        let rootFD = try WorkflowEvaluationOwnership.openDirectory(canonicalRoot)
        defer { Darwin.close(rootFD) }
        let rootIdentity = try WorkflowEvaluationOwnership.directoryIdentity(rootFD)
        if mkdirat(rootFD, "checkpoint-staging", 0o700) != 0, errno != EEXIST { throw WorkflowFileLeaseError.system(errno) }
        let parentFD = openat(rootFD, "checkpoint-staging", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parentFD >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(parentFD) }
        let parentIdentity = try WorkflowEvaluationOwnership.directoryIdentity(parentFD)
        let parentURL = canonicalRoot.appendingPathComponent("checkpoint-staging")
        func checkDirectory(_ url: URL, identity: WorkflowFileIdentity) throws {
            let current = try WorkflowEvaluationOwnership.openDirectory(url)
            defer { Darwin.close(current) }
            guard try WorkflowEvaluationOwnership.directoryIdentity(current) == identity else { throw WorkflowFileLeaseError.unsafePath }
        }
        try checkDirectory(canonicalRoot, identity: rootIdentity)
        try checkDirectory(parentURL, identity: parentIdentity)
        guard mkdirat(parentFD, stageID.uuidString, 0o700) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        let directory = parentURL.appendingPathComponent(stageID.uuidString)
        let directoryFD = openat(parentFD, stageID.uuidString, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directoryFD >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(directoryFD) }
        let directoryIdentity = try WorkflowEvaluationOwnership.directoryIdentity(directoryFD)
        try checkDirectory(directory, identity: directoryIdentity)
        lifetime = try WorkflowExportStorageLifetime(directory: directory, reservation: reservation)
        for (name, bytes) in contents {
            try Task.checkCancellation()
            try checkDirectory(canonicalRoot, identity: rootIdentity)
            try checkDirectory(parentURL, identity: parentIdentity)
            try checkDirectory(directory, identity: directoryIdentity)
            let file = openat(directoryFD, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard file >= 0 else { throw WorkflowFileLeaseError.system(errno) }
            defer { Darwin.close(file) }
            try bytes.withUnsafeBytes { raw in
                var offset = 0
                while offset < raw.count {
                    try Task.checkCancellation()
                    let count = Darwin.write(file, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                    if count < 0, errno == EINTR { continue }
                    guard count > 0 else { throw WorkflowFileLeaseError.system(errno) }
                    offset += count
                }
            }
            guard fsync(file) == 0 else { throw WorkflowFileLeaseError.system(errno) }
            try checkDirectory(directory, identity: directoryIdentity)
        }
        retained = true
        return .init(reference: reference, runID: value.outcome.runID, requestIdentity: value.outcome.requestIdentity,
                     manifestSHA256: manifestPin, directory: directory,
                     files: contents.keys.sorted().map { .init(path:$0,bytes:contents[$0]!.count,sha256:WorkflowArchiveCodec.hash(contents[$0]!)) },
                     storageLifetime: lifetime, dataLoadingIdentity:value.modelContent?.dataLoadingIdentity)
    }

    /// Default portable package excludes private source rows, source bindings,
    /// preparation tables and the training manifest's row-level provenance.
    static func portablePolicyHelpers(gradientSelected:Bool,trainabilitySelected:Bool,artifacts:[String:Data])throws->[String:Data] {
        var required=Set<String>()
        if gradientSelected {required.insert("gradient_policy.py")}
        if trainabilitySelected {required.insert("trainability_recipe.py")}
        guard Set(artifacts.keys) == required else {throw WorkflowArchiveError("Portable policy helpers differ from the captured selected policies.")}
        for (name,bytes) in artifacts {
            guard !bytes.isEmpty,bytes.count<=131072,String(data:bytes,encoding:.utf8) != nil else {throw WorkflowArchiveError("The saved policy helper is empty, oversized or invalid UTF-8: \(name).")}
        }
        return artifacts
    }
    static func stageTrainedExport(_ reference: WorkflowHistoryReference, root: URL, accountingRoot: URL? = nil, storageLimitBytes: Int64? = nil) throws -> WorkflowTrainedExportPackage {
        guard case .verified(let value) = load(reference, root: root) else {
            throw WorkflowArchiveError("Select a completed run with verified model artifacts before exporting.")
        }
        return try stageVerifiedTrainedExport(value,reference:reference,root:root,accountingRoot:accountingRoot,storageLimitBytes:storageLimitBytes)
    }
    /// Verify History before entering the larger package-construction frame.
    @inline(never)
    private static func stageVerifiedTrainedExport(_ value:VerifiedWorkflowOutcome,reference:WorkflowHistoryReference,root:URL,accountingRoot:URL?,storageLimitBytes:Int64?) throws -> WorkflowTrainedExportPackage {
        guard value.outcome.status == .completed, let content = value.modelContent,
              let modelPin = value.outcome.modelManifestSHA256 else {
            throw WorkflowArchiveError("Select a completed run with verified model artifacts before exporting.")
        }
        func artifact(_ name: String) throws -> Data {
            guard let file = value.outcome.artifacts.first(where: { $0.path == "training/artifacts/" + name }) else {
                throw WorkflowArchiveError("A required trained artifact is missing: \(name).")
            }
            return try read(file, reference: reference, root: root)
        }
        let model = try artifact("model.py"), checkpoint = try artifact("checkpoint.pt")
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: artifact("manifest.json"))
        guard case .object(let runtime) = manifest["runtime"], case .string(let torch) = runtime["torch"],
              torch.range(of: "^[0-9]+\\.[0-9]+\\.[0-9]+([a-zA-Z0-9.+-]*)$", options: .regularExpression) != nil,
              torch.utf8.count <= 128,
              let helperURL = Bundle.module.url(forResource: "inference", withExtension: "py") else {
            throw WorkflowArchiveError("Export runtime metadata or inference template is unavailable.")
        }
        let capturedRegistry=try JSONDecoder().decode(ComponentRegistry.self,from:read(value.capture.modelRegistry,reference:reference,root:root))
        let inputContract:TrainedModelInputContract
        let contract:[String:JSONValue]
        if let plan=value.plan.namedInputPlan {
            inputContract=try TrainedModelInputContract(graph:value.plan.model.graph,registry:capturedRegistry,namedInputPlan:plan,recipe:value.plan.lossRecipe)
            let output=try WorkflowMappingRequirements.outputContract(value.plan.model.graph,registry:capturedRegistry)
            var named:[String:JSONValue]=["schemaVersion":4,"input":.object(inputContract.input),"output":.object(["kind":.string(value.plan.lossRecipe == nil ? "logits":"modelOutput"),"dtype":.string("float32"),"shape":.array([.string("N")]+output.layout.sample.dimensions.map {.integer(Int64($0))})])]
            if let recipe=value.plan.lossRecipe {named["objective"] = .object(["recipe":recipe.configuration.recipeValue,"source":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(recipe.source)),"targetMapping":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(recipe.targetMapping)),"scalarization":.string(value.plan.objectiveScalarization.rawValue)])}
            contract=named
        } else if let mapping=value.plan.bagInput {
            guard let context=content.bagContext,context.mapping == mapping,
                  context.runID == value.outcome.workerRunID else {throw WorkflowArchiveError("The saved packet context is unavailable or differs from the captured plan.")}
            inputContract=try TrainedModelInputContract(graph:value.plan.model.graph,registry:capturedRegistry,bagMapping:mapping,targetColumn:value.plan.targetColumn,recipe:value.plan.lossRecipe)
            let requirement=try WorkflowMappingRequirements.bagInputRequirement(value.plan.model.graph,registry:capturedRegistry,context:value.plan.lossRecipe == nil ? .preparedWorkflow:.explicitObjective)
            guard requirement == context.requirement else {throw WorkflowArchiveError("The saved packet source differs from its captured graph.")}
            var packetContract:[String:JSONValue]=["schemaVersion":3,"input":.object(inputContract.input)]
            if let recipe=value.plan.lossRecipe {
                guard case .objective(_,let selected)=context.target,selected == recipe,
                      let evaluation=content.objectiveEvaluationContext,evaluation.recipe == recipe,
                      evaluation.scalarization == value.plan.objectiveScalarization,
                      evaluation.outputSampleShape == context.outputContract.layout.sample.dimensions,
                      content.objectiveModelCode != nil else {throw WorkflowArchiveError("The saved packet objective context differs from the captured plan.")}
                let source=try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(recipe.source))
                let targetMapping=try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(recipe.targetMapping))
                packetContract["output"] = .object(["kind":.string("modelOutput"),"dtype":.string("float32"),"shape":.array([.string("N")]+context.outputContract.layout.sample.dimensions.map {.integer(Int64($0))})])
                packetContract["objective"] = .object(["recipe":recipe.configuration.recipeValue,"source":source,"targetMapping":targetMapping,"scalarization":.string(value.plan.objectiveScalarization.rawValue)])
            } else {
                guard case .binary(let target)=context.target,target == value.plan.targetColumn else {throw WorkflowArchiveError("The saved packet binary target differs from the captured plan.")}
                packetContract["output"] = .object(["kind":.string("logits"),"shape":.array([.string("N"),2]),"classIndices":.array([0,1])])
            }
            contract=packetContract
        } else if let recipe=value.plan.lossRecipe {
            guard let context=content.objectiveContext,context.recipe == recipe,
                  context.scalarization == value.plan.objectiveScalarization,
                  content.objectiveModelCode != nil else {throw WorkflowArchiveError("The saved objective context is unavailable or differs from the captured plan.")}
            inputContract=try TrainedModelInputContract(graph:value.plan.model.graph,registry:capturedRegistry,recipe:recipe)
            let output=try WorkflowMappingRequirements.outputContract(value.plan.model.graph,registry:capturedRegistry)
            try WorkflowMappingRequirements.validateObjective(prediction:output,recipe:recipe)
            guard output.layout.sample.dimensions == context.outputSampleShape else {throw WorkflowArchiveError("The saved prediction shape differs from its captured model.")}
            let source=try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(recipe.source))
            let mapping=try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(recipe.targetMapping))
            contract=["schemaVersion":.integer(2),"input":.object(inputContract.input),
                "output":.object(["kind":.string("modelOutput"),"dtype":.string("float32"),
                    "shape":.array([.string("N")]+output.layout.sample.dimensions.map {.integer(Int64($0))})]),
                "objective":.object(["recipe":recipe.configuration.recipeValue,"source":source,"targetMapping":mapping,
                    "scalarization":.string(value.plan.objectiveScalarization.rawValue)])]
        } else {
            guard let target=value.plan.targetColumn else {throw WorkflowArchiveError("The saved legacy target mapping is unavailable.")}
            inputContract=try TrainedModelInputContract(graph:value.plan.model.graph,registry:capturedRegistry,columns:value.plan.featureColumns,target:target)
            contract=["schemaVersion":.integer(1),"input":.object(inputContract.input),
                "output":.object(["kind":.string("logits"),"shape":.array([.string("N"),.integer(2)]),
                    "classIndices":.array([.integer(0),.integer(1)])])]
        }
        var readme = """
        # KwipML trained model export

        This package contains the captured model definition and learned checkpoint from run \(value.outcome.runID).
        It is independent of the KwipML app and the original run directory.

        Install the recorded PyTorch version from requirements.txt in your Python environment, then run:

            python inference.py --example
            python inference.py --input prepared.json --batch-size 256

        \(inputContract.namedInputPlan != nil ? "The example uses named synthetic tensor records. JSON input is a list of objects containing the captured source columns. The Python API accepts a dictionary of aligned CPU tensors; prepare_records applies the explicit mapping. Captured tokenizer metadata is included when available; other preprocessing must be applied before inference. No raw source observations are included." : inputContract.isBag ? "The example uses synthetic whole-group records, not training observations. JSON input is an array of records with indices and, only when configured, per_sample_weights. The Python API is `from inference import predict`; pass the same whole-group records. The helper constructs the captured packet, including packed offsets, and never loads source rows or fits preprocessing. "+inputContract.valueGuidance : "The example uses synthetic prepared rows, not training observations. The JSON input is an array of\nprepared observations with the nested sample shape in input-contract.json. Fill each sample with columns\nin the listed order, with the last dimension changing fastest. Raw data preparation is not included.\nThe Python API is `from inference import predict`; pass a CPU "+inputContract.tensor!.dtype.rawValue+" tensor\nshaped "+inputContract.pythonShape+", N from 1 to 100000. "+inputContract.valueGuidance)
        \(value.plan.lossRecipe == nil ? "Results are two logits per row; argmax gives class index 0 or 1. Class indices do not name real-world labels.":"Results are the captured modelOutput array per observation, with the exact sample shape in input-contract.json. Inference does not compute a training loss or invent class labels.")
        Prediction batches default to 1024 rows; set batch_size (Python) or --batch-size (CLI) from 1 to 100000.
        Smaller batches reduce intermediate activation memory; larger batches may improve throughput. All results
        remain in input order. Different batch sizes may have small floating-point rounding differences.

        For repeated predictions, verify and load the package once:

            from inference import load_model
            predictor = load_model()
            \(value.plan.lossRecipe == nil ? "logits":"model_output") = predictor.predict(\(inputContract.isBag || inputContract.namedInputPlan != nil ? "records":"features"), batch_size=256)

        A loaded predictor retains its weights and does not reread package files on every call. Call load_model()
        again to verify and adopt changed files. Loading preserves the caller's CPU random-generator state;
        model construction still uses PyTorch's global generator temporarily, so coordinate loading with any
        other threads consuming that generator. The one-shot predict function verifies and loads for each call.

        Apply the same preparation used during training before calling this model. Named text packages may
        include the captured tokenizer for encode_text; they never fit a new vocabulary during inference.
        Other raw preparation and raw source observations are not included. Re-fitting new statistics is
        not equivalent to applying the original learned transform.

        checkpoint.pt contains modelState plus optimizer/RNG state and provenance hashes. It is not a stripped
        weights-only file. The helper uses PyTorch's restricted weights_only loader, strict state loading and
        eval/inference mode. Use checkpoint['modelState'] with Model.load_state_dict for your own integration.
        It does not resume training. Exact continuation requires the matching training contract and data.

        model.py includes the recorded user component names. The checkpoint includes model topology identifiers,
        runtime and training metadata. export-manifest.json inventories exact package bytes. It verifies package
        consistency, not authorship. No training rows, source bindings, absolute source paths, or row membership
        lists are added by this exporter. Numerical parity across different hardware/runtime versions is not promised.
        """
        if let recipe=value.plan.lossRecipe {
            let roles=recipe.targetMapping.roles.map { role in
                "- \(role.role): \(role.dtype.rawValue), sample shape \(role.sampleShape.isEmpty ? "scalar (one value per observation)":String(describing:role.sampleShape)), columns \(role.columns.joined(separator:", "))."
            }.joined(separator:"\n")
            readme += """


            ## Training objective (separate from prediction)

            input-contract.json retains the canonical selected objective, its qualified canvas source,
            ordered target-role mapping and scalarization. objective_recipe.py is a standalone helper for
            a fresh training objective. Inference only verifies its bytes; it never imports or executes it.
            Call create_training_objective(device='cpu') or device='mps' after choosing your training device,
            then evaluate_training_objective(objective, model_output, roles) with your own prepared targets.
            This creates fresh configured fixed weights/buffers; it does not restore objectiveState, optimizer,
            random state, source data or fitted preprocessing. The exact saved checkpoint includes objectiveState.

            The helper returns raw (the configured native reduction) and scalar (the differentiable training
            score). Reduction 'none' keeps individual raw scores; the captured mean/sum scalarization is applied
            separately. Neither raw nor scalar replaces modelOutput. Scalarization: \(value.plan.objectiveScalarization.rawValue).

            Required target roles, with columns in last-direction-fastest order:
            \(roles.isEmpty ? "Triplet loss has no dummy target. Its three input groups are anchor, positive, negative, in the order recorded in targetMapping.tripletSourceGroups.":roles)

            Exact CPU continuation requires the full matching training contract and data. MPS checkpoint use
            is limited to loading weights for fresh fine-tuning. This prediction package does not resume training.
            """
        }
        var bytes: [String: Data] = ["model.py": model, "checkpoint.pt": checkpoint,
            "inference.py": try Data(contentsOf: helperURL), "input-contract.json": try WorkflowArchiveCodec.encode(contract),
            "README.md": Data(readme.utf8), "requirements.txt": Data("torch==\(torch)\n".utf8)]
        if value.plan.lossRecipe != nil {
            let objective=try artifact("objective_recipe.py")
            guard !objective.isEmpty,objective.count<=131072,
                  let code=String(data:objective,encoding:.utf8),code == content.objectiveModelCode else {
                throw WorkflowArchiveError("The saved objective helper is invalid or differs from verified History.")
            }
            bytes["objective_recipe.py"]=objective
        }
        if value.plan.optimizerRecipe != nil {
            let recipe=try artifact("training_recipe.py")
            guard recipe.count<=8192,String(data:recipe,encoding:.utf8) != nil else {throw WorkflowArchiveError("The saved optimizer training recipe is invalid.")}
            bytes["training_recipe.py"]=recipe
            bytes["README.md"] = Data((readme + """


            This package also includes training_recipe.py with the captured optimizer configuration.
            For a fresh training run, construct and move the model to CPU or MPS first, then call
            training_recipe.create_optimizer(model). The helper selects trainable dense float32 settings.
            It creates fresh optimizer state; it does not restore checkpoint state, data, or random state.
            The inference helper verifies this optional file's hash without executing it.
            """).utf8)
        }
        func preparationArtifact(_ name:String)throws->Data {
            guard let file=value.outcome.artifacts.first(where:{$0.path=="preparation/artifacts/"+name}) else {throw WorkflowArchiveError("A required captured preparation artifact is missing: \(name).")}
            return try read(file,reference:reference,root:root)
        }
        if value.plan.namedInputPlan != nil {
            bytes["named_inputs.py"]=try preparationArtifact("named_inputs.py")
            if let plan=value.plan.namedInputPlan,plan.inputs.contains(where:{$0.representation=="tokens"}) {
                let raw=try preparationArtifact("manifest.json")
                guard case .object(let inventory)=try JSONDecoder().decode(JSONValue.self,from:raw),case .array(let entries)=inventory["artifacts"] else {throw WorkflowArchiveError("Captured tokenizer inventory is absent.")}
                var states:[[String:JSONValue]]=[]
                for entry in entries {
                    guard case .object(let item)=entry,item["kind"] == .string("state"),let path=item["path"]?.stringValue else {continue}
                    guard case .object(let state)=try JSONDecoder().decode(JSONValue.self,from:preparationArtifact(path)),state["operator"] == .string("kwip.data.textEncode") else {continue}
                    states.append(state)
                }
                var tokenizers:[String:JSONValue]=[:]
                for input in plan.inputs where input.representation=="tokens" {
                    var sourceColumn=input.columns[0]
                    if plan.sequenceObjective?.inputName==input.name,let node=value.plan.dataflow.nodes.first(where:{$0.id==value.plan.training.node}),node.operatorID==SequenceWindowsConfiguration.operatorID {sourceColumn=node.config["tokenColumn"]?.stringValue ?? sourceColumn}
                    let matches=states.filter {state in guard case .object(let config)=state["configuration"] else {return false};return config["outputColumn"] == .string(sourceColumn)}
                    guard matches.count<=1 else {throw WorkflowArchiveError("Multiple captured tokenizers use the same output column. Give each token output a distinct name before exporting.")}
                    if let state=matches.first {
                        guard case .array(let vocabulary)=state["vocabulary"],vocabulary.count==input.vocabularySize,let config=state["configuration"],let unicode=state["unicodeVersion"],let digest=state["contentDigest"] else {throw WorkflowArchiveError("Captured vocabulary differs from the model mapping.")}
                        tokenizers[input.name] = .object(["schemaVersion":1,"configuration":config,"vocabulary":.array(vocabulary),"unicodeVersion":unicode,"sourceStateSHA256":digest])
                    }
                }
                if !tokenizers.isEmpty {bytes["tokenizers.json"]=try WorkflowArchiveCodec.encode(JSONValue.object(tokenizers));bytes["portable_text_encoding.py"]=try preparationArtifact("portable_text_encoding.py")}
            }
            let existingReadme=String(decoding:bytes["README.md"]!,as:UTF8.self)
            bytes["README.md"]=Data((existingReadme+"\n\nNamed-input API: predictor.predict accepts a dictionary of aligned CPU tensors by model input name. predictor.prepare_records accepts a list of dictionaries containing the captured numeric/token columns. For captured Text encoding inputs, predictor.encode_text(texts,input_name) returns IDs using the saved vocabulary and policies without fitting; predictor.prepare_records applies the saved tensor length policy.\n").utf8)
        }
        if value.plan.schedulerRecipe != nil {
            let trace = try artifact("rate-trace.json")
            guard !trace.isEmpty, trace.count <= 4_000_000,
                  String(data: trace, encoding: .utf8) != nil else {
                throw WorkflowArchiveError("The saved scheduler rate trace is invalid.")
            }
            // Cold History has independently verified the native curve and exact saved inventory.
            // Reopen the pinned bytes here instead of copying a live console or cached preview.
            bytes["rate-trace.json"] = trace
            let existingReadme = String(decoding: bytes["README.md"]!, as: UTF8.self)
            bytes["README.md"] = Data((existingReadme + """


            ## Adjustment schedule

            rate-trace.json records the actual scheduler settings and learning rates from this run.
            Constructor initialization is listed separately from adjustment ticks. Each record distinguishes
            the rate used for an update from the rate selected for the next update. These are training
            settings, not model quality scores. An exhausted OneCycle schedule can have an unused final
            next rate; do not use it for an additional update or continue beyond the planned batch count.
            export-manifest.json retains the captured scheduler settings and qualified canvas source.

            For fresh training, move the model to CPU or MPS before creating these objects:

                optimizer = training_recipe.create_optimizer(model)
                scheduler = training_recipe.create_scheduler(optimizer)
                training_recipe.training_step(loss, optimizer, scheduler)

            Compute loss from your own prepared training batch. training_step checks the pinned horizon,
            updates the optimizer and advances OneCycle once per actual optimizer batch. For StepLR,
            CosineAnnealingLR, LinearLR and ReduceLROnPlateau, finish validation for the entire epoch,
            then call training_recipe.completed_epoch_step(scheduler, validation_loss=loss_value,
            validation_accuracy=accuracy_value). Plateau uses its captured validation metric; the other
            epoch schedules do not require those metric arguments. Do not also call scheduler.step().
            Constructors may adjust the initial rate before the first update; this is not an extra tick.

            These helpers create fresh optimizer and scheduler state. They do not restore this run's
            trace, checkpoint state, random state, data or fitted preprocessing. checkpoint.pt includes
            scheduler state in addition to model, optimizer and RNG state. Exact CPU continuation needs
            the matching training contract and data; MPS continuation is limited to weights for fresh
            fine-tuning. Inference does not execute the training helper or advance this schedule.
            """).utf8)
        }
        if value.plan.initializationRecipe != nil {
            let helper=try artifact("initialization_recipe.py")
            guard !helper.isEmpty,helper.count<=131072,String(data:helper,encoding:.utf8) != nil else {
                throw WorkflowArchiveError("The saved starting-settings helper is missing, oversized or invalid UTF-8.")
            }
            bytes["initialization_recipe.py"]=helper
            let currentReadme=String(decoding:bytes["README.md"]!,as:UTF8.self)
            bytes["README.md"]=Data((currentReadme+"""


            Starting settings
            This package includes initialization_recipe.py with the captured policy for a fresh model.
            Inference loads the learned weights and never imports or executes this helper. To start a
            separate training job, create and materialize a fresh model first, then explicitly call
            initialization_recipe.apply_initialization(model, initialization_recipe.INITIALIZATION_RECIPE)
            before constructing its optimizer. The exported model exposes its exact layer_keys mapping.
            Never apply the fresh policy after loading learned weights or while resuming a checkpoint.
            The seed is reproducible within the verified runtime, not a guarantee across Torch versions.
            """).utf8)
        }
        if value.plan.datasetRecipe != nil || value.plan.loaderRecipe != nil {
            let helper=try artifact("data_loading_recipe.py")
            guard !helper.isEmpty,helper.count<=131072,
                  let source=String(data:helper,encoding:.utf8),source == content.dataLoadingModelCode,
                  content.dataLoadingIdentity != nil else {
                throw WorkflowArchiveError("The saved reader helper differs from verified History.")
            }
            bytes["data_loading_recipe.py"]=helper
            let existing=String(decoding:bytes["README.md"]!,as:UTF8.self)
            bytes["README.md"]=Data((existing+"""


            Data loading
            data_loading_recipe.py retains the selected training reader and batching policy.
            It is for fresh training with your own prepared observations. Prediction verifies its
            bytes and never executes it. The training partition, reader order and final-group
            policy remain part of the full checkpoint contract for exact CPU continuation.
            """).utf8)
        }
        let policyNames=["gradient_policy.py","trainability_recipe.py"]
        var policyArtifacts:[String:Data]=[:]
        for name in policyNames where value.outcome.artifacts.contains(where:{$0.path == "training/artifacts/"+name}) {
            policyArtifacts[name]=try artifact(name)
        }
        let policyHelpers=try portablePolicyHelpers(gradientSelected:value.plan.gradientPolicyRecipe != nil,trainabilitySelected:value.plan.trainabilityRecipe != nil,artifacts:policyArtifacts)
        for (name,helper) in policyHelpers {bytes[name]=helper}
        if !policyHelpers.isEmpty {
            let currentReadme=String(decoding:bytes["README.md"]!,as:UTF8.self)
            bytes["README.md"]=Data((currentReadme+"""


            Gradient and trainability policies
            The optional gradient_policy.py and trainability_recipe.py files are exact hash-pinned
            copies from the verified saved run. Inference never imports or executes either helper.
            export-manifest.json records qualified canvas sources, the full selected recipe hashes,
            public accumulation settings and selected parameter flags. Selected derivative requests,
            upstream arrays and captured derivative samples are not included in this public metadata.
            derivative-results.json remains inspection evidence and is not part of this package.

            For separate fresh training, apply trainability_recipe.apply_trainability(model), if
            supplied, before constructing a fresh optimizer. Frozen parameters keep their values;
            this does not set evaluation mode or disable influence through the model. The gradient
            helper requires an explicit begin_window, backward_contribution, and finish_window sequence;
            read its signatures and provide your own matching prepared batches and objective roles.
            It does not restore pending gradients, training data, optimizer state or random state.
            Do not combine an accumulated update with training_recipe.training_step: that would
            perform another update. OneCycle advances once per completed optimizer update window;
            epoch schedules still advance after validation. Exact CPU continuation requires the
            matching full captured training contract and data. Fine-tuning starts a fresh optimizer
            under explicitly selected next-run flags; MPS does not promise exact continuation.
            """).utf8)
        }
        let files = bytes.keys.sorted().map { name in
            WorkflowArchiveFile(path: name, bytes: bytes[name]!.count, sha256: WorkflowArchiveCodec.hash(bytes[name]!))
        }
        let manifestBytes = try WorkflowArchiveCodec.encode(WorkflowTrainedExportManifest(runID: value.outcome.runID,
            requestIdentity: value.outcome.requestIdentity, sourceRevision: content.capture.revision,
            modelManifestSHA256: modelPin, files: files,
            schedulerRecipe: try value.plan.schedulerRecipe.map { try WorkflowArchiveSchedulerRecipe($0) },
            initializationRecipe: try value.plan.initializationRecipe.map { try WorkflowArchiveInitializationRecipe($0) },
            gradientPolicy: try value.plan.gradientPolicyRecipe.map {try WorkflowExportGradientPolicy($0)},
            trainability: try value.plan.trainabilityRecipe.map {try WorkflowExportTrainability($0)}))
        guard manifestBytes.count <= 2_000_000 else { throw WorkflowArchiveError("Trained export manifest exceeds its byte limit.") }
        var peak = Int64(manifestBytes.count)
        for data in bytes.values {
            let (sum, overflow) = peak.addingReportingOverflow(Int64(data.count))
            guard !overflow else { throw WorkflowStorageAdmission.Failure.invalidScope }
            peak = sum
        }
        try Task.checkCancellation()
        let stageID = UUID()
        let reservation = try storageLimitBytes.map { limit in
            try WorkflowStorageAdmission(workspace: root, accountingRoot: accountingRoot).reserve(scopes: [
                .init(namespace: .trainedExportStaging, components: [stageID.uuidString])
            ], peakBytes: peak, limitBytes: limit)
        }
        var lifetime: WorkflowExportStorageLifetime?, retained = false
        defer {
            if !retained {
                if let lifetime { lifetime.remove() }
                else { reservation?.close() }
            }
        }
        guard let canonical = realpath(root.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let canonicalRoot = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        free(canonical)
        let rootFD = try WorkflowEvaluationOwnership.openDirectory(canonicalRoot)
        defer { Darwin.close(rootFD) }
        let rootIdentity = try WorkflowEvaluationOwnership.directoryIdentity(rootFD)
        if mkdirat(rootFD, "trained-export-staging", 0o700) != 0, errno != EEXIST { throw WorkflowFileLeaseError.system(errno) }
        let parentFD = openat(rootFD, "trained-export-staging", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parentFD >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(parentFD) }
        let parentIdentity = try WorkflowEvaluationOwnership.directoryIdentity(parentFD)
        let parentURL = canonicalRoot.appendingPathComponent("trained-export-staging")
        func checkDirectory(_ url: URL, identity: WorkflowFileIdentity) throws {
            let current = try WorkflowEvaluationOwnership.openDirectory(url)
            defer { Darwin.close(current) }
            guard try WorkflowEvaluationOwnership.directoryIdentity(current) == identity else { throw WorkflowFileLeaseError.unsafePath }
        }
        try checkDirectory(canonicalRoot, identity: rootIdentity)
        try checkDirectory(parentURL, identity: parentIdentity)
        guard mkdirat(parentFD, stageID.uuidString, 0o700) == 0 else { throw WorkflowFileLeaseError.system(errno) }
        let directory = parentURL.appendingPathComponent(stageID.uuidString)
        let directoryFD = openat(parentFD, stageID.uuidString, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directoryFD >= 0 else { throw WorkflowFileLeaseError.unsafePath }
        defer { Darwin.close(directoryFD) }
        let directoryIdentity = try WorkflowEvaluationOwnership.directoryIdentity(directoryFD)
        try checkDirectory(directory, identity: directoryIdentity)
        lifetime = try WorkflowExportStorageLifetime(directory: directory, reservation: reservation)
        func write(_ name: String, _ data: Data) throws {
            try Task.checkCancellation()
            try checkDirectory(canonicalRoot, identity: rootIdentity)
            try checkDirectory(parentURL, identity: parentIdentity)
            try checkDirectory(directory, identity: directoryIdentity)
            let output = openat(directoryFD, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard output >= 0 else { throw WorkflowFileLeaseError.system(errno) }
            defer { Darwin.close(output) }
            try data.withUnsafeBytes { raw in
                var offset = 0
                while offset < raw.count {
                    try Task.checkCancellation()
                    let written = Darwin.write(output, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                    if written < 0, errno == EINTR { continue }
                    guard written > 0 else { throw WorkflowFileLeaseError.system(errno) }
                    offset += written
                }
            }
            guard fsync(output) == 0 else { throw WorkflowFileLeaseError.system(errno) }
            try checkDirectory(directory, identity: directoryIdentity)
        }
        for name in bytes.keys.sorted() { try write(name, bytes[name]!) }
        try write("export-manifest.json", manifestBytes)
        retained = true
        return .init(reference: reference, runID: value.outcome.runID, requestIdentity: value.outcome.requestIdentity,
                     modelManifestSHA256: modelPin, directory: directory, files: files, manifestBytes: manifestBytes,
                     inputContract: inputContract, storageLifetime: lifetime)
    }

    private static let legacyReason = "V1 records do not bind captured workflow, selected Train and registries sufficiently to reproduce source identity. No verified v2 result is claimed."
    private static func errorSuffix(_ text: String?) -> String { text.map { " Recorded terminal error: " + $0 } ?? "" }
    private static func parts(_ reference: WorkflowHistoryReference) -> [String] { ["workflow-runs", reference.projectID, reference.directoryID.uuidString] }
    private static func sameUUID(_ a: String, _ b: String) -> Bool { UUID(uuidString: a) != nil && UUID(uuidString: a) == UUID(uuidString: b) }
    private static func require(_ condition: Bool, _ message: String) throws { if !condition { throw WorkflowArchiveError(message) } }
    private static func read(_ file: WorkflowArchiveFile, reference: WorkflowHistoryReference, root: URL) throws -> Data {
        let relative = try WorkflowArchiveCodec.pathParts(file.path)
        let bytes = try DiagnosticFileIO.read(root: root, parts: parts(reference) + relative, limit: file.bytes)
        try require(bytes.count == file.bytes && WorkflowArchiveCodec.hash(bytes) == file.sha256, "Archived file size/hash differs: \(file.path).")
        return bytes
    }
    private static func receipt(_ bytes: Data, reference: WorkflowHistoryReference) throws -> Receipt {
        let object = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        guard case .integer(let version) = object["schemaVersion"] else { throw WorkflowArchiveError("Invalid outcome schema version.") }
        if version == 2 || version == 3 {
            let value = try WorkflowArchiveCodec.decodeOutcome(bytes)
            try require(sameUUID(value.projectID, reference.projectID) && UUID(uuidString: value.directoryID) == reference.directoryID, "Outcome belongs to another project or directory.")
            return .current(value)
        }
        try require(version == 1, "Unsupported outcome version.")
        let keys: Set<String> = ["schemaVersion", "runID", "projectID", "requestIdentity", "status", "failedPhase", "preparationManifestSHA256", "modelManifestSHA256", "artifactHashes", "sourceHashes", "error"]
        try require(Set(object.keys).isSubset(of: keys), "Unknown legacy outcome fields.")
        let value = try JSONDecoder().decode(WorkflowLegacyOutcome.self, from: bytes)
        try require(sameUUID(value.projectID, reference.projectID) && UUID(uuidString: value.runID) != nil && WorkflowArchiveCodec.isHash(value.requestIdentity), "Invalid legacy outcome identity.")
        try require([WorkflowExecutionPhase.completed, .failed, .cancelled].contains(value.status) && value.artifactHashes.count <= 320 && value.sourceHashes.count <= 128, "Invalid legacy outcome status or inventory bounds.")
        try require(value.artifactHashes.values.allSatisfy(WorkflowArchiveCodec.isHash) && value.sourceHashes.values.allSatisfy(WorkflowArchiveCodec.isHash), "Invalid legacy evidence hash.")
        try require(value.artifactHashes.keys.allSatisfy { $0.utf8.count <= 1024 } && value.sourceHashes.keys.allSatisfy { $0.utf8.count <= 512 } && (value.error?.utf8.count ?? 0) <= 16_384, "Legacy text exceeds bounds.")
        for pin in [value.preparationManifestSHA256, value.modelManifestSHA256].compactMap({ $0 }) { try require(WorkflowArchiveCodec.isHash(pin), "Invalid legacy manifest pin.") }
        return .legacy(value)
    }
    private static func runtimeEvidence(_ bytes: Data) throws {
        let object = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        let keys: Set<String> = ["schemaVersion", "kind", "packageVersions", "manifestSHA256", "workerFiles", "workerInventorySHA256", "coverage"]
        try require(Set(object.keys).isSubset(of: keys), "Unknown runtime evidence fields.")
        let evidence = try JSONDecoder().decode(WorkflowRuntimeEvidence.self, from: bytes)
        try require(evidence.schemaVersion == 1 && ["bundled", "development"].contains(evidence.kind) && evidence.workerFiles.count <= 514 && !evidence.workerFiles.isEmpty && evidence.packageVersions.count <= 128, "Invalid runtime evidence bounds.")
        try require(evidence.workerFiles.values.allSatisfy(WorkflowArchiveCodec.isHash) && evidence.workerFiles["runner/main.py"] != nil && evidence.workerFiles["components.json"] != nil && evidence.workerFiles["runner/dataflow-operators.json"] != nil, "Runtime evidence has incomplete worker inventory.")
        for name in evidence.workerFiles.keys { _ = try WorkflowArchiveCodec.pathParts(name) }
        try require(WorkflowArchiveCodec.registryHash(evidence.workerFiles) == evidence.workerInventorySHA256, "Runtime worker inventory digest differs.")
        let bundled = evidence.kind == "bundled"
        try require(evidence.coverage == (bundled ? "manifestVerifiedWorkerSourceOnly" : "developmentObservedWorkerSourceOnly") && (bundled ? evidence.manifestSHA256.map(WorkflowArchiveCodec.isHash) == true : evidence.manifestSHA256 == nil), "Runtime evidence coverage differs from its source kind.")
    }
    /// Enumerate beneath descriptor-opened real directories; never chase child links.
    /// Count every child, including malformed names, so hostile directory contents
    /// cannot bypass the scan budget. Enumeration truncation is explicit.
    static func scanDirectories(root: URL, projectID: String, limit: Int, namespace:String="workflow-runs") throws -> (directories: [UUID], issues: [String], truncated: Bool) {
        var fd = Darwin.open(root.standardizedFileURL.resolvingSymlinksInPath().path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { if errno == ENOENT { return ([], [], false) }; throw WorkflowArchiveError("History workspace cannot be opened safely.") }
        guard ["workflow-runs","workflow-experiments","workflow-operations"].contains(namespace),UUID(uuidString:projectID) != nil,(1...10000).contains(limit) else {Darwin.close(fd);throw WorkflowArchiveError("Invalid history scan scope.")}
        for part in [namespace, projectID] {
            let next = openat(fd, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            let failure = errno; Darwin.close(fd)
            guard next >= 0 else { if failure == ENOENT { return ([], [], false) }; throw WorkflowArchiveError("History project directory is unsafe or unavailable.") }
            fd = next
        }
        guard let stream = fdopendir(fd) else { Darwin.close(fd); throw WorkflowArchiveError("History directory cannot be enumerated.") }
        defer { closedir(stream) }
        var directories: [UUID] = [], issues: [String] = [], count = 0
        while true {
            errno = 0
            guard let entry = readdir(stream) else {
                if errno != 0 { throw WorkflowArchiveError("History enumeration failed; a complete listing is unavailable.") }
                break
            }
            let name = withUnsafePointer(to: &entry.pointee.d_name) { pointer in
                pointer.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            if name == "." || name == ".." { continue }
            if count == limit { return (directories, issues, true) }; count += 1
            guard let id = UUID(uuidString: name), id.uuidString == name else { issues.append("Unrecognized history directory entry: " + String(name.prefix(128))); continue }
            directories.append(id)
        }
        return (directories, issues, false)
    }
}
