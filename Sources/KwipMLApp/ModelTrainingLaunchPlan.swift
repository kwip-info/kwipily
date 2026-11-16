import Foundation
import KwipMLCore

/// Capture off-main while the caller owns the run reservation. Original graph provenance
/// remains intact; only worker argument paths point into the eventual owned operation.
struct ModelTrainingLaunchPlan {
    let graph: GraphDocument
    let settings: ExecutionSettings
    let inputs: [WorkflowPreviewCapture.Input]
    private let csvName: String?
    private let continuationMode: CheckpointContinuation.Mode?
    private let compositeSHA256: String?
    private let gradientArguments: [String]
    private let sampleCaptureJSON: String?
    private let trainingControlJSON:String?

    static func capture(graph: GraphDocument, settings: ExecutionSettings,
                        modelRequest: ModelExecutionRequest? = nil, continuation: CheckpointContinuation? = nil,
                        sourcePermissions: URL, supportedGradientDevices: [String], registry: ComponentRegistry? = nil,
                        sampleCapture: TrainingSampleCapturePolicy? = nil,
                        checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> Self {
        try checkCancellation()
        try settings.trainingControl?.validate()
        let controlJSON=try settings.trainingControl.map {String(decoding:try JSONEncoder().encode($0),as:UTF8.self)}
        if let request = modelRequest {
            guard request.graph == graph, request.settings == settings else {
                throw WorkflowArchiveError("The captured model and execution settings differ. Capture the current model again.")
            }
        }
        let sampleCaptureJSON: String?
        if let sampleCapture {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let bytes = try encoder.encode(sampleCapture)
            guard bytes.count <= 1024 else { throw WorkflowArchiveError("Training sample capture policy exceeds its 1 KiB limit.") }
            sampleCaptureJSON = String(decoding: bytes, as: UTF8.self)
        } else { sampleCaptureJSON = nil }
        let gradients = try GradientRunArguments.arguments(config: settings.gradientCapture, supportedDevices: supportedGradientDevices, device: settings.device)
        let graphBytes = try JSONEncoder().encode(graph)
        guard graphBytes.count <= 2_000_000 else { throw WorkflowArchiveError("Captured model exceeds its 2 MB limit.") }
        var inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/graph.json", data: graphBytes)]
        let data = try ModelDataLaunchInputs.capture(settings: settings, sourcePermissions: sourcePermissions, checkCancellation: checkCancellation)
        inputs += data.inputs
        let composite = try captureComposite(graph: graph, settings: settings, request: modelRequest, registry: registry)
        if let composite { inputs.append(.bytes(path: "input/composite/source.json", data: composite)) }
        if let composite, try WorkflowArchiveCodec.decodeCompositeSource(composite).nativeExpansionContractVersion == 2 {
            guard let registry else { throw WorkflowArchiveError("Source3 training requires its captured registry contracts.") }
            let bytes = try WorkflowArchiveCodec.encode(registry)
            guard bytes.count <= 2_000_000 else { throw WorkflowArchiveError("Captured model registry exceeds 2 MB.") }
            inputs.append(.bytes(path: "input/model-registry.json", data: bytes))
        }
        try checkCancellation()
        if let continuation {
            let currentRegistry = try registry ?? ComponentRegistry.builtIn()
            if let message = CheckpointCompatibility.message(continuation, graph: graph, execution: settings, registry: currentRegistry) {
                throw WorkflowArchiveError(message)
            }
            inputs += try captureParent(continuation.parent, checkCancellation: checkCancellation)
        }
        try checkCancellation()
        return Self(graph: graph, settings: settings, inputs: inputs, csvName: data.csvName,
            continuationMode: continuation?.mode, compositeSHA256: composite.map(WorkflowArchiveCodec.hash), gradientArguments: gradients, sampleCaptureJSON: sampleCaptureJSON,trainingControlJSON:controlJSON)
    }
    func arguments(in directory: URL) -> [String] {
        func path(_ relative: String) -> String { directory.appendingPathComponent(relative).path }
        var result = ["--graph", path("input/graph.json"), "--output", path("artifacts"), "--epochs", String(settings.epochs),
            "--seed", String(settings.seed), "--device", settings.device, "--batch-size", String(settings.batchSize), "--learning-rate", String(settings.learningRate)]
        if let trainingControlJSON {result += ["--training-control-json",trainingControlJSON]}
        result += gradientArguments
        if let sampleCaptureJSON { result += ["--sample-capture-json", sampleCaptureJSON] }
        result += ModelDataLaunchInputs.arguments(settings: settings, csvName: csvName, directory: directory)
        if let continuationMode {
            result += [continuationMode == .exactResume ? "--resume-checkpoint" : "--finetune-checkpoint", path("input/continuation-parent/checkpoint.pt")]
        }
        if let compositeSHA256 { result += ["--composite-source", path("input/composite/source.json"), "--composite-source-sha256", compositeSHA256] }
        return result
    }
    private static func captureComposite(graph: GraphDocument, settings: ExecutionSettings, request: ModelExecutionRequest?, registry: ComponentRegistry?) throws -> Data? {
        if let provenance = request?.snapshot.provenance {
            if provenance.nativeExpansionContractVersion == 2 && registry == nil { throw WorkflowArchiveError("Source3 requires captured model contracts.") }
            let contracts = try registry ?? ComponentRegistry.builtIn()
            _ = try provenance.verifiedExpansion(registry: contracts)
            guard provenance.expandedGraph == graph,
                  try CompositeArchitectureIdentity.digest(provenance.expandedGraph, registry: contracts) == request?.snapshot.semanticSHA256 else {
                throw WorkflowArchiveError("Captured Block provenance does not match its executable graph.")
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let bytes = try encoder.encode(provenance)
            guard bytes.count <= 2_000_000 else { throw WorkflowArchiveError("Captured Block provenance exceeds 2 MB.") }
            return bytes
        }
        guard let reference = settings.compositeSource else { return nil }
        let source = URL(fileURLWithPath: reference.path)
        let bytes = try DiagnosticFileIO.read(root: source.deletingLastPathComponent(), parts: [source.lastPathComponent], limit: 2_000_000)
        guard WorkflowArchiveCodec.hash(bytes) == reference.sha256 else { throw WorkflowArchiveError("Composite source snapshot changed. Adopt the source again before training.") }
        let snapshot = try WorkflowArchiveCodec.decodeCompositeSource(bytes)
        if snapshot.nativeExpansionContractVersion == 2 && registry == nil { throw WorkflowArchiveError("Source3 requires captured model contracts.") }
        let contracts = try registry ?? ComponentRegistry.builtIn()
        _ = try snapshot.verifiedExpansion(registry: contracts)
        let identity = try CompositeArchitectureIdentity.digest(snapshot.expandedGraph, registry: contracts)
        let legacy = try CompositeArchitectureIdentity.legacyDigest(snapshot.expandedGraph)
        guard reference.expandedSemanticSHA256 == identity || reference.expandedSemanticSHA256 == legacy else {
            throw WorkflowArchiveError("Composite source architecture identity does not match its selected model.")
        }
        guard try CompositeArchitectureIdentity.digest(graph, registry: contracts) == identity else { return nil }
        return bytes
    }
    private static func captureParent(_ parent: CheckpointParent, checkCancellation: () throws -> Void) throws -> [WorkflowPreviewCapture.Input] {
        try checkCancellation()
        let manifest = try DiagnosticFileIO.read(root: parent.directory, parts: ["manifest.json"], limit: 20_000_000)
        guard WorkflowArchiveCodec.hash(manifest) == parent.manifestSHA256,
              let object = try JSONSerialization.jsonObject(with: manifest) as? [String: Any], object["status"] as? String == "completed",
              object["runID"] as? String == parent.runID, let checkpoint = object["checkpoint"] as? [String: Any],
              checkpoint["schemaVersion"] as? Int == 1, checkpoint["completedEpoch"] as? Int == parent.completedEpoch,
              checkpoint["sha256"] as? String == parent.checkpointSHA256, checkpoint["metricsSHA256"] as? String == parent.metricsSHA256,
              let size = checkpoint["bytes"] as? Int, (0...100_000_000).contains(size),
              let graphObject = object["graph"],
              try JSONDecoder().decode(GraphDocument.self, from: JSONSerialization.data(withJSONObject: graphObject)) == parent.graph,
              let contract = checkpoint["contract"] as? [String: Any], let runtime = contract["runtime"] as? [String: Any],
              runtime["device"] as? String == parent.device,
              ((object["lineage"] as? [String: Any])?["mode"] as? String ?? "newTraining") == parent.lineageMode,
              (object["lineage"] as? [String: Any])?["parentRunID"] as? String == parent.parentRunID else {
            throw WorkflowArchiveError("Selected parent changed after selection. Select it again before continuing.")
        }
        try checkCancellation()
        let metrics = try DiagnosticFileIO.read(root: parent.directory, parts: ["metrics.json"], limit: 4_000_000)
        guard WorkflowArchiveCodec.hash(metrics) == parent.metricsSHA256 else { throw WorkflowArchiveError("Parent metrics changed after selection.") }
        let checkpointURL = parent.directory.appendingPathComponent("checkpoint.pt")
        let info = try checkpointURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard info.isRegularFile == true, info.isSymbolicLink != true, info.fileSize == size,
              WorkflowArchiveCodec.isHash(parent.checkpointSHA256) else { throw WorkflowArchiveError("Parent checkpoint changed after selection.") }
        // Large checkpoint bytes are streamed and rechecked against this retained pin by the service.
        return [.bytes(path: "input/continuation-parent/manifest.json", data: manifest),
                .bytes(path: "input/continuation-parent/metrics.json", data: metrics),
                .file(path: "input/continuation-parent/checkpoint.pt", source: checkpointURL, expectedSHA256: parent.checkpointSHA256)]
    }
}
