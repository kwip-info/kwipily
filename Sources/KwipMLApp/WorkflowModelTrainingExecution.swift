import Foundation
import KwipMLCore

@MainActor enum WorkflowModelTrainingExecution {
    static func launch(workspace: URL, accountingRoot: URL? = nil, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, operationID: String = UUID().uuidString, storageLimitBytes: Int64? = nil,
                       inputs: @escaping (URL) throws -> [WorkflowPreviewCapture.Input], arguments: (URL) -> [String],
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onLine: @escaping (String) -> Void, onExit: @escaping (Int32, String, URL?) -> Void) async throws -> WorkflowWorkerHandle {
        try await WorkflowCapturedToolExecution.launch(mode: .modelTraining, workspace: workspace, accountingRoot: accountingRoot, projectID: projectID,
            runtime: runtime, window: window, operationID: operationID, storageLimitBytes: storageLimitBytes, inputs: inputs, arguments: arguments,
            launcher: launcher, onLine: onLine, onExit: onExit)
    }
    nonisolated static func verifyResult(directory: URL, request: WorkflowToolRequest, completion: Data,
                                         window: ExecutionBudgetWindow) throws -> URL {
        func check() throws { try Task.checkCancellation(); _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime) }
        try check()
        try request.validate(directory: directory, verifyInputs: true)
        let output = directory.appendingPathComponent("artifacts")
        let event = try JSONDecoder().decode([String: JSONValue].self, from: completion)
        guard completion.count <= 1_000_000, event["event"] == .string("completed"),
              event["outputDir"] == .string(output.path) else { throw WorkflowPhaseError("Training completion identifies a different output directory.") }
        var args: [String: String] = [:], index = 0
        while index < request.arguments.count {
            let flag = request.arguments[index]
            if flag == "--capture-gradients" { args[flag] = "true"; index += 1 }
            else { args[flag] = request.arguments[index + 1]; index += 2 }
        }
        func required(_ flag: String) throws -> String {
            guard let value = args[flag] else { throw WorkflowFileLeaseError.unsafePath }; return value
        }
        func integer(_ flag: String) throws -> Int {
            guard let value = Int(try required(flag)) else { throw WorkflowFileLeaseError.unsafePath }; return value
        }
        let graphBytes = try DiagnosticFileIO.read(root: directory, parts: ["input", "graph.json"], limit: 2_000_000)
        guard request.inputFiles.contains(.init(path: "input/graph.json", bytes: graphBytes.count, sha256: WorkflowArchiveCodec.hash(graphBytes))) else { throw WorkflowFileLeaseError.unsafePath }
        let graph = try JSONDecoder().decode(GraphDocument.self, from: graphBytes)
        var settings = graph.execution ?? ExecutionSettings()
        settings.epochs = try integer("--epochs"); settings.seed = try integer("--seed")
        settings.batchSize = try integer("--batch-size"); settings.device = try required("--device")
        guard let learningRate = Double(try required("--learning-rate")) else { throw WorkflowFileLeaseError.unsafePath }
        settings.learningRate = learningRate
        settings.trainingControl=try args["--training-control-json"].map {try JSONDecoder().decode(TrainingControlPolicy.self,from:Data($0.utf8))}
        if args["--capture-gradients"] != nil {
            settings.gradientCapture = .init(everyEpochs: try integer("--gradient-every-epochs"),
                maxBatchesPerEpoch: try integer("--gradient-max-batches-per-epoch"), maxCaptures: try integer("--gradient-max-captures"))
        } else { settings.gradientCapture = nil }
        var prepared: PreparedTrainingSelection?
        if let manifest = args["--prepared-dataflow"] {
            let names = try JSONDecoder().decode([String].self, from: Data(required("--feature-columns-json").utf8))
            prepared = try .init(manifestPath: manifest, manifestSHA256: required("--prepared-manifest-sha256"), graphDigest: required("--prepared-graph-digest"),
                trainingNode: required("--prepared-training-node"), trainingPort: required("--prepared-training-port"),
                validationNode: required("--prepared-validation-node"), validationPort: required("--prepared-validation-port"),
                featureColumns: names, targetColumn: required("--target-column"))
        }
        var composite: CompositeSourceSnapshot?
        if args["--composite-source"] != nil {
            let data = try DiagnosticFileIO.read(root: directory, parts: ["input", "composite", "source.json"], limit: 2_000_000)
            composite = try WorkflowArchiveCodec.decodeCompositeSource(data)
            guard WorkflowArchiveCodec.hash(data) == args["--composite-source-sha256"] else { throw WorkflowFileLeaseError.unsafePath }
        }
        let mode: WorkflowTrainingMode = args["--resume-checkpoint"] != nil ? .exactResume : args["--finetune-checkpoint"] != nil ? .fineTune : .fresh
        var parent: WorkflowLifecycleParentEvidence?
        if mode != .fresh {
            let bytes = try DiagnosticFileIO.read(root: directory, parts: ["input", "continuation-parent", "manifest.json"], limit: 20_000_000)
            struct Parent: Decodable { let runID: String; let checkpoint: Checkpoint; struct Checkpoint: Decodable { let completedEpoch: Int; let sha256: String; let metricsSHA256: String } }
            guard request.inputFiles.contains(.init(path: "input/continuation-parent/manifest.json", bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes))) else { throw WorkflowFileLeaseError.unsafePath }
            let parentInputs=request.inputFiles.filter {$0.path.hasPrefix("input/continuation-parent/")}
            var contents:[String:Data]=[:]
            for file in parentInputs {
                let name=String(file.path.dropFirst("input/continuation-parent/".count))
                contents[name]=try DiagnosticFileIO.read(root:directory,parts:["input","continuation-parent",name],limit:WorkflowModelTrainingToolContract.fileLimit(path:file.path))
            }
            try WorkflowCheckpointInventory.verify(contents)
            let source = try JSONDecoder().decode(Parent.self, from: bytes)
            guard request.inputFiles.first(where: { $0.path == "input/continuation-parent/checkpoint.pt" })?.sha256 == source.checkpoint.sha256,
                  request.inputFiles.first(where: { $0.path == "input/continuation-parent/metrics.json" })?.sha256 == source.checkpoint.metricsSHA256 else { throw WorkflowFileLeaseError.unsafePath }
            parent = .init(workerRunID: source.runID, checkpointSHA256: source.checkpoint.sha256, completedEpoch: source.checkpoint.completedEpoch,hasScheduler:try WorkflowLifecycleParentRecord.hasScheduler(bytes))
        }
        let sampleCapture = try args["--sample-capture-json"].map {
            try JSONDecoder().decode(TrainingSampleCapturePolicy.self, from: Data($0.utf8))
        }
        let optimizerConfiguration=try args["--optimizer-recipe-json"].map {raw in try OptimizerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)))}
        let schedulerConfiguration=try args["--scheduler-recipe-json"].map {raw in try SchedulerConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)))}
        let initializationConfiguration = try args["--initialization-recipe-json"].map { raw in
            try InitializationConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8)))
        }
        let capturedRegistry: ComponentRegistry?
        if composite?.nativeExpansionContractVersion == 2 || graph.compositionPlan != nil {
            let registryBytes = try DiagnosticFileIO.read(root: directory, parts: ["input","model-registry.json"], limit: 2_000_000)
            guard request.inputFiles.contains(.init(path: "input/model-registry.json", bytes: registryBytes.count, sha256: WorkflowArchiveCodec.hash(registryBytes))) else { throw WorkflowFileLeaseError.unsafePath }
            let registry = try JSONDecoder().decode(ComponentRegistry.self, from: registryBytes)
            capturedRegistry = registry
            if let composite { _ = try composite.verifiedExpansion(registry: registry) }
        } else { capturedRegistry = nil }
        let verified = try WorkflowPhaseArtifacts.trainingArtifacts(graph: graph, settings: settings, preparation: prepared,
            compositeSource: composite, trainingMode: mode, requiresLifecycleEvidence: true, lifecycleParent: parent,
            readRoot: directory, parts: ["artifacts"], artifactPrefix: "artifacts/", expectedCPUThreads: request.executionBudget.resources?.cpuThreads, sampleCapture:sampleCapture,optimizerConfiguration:optimizerConfiguration,schedulerConfiguration:schedulerConfiguration,initializationRecipe:initializationConfiguration,capturedModelRegistry:capturedRegistry)
        guard event["runID"] == .string(verified.workerRunID) else { throw WorkflowPhaseError("Training completion run identity differs from its manifest.") }
        let manifestBytes = try DiagnosticFileIO.read(root: output, parts: ["manifest.json"], limit: 20_000_000)
        guard WorkflowArchiveCodec.hash(manifestBytes) == verified.manifestSHA256 else { throw WorkflowFileLeaseError.unsafePath }
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: manifestBytes)
        let expectedCapture = try sampleCapture.map {
            try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode($0))
        }
        guard event["sampleCapture"] == expectedCapture, manifest["sampleCapture"] == expectedCapture else {
            throw WorkflowPhaseError("Training sample capture differs from the policy accepted for this run.")
        }
        if let sampleCapture,sampleCapture.savesComponentOutputs {
            let expectedStatus: PredictionPreview.CaptureStatus = sampleCapture.predictionRows ? .captured : .omittedByPolicy
            guard verified.predictionPreview?.captureStatus == expectedStatus else {
                throw WorkflowPhaseError("Saved prediction samples differ from the accepted capture policy.")
            }
        }
        guard case .object(let evaluation) = manifest["evaluation"], case .object(let runtime) = manifest["runtime"],
              event["evaluationCount"] == evaluation["count"], event["accuracy"] == evaluation["accuracy"],
              event["validationLoss"] == evaluation["validationLoss"], event["device"] == runtime["device"],
              event["gradientCapture"] == manifest["gradientCapture"], event["performanceCapture"] == manifest["performanceCapture"] else { throw WorkflowPhaseError("Training completion differs from its saved evaluation or gradient evidence.") }
        try WorkflowLifecycleVerifier.validateAttentionExecution(runtime,graph:graph)
        guard event["attentionExecution"] == runtime["attentionExecution"] else {throw WorkflowPhaseError("Training completion differs from its saved attention execution policy.")}
        if sampleCapture?.savesComponentOutputs != false {
        let predictions = try JSONDecoder().decode([String: JSONValue].self,
            from: DiagnosticFileIO.read(root: output, parts: ["predictions.json"], limit: 1_000_000))
        guard verified.predictionPreview != nil, event["predictionRows"] == predictions["predictionRows"],
              event["predictionInput"] == predictions["predictionInput"] else { throw WorkflowPhaseError("Training completion differs from saved predictions.") }
        }
        if let saved = verified.activationPreviews, saved != event["previews"] {
            throw WorkflowPhaseError("Training completion differs from saved activation samples.")
        }
        try verifyPresentation(event: event, graph: graph, sampleCapture: sampleCapture)
        if prepared == nil { try verifyGeneratedData(manifest: manifest, args: args, request: request) }
        else {
            // The worker archives the full preparation inventory for portable reuse. Verify that
            // durable copy against the captured pins before private operation inputs are removed.
            for file in request.inputFiles where file.path.hasPrefix("input/preparation/") {
                try check()
                let archived = WorkflowArchiveFile(path: String(file.path.dropFirst("input/".count)), bytes: file.bytes, sha256: file.sha256)
                try WorkflowPreviewToolContract.verifyFile(file: archived, directory: output,
                    limit: WorkflowModelTrainingToolContract.fileLimit(path: file.path))
            }
        }
        if let composite {
            let archived = try DiagnosticFileIO.read(root: output, parts: ["composite", "source.json"], limit: 2_000_000)
            guard WorkflowArchiveCodec.hash(archived) == args["--composite-source-sha256"],
                  try JSONDecoder().decode(CompositeSourceSnapshot.self, from: archived) == composite else { throw WorkflowFileLeaseError.unsafePath }
        }
        try check()
        return output
    }
    private nonisolated static func verifyGeneratedData(manifest: [String: JSONValue], args: [String: String], request: WorkflowToolRequest) throws {
        guard case .object(let settings) = manifest["settings"], case .object(let provenance) = manifest["provenance"] else { throw WorkflowFileLeaseError.unsafePath }
        for (flag, key) in [("--sample-limit", "sampleLimit"), ("--split-seed", "splitSeed"), ("--validation-percent", "validationPercent")] {
            guard let raw = args[flag], let value = Int64(raw), settings[key] == .integer(value), provenance[key] == .integer(value) else { throw WorkflowPhaseError("Training split settings differ from capture.") }
        }
        guard case .string(let hash) = provenance["sourceHash"], WorkflowArchiveCodec.isHash(hash) else { throw WorkflowFileLeaseError.unsafePath }
        if let path = args["--dataset-csv"] {
            guard request.inputFiles.first(where: { $0.path == "input/data/" + URL(fileURLWithPath: path).lastPathComponent })?.sha256 == hash,
                  provenance["dataset"] == .string("local-csv-v1"), provenance["sourceName"] == .string(URL(fileURLWithPath: path).lastPathComponent),
                  let names = args["--feature-columns-json"], provenance["featureColumns"] == (try JSONDecoder().decode(JSONValue.self, from: Data(names.utf8))),
                  provenance["targetColumn"] == args["--target-column"].map(JSONValue.string) else { throw WorkflowPhaseError("Training source provenance differs from the captured CSV.") }
        } else { guard provenance["dataset"] == .string("kwipml-two-moons-v2") else { throw WorkflowFileLeaseError.unsafePath } }
    }
    /// Live validation only. The synthetic envelope reuses saved-row checks;
    /// its locally computed pin is not evidence of an archived file identity.
    nonisolated static func objectivePresentation(event:[String:JSONValue],context c:WorkflowObjectiveArtifactContext)throws->ObjectivePredictionPreview? {
        guard event["runID"] == .string(c.runID),event["device"] == .string(c.device.rawValue),
              event["evaluationCount"]?.integerValue == c.evaluationCount,
              event["predictions"] == .array([]),let objective=event["objective"],
              let loss=event["validationLoss"]?.doubleValue,loss.isFinite else {throw WorkflowPhaseError("Live objective presentation differs from the captured completion contract.")}
        let policy=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(c.policy))
        guard event["sampleCapture"] == (c.explicitSampleCapture ? policy:nil) else {throw WorkflowPhaseError("Live objective capture permission differs.")}
        let evaluation=try WorkflowObjectiveArtifacts.evaluation(objective,context:c)
        guard abs(loss-evaluation.effectiveScalar)<=max(1e-6,max(abs(loss),abs(evaluation.effectiveScalar))*1e-5) else {throw WorkflowPhaseError("Live objective score differs from the measured aggregate.")}
        if let expected=evaluation.accuracy {
            guard let accuracy=event["accuracy"]?.doubleValue,accuracy.isFinite,abs(accuracy-expected)<=1e-6 else {throw WorkflowPhaseError("Live classification accuracy differs from measured counts.")}
        } else {guard event["accuracy"] == nil else {throw WorkflowPhaseError("This objective has no measured classification accuracy.")}}
        guard let input=event["predictionInput"],let rows=event["predictionRows"] else {throw WorkflowPhaseError("Live objective rows are missing.")}
        let schema=c.usesNamedValues ? 4:c.explicitSampleCapture ? 2:1
        var artifact:[String:JSONValue]=["schemaVersion":.integer(Int64(schema)),"runID":.string(c.runID),"selection":.string("firstRowsOfValidationPartition"),"predictionInput":input,"predictionRows":rows,"evaluationCount":.integer(Int64(c.evaluationCount))]
        if c.explicitSampleCapture || c.usesNamedValues {artifact["captureStatus"] = .string(c.policy.predictionRows ? "captured":"omittedByPolicy")}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let data=try encoder.encode(artifact),pin:JSONValue = .object(["schemaVersion":.integer(Int64(schema)),"file":.string("predictions.json"),"bytes":.integer(Int64(data.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(data))])
        return try WorkflowObjectiveArtifacts.predictions(data,metadata:pin,context:c)
    }
    nonisolated static func verifyPresentation(event: [String: JSONValue], graph: GraphDocument, sampleCapture: TrainingSampleCapturePolicy?, objectiveContext:WorkflowObjectiveArtifactContext? = nil, capturedRegistry:ComponentRegistry? = nil, bagContext:WorkflowBagArtifactContext? = nil, objectiveEvaluationContext:WorkflowObjectiveEvaluationContext? = nil) throws {
        if sampleCapture?.savesComponentOutputs == false {
            guard event["previews"]==nil,event["predictions"]==nil,event["predictionRows"]==nil,event["predictionInput"]==nil else {throw WorkflowPhaseError("Disabled History capture cannot publish optional output samples.")}
            return
        }
        guard case .object(let previews) = event["previews"], Set(previews.keys) == Set(graph.nodes.map(\.id)),
              case .array(let predictions) = event["predictions"], predictions.count <= 200,
              case .array(let rows) = event["predictionRows"], case .object(let input) = event["predictionInput"] else { throw WorkflowFileLeaseError.unsafePath }
        if let bagContext {
            guard objectiveContext == nil,let registry=capturedRegistry else {throw WorkflowPhaseError("Bag presentation requires its captured registry and distinct packet context.")}
            try WorkflowBagActivationEvidence.presentation(event,context:bagContext,objectiveContext:objectiveEvaluationContext)
            try WorkflowBagActivationEvidence.validate(previews,graph:graph,registry:registry,sampleCapture:sampleCapture,context:bagContext)
            return
        }
        guard objectiveEvaluationContext == nil else {throw WorkflowPhaseError("Input-independent completion context requires a captured bag input.")}
        if let objectiveContext {
            guard capturedRegistry != nil,sampleCapture == (objectiveContext.explicitSampleCapture ? objectiveContext.policy:nil) else {throw WorkflowPhaseError("Objective activation presentation requires captured metadata and exact policy.")}
            _ = try objectivePresentation(event:event,context:objectiveContext)
            try verifyActivationPreviews(previews,graph:graph,sampleCapture:sampleCapture,objectiveContext:objectiveContext,capturedRegistry:capturedRegistry)
            return
        }
        var expected: [JSONValue] = []
        if input["dtype"] == .string("float32"), input["sampleShape"] == .array([.integer(2)]) {
            for row in rows {
                guard case .object(let fields) = row, case .array(let values) = fields["input"], values.count == 2,
                      let label = fields["label"], let predicted = fields["predicted"] else { throw WorkflowFileLeaseError.unsafePath }
                expected.append(.object(["x": values[0], "y": values[1], "label": label, "predicted": predicted]))
            }
        }
        guard predictions == expected else { throw WorkflowPhaseError("Legacy prediction presentation differs from saved tensor predictions.") }
        try verifyActivationPreviews(previews, graph:graph, sampleCapture:sampleCapture, capturedRegistry:capturedRegistry)
    }
    nonisolated static func verifyActivationPreviews(_ previews:[String:JSONValue], graph:GraphDocument,
                                                    sampleCapture:TrainingSampleCapturePolicy?, objectiveContext:WorkflowObjectiveArtifactContext? = nil, capturedRegistry:ComponentRegistry? = nil, bagContext:WorkflowBagArtifactContext? = nil,maximumEpoch:Int=10000) throws {
        if (sampleCapture?.heatmapFrameLimit ?? 0)>0 {
            for preview in previews.values {
                guard case .object(let raw)=preview else {continue}
                let outputs: [JSONValue]
                if case .object(let ports)?=raw["outputs"] {outputs=Array(ports.values)} else {outputs=[preview]}
                for output in outputs {
                    if case .object(let tensor)=output,case .object(let playback)?=tensor["heatmapPlayback"],case .array(let frames)?=playback["frames"] {
                        for frame in frames {
                            guard case .object(let fields)=frame,let epoch=fields["epoch"]?.integerValue,epoch<=maximumEpoch else {throw WorkflowPhaseError("Heatmap frame exceeds the completed training epoch.")}
                        }
                    }
                }
            }
        }
        if let bagContext {
            guard objectiveContext == nil,let registry=capturedRegistry else {throw WorkflowPhaseError("Bag activations require their captured registry and distinct packet context.")}
            try WorkflowBagActivationEvidence.validate(previews,graph:graph,registry:registry,sampleCapture:sampleCapture,context:bagContext)
            return
        }
        guard Set(previews.keys) == Set(graph.nodes.map(\.id)) else { throw WorkflowFileLeaseError.unsafePath }
        if let objectiveContext {guard capturedRegistry != nil,sampleCapture == (objectiveContext.explicitSampleCapture ? objectiveContext.policy:nil) else {throw WorkflowPhaseError("Objective previews require their captured registry and exact capture policy.")}}
        let registry=try capturedRegistry ?? ComponentRegistry.builtIn()
        let attentionTypes = AttentionConfiguration.componentIDs
        let recurrentTypes = RecurrentConfiguration.componentIDs
        let routingValidation = graph.nodes.contains { $0.type == "split" || $0.type == "chunk" || recurrentTypes.contains($0.type) || attentionTypes.contains($0.type) }
            || capturedRegistry != nil || objectiveContext != nil
            ? try GraphValidator.validate(graph, registry:registry, context:objectiveContext != nil ? .explicitObjective : capturedRegistry != nil ? .preparedWorkflow : .bundledDataset) : nil
        if capturedRegistry != nil,let validation=routingValidation,!validation.issues.isEmpty {throw WorkflowPhaseError("Captured model metadata cannot resolve its exact activation contracts.")}
        // Resolve contracts before entering the per-node verification frame. History
        // verification also holds the authenticated training manifest on this stack.
        try verifyResolvedActivationPreviews(previews, graph:graph, sampleCapture:sampleCapture,
            objectiveContext:objectiveContext, registry:registry, routingValidation:routingValidation)
    }

    /// Keep the contract resolver and the larger per-node verification frame disjoint.
    /// This boundary preserves the same validation on cooperative worker stacks.
    @inline(never)
    private nonisolated static func verifyResolvedActivationPreviews(_ previews:[String:JSONValue], graph:GraphDocument,
                                                                    sampleCapture:TrainingSampleCapturePolicy?, objectiveContext:WorkflowObjectiveArtifactContext?,
                                                                    registry:ComponentRegistry, routingValidation:ValidationResult?) throws {
        let attentionTypes = AttentionConfiguration.componentIDs
        let recurrentTypes = RecurrentConfiguration.componentIDs
        func verifyTensor(_ item: [String: JSONValue], expectedType: String? = nil, status: JSONValue?) throws {
            var item=item
            let heatmap=item.removeValue(forKey:"heatmap")
            let playback=item.removeValue(forKey:"heatmapPlayback")
            if (sampleCapture?.heatmapFrameLimit ?? 0)>0 {
                guard let playback else {throw WorkflowPhaseError("Missing requested heatmap playback evidence.")}
                _=try TensorHeatmapPlayback(value:playback,maximumFrames:sampleCapture!.heatmapFrameLimit)
            } else if playback != nil {throw WorkflowPhaseError("Undeclared heatmap playback capture.")}
            guard Set(item.keys) == ["shape", "dtype", "values"],
                  case .array(let shape) = item["shape"], (1...8).contains(shape.count),
                  shape.allSatisfy({ $0.integerValue.map { $0 > 0 && $0 <= 100_000_000 } ?? false }),
                  case .string(let dtype) = item["dtype"], ["float32", "int64"].contains(dtype),
                  expectedType == nil || dtype == expectedType,
                  case .array(let values) = item["values"], values.count <= 8,
                  sampleCapture?.activationValues != false || values.isEmpty,
                  status == sampleCapture.map({ .string($0.activationValues ? "captured" : "omittedByPolicy") }),
                  values.allSatisfy({ value in
                    if dtype == "int64" { if case .integer = value { return true }; return false }
                    switch value { case .integer: return true; case .number(let number): return number.isFinite; default: return false }
                  }) else { throw WorkflowPhaseError("Training tensor previews exceed their display contract.") }
            if sampleCapture?.tensorHeatmaps == true {
                guard let heatmap else {throw WorkflowPhaseError("Missing requested tensor heatmap evidence.")}
                _=try TensorHeatmapSummary(value:heatmap,expectedShape:shape.compactMap {value in value.integerValue.map {Int($0)}})
            } else if heatmap != nil {throw WorkflowPhaseError("Undeclared tensor heatmap capture.")}
        }
        for node in graph.nodes {
            guard case .object(var item) = previews[node.id] else { throw WorkflowFileLeaseError.unsafePath }
            let status = item.removeValue(forKey: "captureStatus")
            if attentionTypes.contains(node.type) {
                let registry=registry
                let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                let configuration=try AttentionConfiguration(componentID:node.type,values:values)
                var layouts:[BatchedTensorLayout]=[]
                var observations:JSONValue?
                for port in configuration.configuredInputs {
                    let incoming=graph.edges.filter {$0.target == node.id && $0.targetPort == port}
                    guard incoming.count == 1,
                          let contract=routingValidation?.outputTensorContracts[.init(incoming[0].source,incoming[0].sourcePort)],
                          case .object(var sourcePreview)=previews[incoming[0].source] else {throw WorkflowPhaseError("Attention previews have no resolved source or mask contract.")}
                    let sourceStatus=sourcePreview.removeValue(forKey:"captureStatus")
                    let sourceTensor:[String:JSONValue]
                    if case .object(let named)=sourcePreview["outputs"] {
                        guard case .object(let selected)=named[incoming[0].sourcePort] else {throw WorkflowPhaseError("Attention preview omits a connected named source.")}
                        sourceTensor=selected
                    } else {sourceTensor=sourcePreview}
                    try verifyTensor(sourceTensor,expectedType:contract.dtype.rawValue,status:sourceStatus)
                    if contract.dtype == .int64,case .array(let numbers)=sourceTensor["values"] {
                        guard numbers.allSatisfy({$0 == .integer(0) || $0 == .integer(1)}) else {throw WorkflowPhaseError("Attention mask previews require binary whole numbers: 1 blocked, 0 allowed.")}
                    }
                    guard case .array(let shape)=sourceTensor["shape"],let count=shape.first,
                          Array(shape.dropFirst()) == contract.layout.sample.dimensions.map({.integer(Int64($0))}),
                          observations == nil || observations == count else {throw WorkflowPhaseError("Attention sources and masks differ from resolved directions or observation count.")}
                    observations=count;layouts.append(contract.layout)
                }
                let resolved=try configuration.resolved(layouts:layouts)
                guard let observations,Set(item.keys) == ["outputs"],case .object(let outputs)=item["outputs"],Set(outputs.keys) == Set(resolved.outputs.keys) else {throw WorkflowPhaseError("Attention previews differ from their exact named results, weights or mask roles.")}
                for (port,layout) in resolved.outputs {
                    guard let contract=routingValidation?.outputTensorContracts[.init(node.id,port)],
                          contract.layout == layout,contract.dtype == resolved.outputDTypes[port],case .object(let tensor)=outputs[port],
                          tensor["shape"] == .array([observations]+layout.sample.dimensions.map({.integer(Int64($0))})) else {throw WorkflowPhaseError("Attention preview differs from its resolved named shape or type.")}
                    try verifyTensor(tensor,expectedType:contract.dtype.rawValue,status:status)
                    if contract.dtype == .int64,case .array(let numbers)=tensor["values"] {
                        guard numbers.allSatisfy({$0 == .integer(0) || $0 == .integer(1)}) else {throw WorkflowPhaseError("Attention mask output previews must contain binary whole numbers.")}
                    }
                }
            } else if recurrentTypes.contains(node.type) {
                let registry = registry
                let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                let configuration = try RecurrentConfiguration(componentID: node.type, values: values)
                var layouts: [BatchedTensorLayout] = []
                var observations: JSONValue?
                for port in configuration.configuredInputs {
                    let incoming = graph.edges.filter { $0.target == node.id && $0.targetPort == port }
                    guard incoming.count == 1,
                          let contract = routingValidation?.outputTensorContracts[.init(incoming[0].source, incoming[0].sourcePort)],
                          contract.dtype == .float32,
                          case .object(var sourcePreview) = previews[incoming[0].source] else {
                        throw WorkflowPhaseError("Training memory previews have no resolved input or state contract.")
                    }
                    let sourceStatus = sourcePreview.removeValue(forKey: "captureStatus")
                    let sourceTensor: [String: JSONValue]
                    if case .object(let named) = sourcePreview["outputs"] {
                        guard case .object(let selected) = named[incoming[0].sourcePort] else {
                            throw WorkflowPhaseError("Training memory previews omit a connected named source.")
                        }
                        sourceTensor = selected
                    } else { sourceTensor = sourcePreview }
                    try verifyTensor(sourceTensor, expectedType: "float32", status: sourceStatus)
                    guard case .array(let shape) = sourceTensor["shape"], let count = shape.first,
                          Array(shape.dropFirst()) == contract.layout.sample.dimensions.map({ .integer(Int64($0)) }),
                          observations == nil || observations == count else {
                        throw WorkflowPhaseError("Training memory input and state previews differ from their resolved shapes or observation count.")
                    }
                    observations = count
                    layouts.append(contract.layout)
                }
                let resolved = try configuration.resolved(layouts: layouts)
                guard let observations, Set(item.keys) == ["outputs"],
                      case .object(let outputs) = item["outputs"], Set(outputs.keys) == Set(resolved.outputs.keys) else {
                    throw WorkflowPhaseError("Training memory previews differ from their exact named output roles.")
                }
                if node.type.hasSuffix("_cell"), outputs["out"] != outputs["hidden"] {
                    throw WorkflowPhaseError("Training memory cell result and hidden previews differ from their exact alias contract.")
                }
                for (port, layout) in resolved.outputs {
                    guard let contract = routingValidation?.outputTensorContracts[.init(node.id, port)],
                          contract.dtype == .float32, contract.layout == layout,
                          case .object(let tensor) = outputs[port],
                          tensor["shape"] == .array([observations] + layout.sample.dimensions.map({ .integer(Int64($0)) })) else {
                        throw WorkflowPhaseError("Training memory preview differs from its resolved named output shape.")
                    }
                    try verifyTensor(tensor, expectedType: "float32", status: status)
                }
            } else if node.type == "split" || node.type == "chunk" {
                let incoming = graph.edges.filter { $0.target == node.id && $0.targetPort == "in" }
                guard incoming.count == 1,
                      let sourceContract = routingValidation?.outputTensorContracts[.init(incoming[0].source, incoming[0].sourcePort)],
                      case .object(let sourcePreview) = previews[incoming[0].source] else {
                    throw WorkflowPhaseError("Training routing previews have no resolved source contract.")
                }
                let sourceTensor: [String: JSONValue]
                if case .object(let named) = sourcePreview["outputs"],
                   case .object(let selected) = named[incoming[0].sourcePort] { sourceTensor = selected }
                else { sourceTensor = sourcePreview }
                guard case .array(let sourceShape) = sourceTensor["shape"],
                      let observations = sourceShape.first,
                      Array(sourceShape.dropFirst()) == sourceContract.layout.sample.dimensions.map({ .integer(Int64($0)) }),
                      Set(item.keys) == ["outputs"], case .object(let outputs) = item["outputs"] else {
                    throw WorkflowPhaseError("Training routing previews differ from their source contract.")
                }
                let routing = try TensorRoutingConfiguration(componentID: node.type, values: node.config)
                let resolved = try routing.resolved(layouts: [sourceContract.layout])
                guard Set(outputs.keys) == Set(resolved.keys) else {
                    throw WorkflowPhaseError("Training routing previews differ from the actual named outputs.")
                }
                for (port, layout) in resolved {
                    guard case .object(let tensor) = outputs[port],
                          tensor["shape"] == .array([observations] + layout.sample.dimensions.map({ .integer(Int64($0)) })) else {
                        throw WorkflowPhaseError("Training routing preview differs from its named output shape.")
                    }
                    try verifyTensor(tensor, expectedType: "float32", status: status)
                }
            } else if let bundle = BundledPoolingComponents.entries[node.type], bundle.published,
               bundle.outputConditions.contains(where: { node.config[$0.value] == .bool(true) }) {
                guard Set(item.keys) == ["outputs"], case .object(let outputs) = item["outputs"],
                      Set(outputs.keys) == ["out", "indices"],
                      case .object(let values) = outputs["out"], case .object(let indices) = outputs["indices"],
                      values["shape"] == indices["shape"] else {
                    throw WorkflowPhaseError("Training tensor previews differ from the named output contract.")
                }
                try verifyTensor(values, expectedType: "float32", status: status)
                try verifyTensor(indices, expectedType: "int64", status: status)
            } else {
                try verifyTensor(item, status: status)
                if let objectiveContext {
                    guard let contract=routingValidation?.outputTensorContracts[.init(node.id,"out")],
                          item["dtype"] == .string(contract.dtype.rawValue),
                          item["shape"] == .array([.integer(Int64(min(8,objectiveContext.evaluationCount)))]+contract.layout.sample.dimensions.map {.integer(Int64($0))}) else {throw WorkflowPhaseError("Objective activation preview differs from its captured output shape or dtype.")}
                }
            }
        }
    }
}
