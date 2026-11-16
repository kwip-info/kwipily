import Foundation
import KwipMLCore

extension WorkflowPhaseArtifacts {
    static func verifyBagCompletion(event: [String: JSONValue], trained: Trained, graph: GraphDocument,
                                    registry: ComponentRegistry?, sampleCapture: TrainingSampleCapturePolicy?,
                                    root: URL) throws {
        guard let context = trained.bagContext else {
            throw WorkflowPhaseError("Bag completion requires verified packet artifacts.")
        }
        let manifestBytes = try DiagnosticFileIO.read(root: root, parts: ["training", "artifacts", "manifest.json"], limit: 20_000_000)
        let predictionBytes = try DiagnosticFileIO.read(root: root, parts: ["training", "artifacts", "predictions.json"], limit: 1_000_000)
        guard hash(manifestBytes) == trained.manifestSHA256,
              hash(predictionBytes) == trained.hashes["training/artifacts/predictions.json"] else {
            throw WorkflowPhaseError("Bag artifacts changed before completion publication.")
        }
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: manifestBytes)
        let predictions = try JSONDecoder().decode([String: JSONValue].self, from: predictionBytes)
        guard case .object(let evaluation) = manifest["evaluation"], case .object(let runtime) = manifest["runtime"],
              event["runID"] == .string(trained.workerRunID), event["predictions"] == .array([]),
              event["predictionRows"] == predictions["predictionRows"], event["predictionInput"] == predictions["predictionInput"],
              event["objective"] == evaluation["objective"], event["accuracy"] == evaluation["accuracy"],
              event["validationLoss"] == evaluation["validationLoss"], event["evaluationCount"] == evaluation["count"],
              event["device"] == runtime["device"], event["gradientCapture"] == manifest["gradientCapture"],
              trained.activationPreviews == event["previews"] else {
            throw WorkflowPhaseError("Live bag completion differs from its verified saved evidence.")
        }
        try WorkflowModelTrainingExecution.verifyPresentation(event: event, graph: graph, sampleCapture: sampleCapture,
            capturedRegistry: registry, bagContext: context, objectiveEvaluationContext: trained.objectiveEvaluationContext)
    }

    /// Rebuild packet authority from the captured plan and anchored prepared bytes.
    /// A completion event or the mutable editor never supplies this context.
    static func bagTrainingArtifacts(expectation: WorkflowPhaseExpectation, root: URL,
                                     selection: WorkflowPreparedSelection, storageRoot: URL?,
                                     storageParts: [String]) throws -> Trained {
        let plan = expectation.plan
        guard let mapping = plan.bagInput, let registry = expectation.capturedModelRegistry,
              try WorkflowArchiveCodec.registryHash(registry) == expectation.modelRegistrySHA256 else {
            throw WorkflowPhaseError("Bag verification requires the captured mapping and model registry.")
        }
        let validation: GraphValidationContext = plan.lossRecipe == nil ? .preparedWorkflow : .explicitObjective
        let requirement = try WorkflowMappingRequirements.bagInputRequirement(plan.model.graph, registry: registry, context: validation)
        let output = try WorkflowMappingRequirements.outputContract(plan.model.graph, registry: registry)
        let readRoot = storageRoot ?? root
        let prepared = try WorkflowBagPreparedEvidence.preparedEvidence(root: readRoot, parts: storageParts,
            selection: selection, mapping: mapping, requirement: requirement)
        let reference = selection.reference
        guard reference.trainingNode == plan.training.node, reference.trainingPort == plan.training.port,
              reference.validationNode == plan.validation.node, reference.validationPort == plan.validation.port else {
            throw WorkflowPhaseError("Bag preparation endpoints differ from the captured run.")
        }
        let graphBytes = try DiagnosticFileIO.read(root: readRoot,
            parts: storageParts + ["preparation", "artifacts", "dataflow.json"], limit: 2_000_000)
        let registryBytes = try DiagnosticFileIO.read(root: readRoot,
            parts: storageParts + ["preparation", "artifacts", "dataflow-operators.json"], limit: 2_000_000)
        guard try DataFlowCodec.decodeGraph(graphBytes) == plan.dataflow,
              try WorkflowArchiveCodec.registryHash(JSONDecoder().decode(DataFlowOperatorRegistry.self, from: registryBytes)) == expectation.dataRegistrySHA256 else {
            throw WorkflowPhaseError("Bag preparation graph or data registry differs from the captured run.")
        }
        let target: BagArtifactTarget
        let legacy: PreparedTrainingSelection?
        let objective: PreparedObjectiveSelection?
        switch selection {
        case .legacy(let value):
            guard plan.lossRecipe == nil, plan.targetColumn == value.targetColumn,
                  plan.featureColumns == value.featureColumns else {
                throw WorkflowPhaseError("Bag binary selection differs from the captured columns.")
            }
            target = .binary(value.targetColumn); legacy = value; objective = nil
        case .objective(let value):
            guard let recipe = plan.lossRecipe, plan.targetColumn == nil,
                  value.objectiveRecipe == recipe.configuration.recipeValue,
                  value.targetMapping == recipe.targetMapping,
                  value.scalarization == plan.objectiveScalarization else {
                throw WorkflowPhaseError("Bag objective selection differs from the captured recipe and roles.")
            }
            target = .objective(value, recipe); legacy = nil; objective = value
        }
        let parts = storageParts + ["training", "artifacts"]
        let manifestBytes = try DiagnosticFileIO.read(root: readRoot, parts: parts + ["manifest.json"], limit: 20_000_000)
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: manifestBytes)
        guard let runID = manifest["runID"]?.stringValue, !runID.isEmpty,
              case .object(let runtime) = manifest["runtime"],
              let deviceName = runtime["device"]?.stringValue, let device = ContractDevice(rawValue: deviceName),
              [.cpu, .mps].contains(device) else {
            throw WorkflowPhaseError("Bag training omitted its actual run or device identity.")
        }
        let context = WorkflowBagArtifactContext(requirement: requirement, mapping: mapping,
            preparation: prepared, runID: runID, device: device, outputContract: output, target: target,
            policy: expectation.sampleCapture ?? .factory, explicitSampleCapture: expectation.sampleCapture != nil)
        var gradientEvidence: PreparedGradientEvidence?
        if plan.gradientPolicyRecipe != nil || plan.trainabilityRecipe != nil || plan.datasetRecipe != nil || plan.loaderRecipe != nil {
            let bytes = try DiagnosticFileIO.read(root: readRoot,
                parts: storageParts + ["preparation", "artifacts", "manifest.json"], limit: 2_000_000)
            guard hash(bytes) == selection.manifestSHA256 else {
                throw WorkflowPhaseError("Bag derivative preparation changed during verification.")
            }
            let train = prepared.training, val = prepared.validation
            gradientEvidence = .init(rows: .init(trainingRows: train.count, validationRows: val.count,
                trainingTableSHA256: train.tableSHA256, validationTableSHA256: val.tableSHA256,
                trainingRowIDs: train.rowIDs, validationRowIDs: val.rowIDs),
                manifest: bytes, graphDigest: selection.graphDigest)
        }
        return try trainingArtifacts(graph: plan.model.graph, settings: plan.settings, preparation: legacy,
            compositeSource: plan.model.provenance, trainingMode: plan.trainingMode,
            requiresLifecycleEvidence: expectation.requiresLifecycleEvidence, lifecycleParent: expectation.lifecycleParent,
            readRoot: readRoot, parts: parts, artifactPrefix: "training/artifacts/",
            expectedCPUThreads: expectation.expectedCPUThreads, sampleCapture: expectation.sampleCapture,
            optimizerRecipe: plan.optimizerRecipe, schedulerRecipe: plan.schedulerRecipe,
            initializationRecipe: plan.initializationRecipe?.configuration, objectiveSelection: objective,
            objectiveRecipe: plan.lossRecipe, capturedModelRegistry: registry,
            capturedModelRegistrySHA256: expectation.modelRegistrySHA256,
            gradientPolicy: plan.gradientPolicyRecipe?.configuration, trainability: plan.trainabilityRecipe?.configuration,
            derivativeCapture: expectation.derivativeCapture, preparedGradientEvidence: gradientEvidence, bagContext: context, datasetRecipe:plan.datasetRecipe, loaderRecipe:plan.loaderRecipe,trainOrigin:plan.train,inlineRoles:plan.inlineDataLoadingRoles)
    }
}
