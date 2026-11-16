import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

final class BagPhaseArtifactTests: XCTestCase {
    func testActualBinaryBagRunUsesTypedPhaseArtifactReader() throws { try check(objective: false) }
    func testActualObjectiveBagRunUsesSharedEvaluationAndTypedRows() throws { try check(objective: true) }
    func testCapturedBinaryBagExpectationBuildsItsOwnVerifiedContext() throws {try check(objective:false,throughExpectation:true)}
    func testCapturedObjectiveBagExpectationBuildsItsOwnVerifiedContext() throws {try check(objective:true,throughExpectation:true)}
    func testActualLiveBagCompletionBindsSavedEvidenceAndRejectsMutations() throws {
        for objective in [false, true] {
            let fixtures = BagActivationArtifactTests(), f = try fixtures.fixture(objective: objective)
            defer {try? FileManager.default.removeItem(at: f.root)}
            let c = try fixtures.context(f)
            let legacy: PreparedTrainingSelection?, selected: PreparedObjectiveSelection?, recipe: LossRecipe?
            switch c.target {
            case .binary: guard case .legacy(let value) = f.selection else {return XCTFail("Binary preparation missing")}; legacy=value;selected=nil;recipe=nil
            case .objective(let value, let loss): legacy=nil;selected=value;recipe=loss
            }
            let trained = try WorkflowPhaseArtifacts.trainingArtifacts(graph:f.graph,settings:.init(device:"cpu",epochs:1,batchSize:3),preparation:legacy,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:true,lifecycleParent:nil,readRoot:f.root,parts:["training","artifacts"],artifactPrefix:"training/artifacts/",sampleCapture:c.policy,objectiveSelection:selected,objectiveRecipe:recipe,capturedModelRegistry:f.registry,capturedModelRegistrySHA256:WorkflowArchiveCodec.registryHash(f.registry),bagContext:c)
            func verify(_ event:[String:JSONValue]) throws {
                try WorkflowPhaseArtifacts.verifyBagCompletion(event:event,trained:trained,graph:f.graph,registry:f.registry,sampleCapture:c.policy,root:f.root)
            }
            try verify(f.event)
            for key in ["predictionRows", "predictionInput", "accuracy", "validationLoss", "evaluationCount", "device", "gradientCapture"] {
                var changed=f.event;changed[key] = .string("foreign")
                XCTAssertThrowsError(try verify(changed),key)
            }
        }
    }
    func testActualCaptureOffBinaryAndObjectiveBagsKeepRequiredModelEvidence()throws {for objective in [false,true] {try check(objective:objective,throughExpectation:true,retention:false)}}
    private func check(objective: Bool, throughExpectation: Bool = false,retention:Bool=true) throws {
        let fixtures = BagArtifactTests(), f = try fixtures.fixture(objective: objective,retention:retention)
        defer { try? FileManager.default.removeItem(at: f.root) }
        let context = try fixtures.context(f)
        let graph = try JSONDecoder().decode(GraphDocument.self, from: WorkflowPhaseArtifacts.encode(XCTUnwrap(f.manifest["graph"])))
        let registry = try BagWorkflowArchiveTests().registry()
        let preparation: PreparedTrainingSelection?, selected: PreparedObjectiveSelection?, recipe: LossRecipe?
        switch f.target {
        case .binary: guard case .legacy(let value)=f.selection else {return XCTFail("Actual binary selection missing")};preparation=value;selected=nil;recipe=nil
        case .objective(let value,let loss):preparation=nil;selected=value;recipe=loss
        }
        let result:WorkflowPhaseArtifacts.Trained
        if throughExpectation {
            let dataflow=try DataFlowCodec.decodeGraph(Data(contentsOf:f.root.appendingPathComponent("preparation/artifacts/dataflow.json")))
            let dataRegistry=try JSONDecoder().decode(DataFlowOperatorRegistry.self,from:Data(contentsOf:f.root.appendingPathComponent("preparation/artifacts/dataflow-operators.json")))
            let snapshot=try ModelExecutionSnapshotBuilder.build(.legacyFlat(graph),registry:registry,context:objective ? .explicitObjective:.preparedWorkflow)
            let reference=f.selection.reference
            var plan=WorkflowExecutionPlan(projectID:UUID().uuidString,revision:0,train:WorkflowExamples.trainAddress,dataflow:dataflow,sourceAddresses:[],model:snapshot,training:.init(node:reference.trainingNode,port:reference.trainingPort),validation:.init(node:reference.validationNode,port:reference.validationPort),featureColumns:f.mapping.indicesColumns,targetColumn:objective ? nil:"label",settings:.init(device:"cpu",epochs:1,batchSize:3),preparationCachePolicy:.off,trainingMode:.fresh)
            plan.bagInput=f.mapping;plan.lossRecipe=recipe
            let expectation=WorkflowPhaseExpectation(plan:plan,modelRegistrySHA256:try WorkflowArchiveCodec.registryHash(registry),dataRegistrySHA256:try WorkflowArchiveCodec.registryHash(dataRegistry),compilerContractVersion:11,sampleCapture:context.policy,capturedModelRegistry:registry)
            result=try WorkflowPhaseArtifacts.training(expectation:expectation,root:f.root,preparation:f.selection)
        } else {
            result = try WorkflowPhaseArtifacts.trainingArtifacts(graph:graph,settings:.init(device:"cpu",epochs:1,batchSize:3),preparation:preparation,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:true,lifecycleParent:nil,readRoot:f.root,parts:["training","artifacts"],artifactPrefix:"training/artifacts/",sampleCapture:context.policy,objectiveSelection:selected,objectiveRecipe:recipe,capturedModelRegistry:registry,capturedModelRegistrySHA256:WorkflowArchiveCodec.registryHash(registry),bagContext:context)
        }
        XCTAssertEqual(result.workerRunID,context.runID)
        XCTAssertNil(result.predictionPreview);XCTAssertNil(result.objectivePredictionPreview)
        if retention {XCTAssertEqual(result.bagPredictionPreview?.rows.count,8)}else {XCTAssertNil(result.bagPredictionPreview);XCTAssertNil(result.activationPreviews)}
        XCTAssertEqual(result.partitionEvidence?.trainingCount,16)
        XCTAssertEqual(result.metrics.count,1)
        if retention {XCTAssertNotNil(result.activationPreviews)}
        if objective {XCTAssertNotNil(result.objectiveEvaluationContext);XCTAssertNotNil(result.objectiveEvaluation);XCTAssertNotNil(result.objectiveModelCode);XCTAssertNil(result.metrics.first?.accuracy)}
        else {XCTAssertNil(result.objectiveEvaluationContext);XCTAssertNotNil(result.metrics.first?.accuracy)}
    }
}
