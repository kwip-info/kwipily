import XCTest
@testable import KwipMLCore

final class RetiredEstimatorBoundaryTests: XCTestCase {
    func fixture() throws -> (WorkflowDocument, ComponentRegistry, DataFlowOperatorRegistry) {
        let (model, registry) = try RetiredDataExecutionBoundaryTests().registries()
        var document = try WorkflowExamples.preparedSklearnClassifier()
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address.localID == "cast" && $0.address.domain == .data })
        document.nodes[index].payload = .dataOperator(operatorID: "kwip.data.cast", config: ["columns": .object(["signal": .string("float64"), "context": .string("float64"), "label": .string("int64")]), "missingPolicy": .string("error")])
        return (document, model, registry)
    }
    func testFreshEstimatorRefusesRetiredDataWithMatchingCapturedRegistry() throws {
        let (document, model, registry) = try fixture()
        XCTAssertThrowsError(try WorkflowEstimatorCompiler.compile(document, estimator: WorkflowExamples.sklearnEstimatorAddress, modelRegistry: model, dataRegistry: registry)) { error in
            XCTAssertTrue(error.localizedDescription.contains("retired data"), error.localizedDescription)
        }
    }
    func testFreshEstimatorAlsoRefusesDisconnectedRetiredTrainingRecipe() throws {
        let (_, model, registry) = try fixture()
        var document = try WorkflowExamples.preparedSklearnClassifier()
        document.schemaVersion = max(10, document.schemaVersion)
        document.nodes.append(.init(address: .init(domain: .control, localID: "old-policy"), payload: .trainabilityRecipe(type: "trainability", configurationVersion: 1, config: [:])))
        XCTAssertThrowsError(try WorkflowEstimatorCompiler.compile(document, estimator: WorkflowExamples.sklearnEstimatorAddress, modelRegistry: model, dataRegistry: registry)) { error in
            XCTAssertTrue(error.localizedDescription.contains("retired training"), error.localizedDescription)
        }
    }
    func testCapturedEstimatorPlanAndCodePreserveOriginalRetiredData() throws {
        let (legacy, model, registry) = try fixture()
        let original = try WorkflowCodec.encode(legacy)
        let current = try WorkflowExamples.preparedSklearnClassifier(projectID: legacy.projectID)
        let currentPlan = try WorkflowEstimatorCompiler.compile(current, estimator: WorkflowExamples.sklearnEstimatorAddress, modelRegistry: model, dataRegistry: registry)
        let historical = try WorkflowEstimatorCompiler.compileHistoricalEvidence(legacy, estimator: WorkflowExamples.sklearnEstimatorAddress, modelRegistry: model, dataRegistry: registry)
        let oldNode = try XCTUnwrap(legacy.nodes.first { $0.address.localID == "cast" && $0.address.domain == .data })
        guard case .dataOperator(let oldID, let oldConfig) = oldNode.payload else { return XCTFail("Missing fixture cast") }
        var expectedGraph = currentPlan.dataflow
        let index = try XCTUnwrap(expectedGraph.nodes.firstIndex { $0.id == "cast" })
        expectedGraph.nodes[index] = .init(id: "cast", operatorID: oldID, config: oldConfig)
        XCTAssertEqual(historical.dataflow, expectedGraph)
        XCTAssertEqual(historical.configuration, currentPlan.configuration)
        XCTAssertEqual(historical.training, currentPlan.training)
        XCTAssertEqual(historical.validation, currentPlan.validation)
        XCTAssertEqual(historical.sourceAddresses, currentPlan.sourceAddresses)
        XCTAssertEqual(historical.model, currentPlan.model)
        let modernCode = try WorkflowCodePlan.compile(current, modelRegistry: model, dataRegistry: registry)
        let prep = try XCTUnwrap(modernCode.preparation)
        var expectedCodeGraph = prep.graph
        let codeIndex = try XCTUnwrap(expectedCodeGraph.nodes.firstIndex { $0.id == "cast" })
        expectedCodeGraph.nodes[codeIndex] = .init(id: "cast", operatorID: oldID, config: oldConfig)
        // Whole-workflow Code retains terminal Views; the estimator dependency slice does not.
        let captured = WorkflowCodePlan(schemaVersion: modernCode.schemaVersion, scope: modernCode.scope, document: legacy, compilerContractVersion: modernCode.compilerContractVersion, modelRegistry: model, dataRegistry: registry, selectedTargets: modernCode.selectedTargets, preparation: .init(graph: expectedCodeGraph, nodeLabels: prep.nodeLabels, engine: prep.engine, sampling: prep.sampling, completeNodes: prep.completeNodes), models: modernCode.models, trainings: modernCode.trainings)
        let bytes = try captured.encoded()
        let decoded = try JSONDecoder().decode(WorkflowCodePlan.self, from: bytes)
        XCTAssertNoThrow(try decoded.validate())
        XCTAssertEqual(try decoded.encoded(), bytes)
        XCTAssertEqual(try WorkflowCodec.encode(legacy), original)
        XCTAssertThrowsError(try WorkflowCodePlan.compile(legacy, modelRegistry: model, dataRegistry: registry))
    }
}
