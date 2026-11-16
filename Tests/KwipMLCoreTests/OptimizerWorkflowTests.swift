import XCTest
@testable import KwipMLCore

final class OptimizerWorkflowTests: XCTestCase {
    private var data: DataFlowOperatorRegistry { get throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try .load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
    } }
    private func compile(_ document: WorkflowDocument, version: Int = 12) throws -> WorkflowExecutionPlan {
        let train = try XCTUnwrap(document.nodes.first { if case .train = $0.payload { return true }; return false })
        return try WorkflowCompiler.compile(document, train: train.address, modelRegistry: .builtIn(), dataRegistry: data, contractVersion: version)
    }
    func testFiveExplicitRecipesOwnRateAndRemainOutsideModelForward() throws {
        for id in OptimizerConfiguration.componentIDs {
            var doc = try WorkflowExamples.preparedOptimizerClassifier(componentID: id, device: "cpu")
            let index = try XCTUnwrap(doc.nodes.firstIndex { if case .train = $0.payload { return true }; return false })
            doc.nodes[index].payload = .train(config:["optimizer":try OptimizerConfiguration(componentID:id,values:["lr":0]).recipeValue])
            let plan = try compile(doc)
            XCTAssertEqual(plan.optimizerRecipe?.configuration.componentID, id)
            XCTAssertEqual(plan.settings.learningRate, 0)
            XCTAssertFalse(plan.model.graph.nodes.contains { $0.type.hasPrefix("optimizer_") })
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)), doc)
            XCTAssertThrowsError(try compile(doc, version: 4))
        }
    }
    func testDisconnectedRecipePreservesLegacyEffectiveRate() throws {
        var doc = try WorkflowExamples.preparedSGDClassifier(device: "cpu")
        let i=try XCTUnwrap(doc.nodes.firstIndex {if case .train=$0.payload {return true};return false})
        doc.nodes[i].payload = .train(config:[:])
        let plan = try compile(doc)
        XCTAssertNil(plan.optimizerRecipe)
        let old = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        XCTAssertEqual(plan.settings, try compile(old, version: 12).settings)
    }
    func testRecipePortIsOptionalTypedAndRejectsWrongWires() throws {
        var doc = try WorkflowExamples.preparedAdamWClassifier(device: "cpu")
        let train = try XCTUnwrap(doc.nodes.first { if case .train = $0.payload { return true }; return false })
        let ports = try WorkflowPortContracts.resolve(train, document: doc, modelRegistry: .builtIn(), dataRegistry: data)
        XCTAssertEqual(ports.inputs.map(\.id),["model","data","configuration"])
        let address=WorkflowNodeAddress(domain:.control,localID:"obsolete")
        doc.nodes.append(.init(address:address,payload:.optimizerRecipe(type:"optimizer_adamw",configurationVersion:1,config:[:])))
        doc.edges.append(.init(from:.init(address:address,port:"out"),to:.init(address:train.address,port:"optimizer"),kind:.optimizerRecipe))
        XCTAssertThrowsError(try compile(doc))
        XCTAssertThrowsError(try RetiredTrainingRecipeMigration.migrate(doc),"Inline and legacy authorities cannot be silently merged.")
    }
    func testExperimentRatesTargetRecipeAndRefuseInactiveSetupRate() throws {
        let doc = try WorkflowExamples.preparedSGDClassifier(device: "cpu")
        let train = try XCTUnwrap(doc.nodes.first { if case .train = $0.payload { return true }; return false })
        let budget = WorkflowExperimentBudget(maximumRuns: 2, maximumTotalEpochs: 100)
        let plan = try WorkflowExperimentPlan.build(doc, train: train.address, axes: [.init(field: "optimizer.lr", values: [0, 0.05])], budget: budget, modelRegistry: .builtIn(), dataRegistry: data)
        XCTAssertEqual(plan.variants.map { $0.execution.settings.learningRate }, [0, 0.05])
        XCTAssertEqual(plan.variants[0].execution.optimizerRecipe?.configuration.learningRate, 0)
        XCTAssertThrowsError(try WorkflowExperimentPlan.build(doc, train: train.address, axes: [.init(field: "learningRate", values: [0.01, 0.05])], budget: budget, modelRegistry: .builtIn(), dataRegistry: data))
        let legacy = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        let terminal = try XCTUnwrap(legacy.nodes.first { if case .train = $0.payload { return true }; return false })
        XCTAssertThrowsError(try WorkflowExperimentPlan.build(legacy, train: terminal.address, axes: [.init(field: "optimizer.lr", values: [0.01])], budget: budget, modelRegistry: .builtIn(), dataRegistry: data))
    }
    func testCanonicalExamplesCompileAndCanBeEmittedForHostVerification() throws {
        for id in OptimizerConfiguration.componentIDs.sorted() {
            let document = try WorkflowExamples.preparedOptimizerClassifier(componentID: id, device: "auto")
            let plan = try compile(document)
            XCTAssertEqual(plan.optimizerRecipe?.configuration.componentID, id)
            XCTAssertEqual(plan.settings.epochs, 3)
            XCTAssertEqual(plan.settings.batchSize, 32)
            XCTAssertEqual(plan.featureColumns.count, 2)
            XCTAssertEqual(plan.optimizerRecipe?.configuration.learningRate, 0.03)
            if let path = ProcessInfo.processInfo.environment["KWIPML_OPTIMIZER_CANONICAL_FIXTURES"] {
                let directory = URL(fileURLWithPath: path).appendingPathComponent(id)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
                try encoder.encode(document).write(to: directory.appendingPathComponent("workflow.json"))
                try encoder.encode(plan.model.graph).write(to: directory.appendingPathComponent("model.json"))
                try encoder.encode(plan.dataflow).write(to: directory.appendingPathComponent("dataflow.json"))
                try encoder.encode(plan.optimizerRecipe!.configuration.recipeValue).write(to: directory.appendingPathComponent("recipe.json"))
                try encoder.encode(plan.settings).write(to: directory.appendingPathComponent("settings.json"))
            }
        }
    }
}
