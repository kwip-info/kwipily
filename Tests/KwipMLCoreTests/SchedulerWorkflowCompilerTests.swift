import XCTest
@testable import KwipMLCore
final class SchedulerWorkflowCompilerTests:XCTestCase {
    private func compile(_ document:WorkflowDocument,version:Int=12)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let train=try XCTUnwrap(document.nodes.first {if case .train=$0.payload{return true};return false})
        return try WorkflowCompiler.compile(document,train:train.address,modelRegistry:.builtIn(),dataRegistry:data,contractVersion:version)
    }
    func testAllFiveResolveQualifiedRecipesAndPreserveModel()throws {
        for id in SchedulerConfiguration.componentIDs {
            let document=try WorkflowExamples.preparedSchedulerClassifier(componentID:id,device:"cpu")
            let plan=try compile(document)
            XCTAssertEqual(plan.schedulerRecipe?.configuration.componentID,id)
            XCTAssertEqual(plan.schedulerRecipe?.source,WorkflowExamples.trainAddress)
            XCTAssertEqual(plan.optimizerRecipe?.configuration.componentID,"optimizer_sgd")
            XCTAssertFalse(plan.model.graph.nodes.contains {$0.type.hasPrefix("scheduler_")})
            XCTAssertThrowsError(try compile(document,version:5))
            if let path = ProcessInfo.processInfo.environment["KWIPML_SCHEDULER_CANONICAL_FIXTURES"] {
                let directory = URL(fileURLWithPath: path).appendingPathComponent(id)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
                try encoder.encode(document).write(to: directory.appendingPathComponent("workflow.json"))
                try encoder.encode(plan.model.graph).write(to: directory.appendingPathComponent("model.json"))
                try encoder.encode(plan.dataflow).write(to: directory.appendingPathComponent("dataflow.json"))
                try encoder.encode(plan.optimizerRecipe!.configuration.recipeValue).write(to: directory.appendingPathComponent("optimizer-recipe.json"))
                try encoder.encode(plan.schedulerRecipe!.configuration.recipeValue).write(to: directory.appendingPathComponent("scheduler-recipe.json"))
                try encoder.encode(plan.settings).write(to: directory.appendingPathComponent("settings.json"))
            }
        }
    }
    func testPreparedMenuFactoriesResolveAllFiveRealWorkflows()throws {
        for id in SchedulerConfiguration.componentIDs {
            let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
            XCTAssertEqual(try compile(document).schedulerRecipe?.configuration.componentID,id)
        }
    }
    func testKnownIncompatibleOptimizerControlsRefuseBeforeExecution()throws {
        let cases:[(String,String,[String:JSONValue],[String:JSONValue])]=[
            ("scheduler_cosine_annealing_lr","optimizer_sgd",["eta_min":0.1],["lr":0.03]),
            ("scheduler_one_cycle_lr","optimizer_adagrad",[:],["lr":0.03]),
            ("scheduler_one_cycle_lr","optimizer_adam",["max_momentum":1],["lr":0.03]),
            ("scheduler_one_cycle_lr","optimizer_sgd",["base_momentum":0],["lr":0.03,"momentum":0.5,"nesterov":true])]
        for (scheduler,optimizer,controls,options) in cases {
            var document=try WorkflowExamples.preparedSchedulerClassifier(componentID:scheduler,device:"cpu")
            let train=try XCTUnwrap(document.nodes.firstIndex {if case .train=$0.payload{return true};return false})
            XCTAssertThrowsError(try {
                document.nodes[train].payload = .train(config:["scheduler":try SchedulerConfiguration(componentID:scheduler,values:controls).recipeValue,"optimizer":try OptimizerConfiguration(componentID:optimizer,values:options).recipeValue])
                return try compile(document)
            }(),scheduler+" / "+optimizer)
        }
    }
    func testSchedulerExperimentVariantsRetainQualifiedRecipe()throws {
        let document=try WorkflowExamples.preparedStepLRClassifier(device:"cpu")
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let baseline=try compile(document)
        let variants=try WorkflowExperimentPlan.build(document,train:baseline.train,axes:[.init(field:"optimizer.lr",values:[0.01,0.05])],budget:.init(maximumRuns:2,maximumTotalEpochs:6),modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(variants.variants.count,2)
        XCTAssertEqual(variants.variants.map { $0.execution.schedulerRecipe },[baseline.schedulerRecipe,baseline.schedulerRecipe])
        XCTAssertEqual(variants.variants.map { $0.execution.settings.learningRate },[0.01,0.05])
    }
    func testSchedulerRequiresExplicitOptimizerAndDisconnectedIsIgnored()throws {
        var document=try WorkflowExamples.preparedStepLRClassifier(device:"cpu")
        let i=try XCTUnwrap(document.nodes.firstIndex {if case .train=$0.payload{return true};return false})
        guard case .train(var configuration)=document.nodes[i].payload else {return XCTFail("Train missing")}
        configuration.removeValue(forKey:"optimizer");document.nodes[i].payload = .train(config:configuration)
        XCTAssertThrowsError(try compile(document))
        configuration.removeValue(forKey:"scheduler");document.nodes[i].payload = .train(config:configuration)
        XCTAssertNil(try compile(document).schedulerRecipe)
        let old=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        XCTAssertNil(try compile(old,version:5).schedulerRecipe)
    }
}
