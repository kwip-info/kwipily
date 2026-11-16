import XCTest
@testable import KwipMLCore

final class WorkflowTrainRecipeTests:XCTestCase {
    private var registry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    private func document(_ config:[String:JSONValue])throws->WorkflowDocument {
        var value=try WorkflowExamples.preparedParallelClassifier(device:"cpu");value.schemaVersion=13
        let index=try XCTUnwrap(value.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        value.nodes[index].payload = .train(config:config)
        return value
    }
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13)}
    func testInlineOptimizerUsesExistingRecipeAndTrainOrigin()throws {
        let recipe=try OptimizerConfiguration(componentID:"optimizer_sgd",values:["lr":.number(0.05)])
        let value=try document(["optimizer":recipe.recipeValue]),plan=try compile(value)
        XCTAssertEqual(plan.optimizerRecipe?.configuration,recipe)
        XCTAssertEqual(plan.optimizerRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(plan.settings.learningRate,0.05)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(value)),value)
    }
    private func allLegacyRecipes()throws->WorkflowDocument {
        var value=try GradientWorkflowExamples.document(task:.keepSettings,device:"cpu")
        let before=try compile(value),owner=try XCTUnwrap(before.model.graph.nodes.first {$0.type == "linear"})
        let train=try XCTUnwrap(value.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        value.nodes[train].payload = .train(config:[:])
        for index in value.nodes.indices {value.nodes[index].parameterSettings=nil}
        let initial:[String:JSONValue]=["seed":13,"rules":.array([.object(["owner":.object(["node":.string(owner.id),"parameter":.string("weight")]),"algorithm":.string("zeros"),"settings":.object([:])])])]
        let controls:[(String,WorkflowNodePayload,WorkflowEdgeKind)]=[
            ("optimizer",.optimizerRecipe(type:try XCTUnwrap(before.optimizerRecipe).configuration.componentID,configurationVersion:1,config:try XCTUnwrap(before.optimizerRecipe).configuration.values),.optimizerRecipe),
            ("objective",.lossRecipe(type:try XCTUnwrap(before.lossRecipe).configuration.componentID,configurationVersion:1,config:try XCTUnwrap(before.lossRecipe).configuration.values),.objectiveRecipe),
            ("gradientPolicy",.gradientPolicyRecipe(type:"gradient_policy",configurationVersion:1,config:try XCTUnwrap(before.gradientPolicyRecipe).configuration.values),.gradientPolicyRecipe),
            ("trainability",.trainabilityRecipe(type:"trainability",configurationVersion:1,config:try XCTUnwrap(before.trainabilityRecipe).configuration.values),.trainabilityRecipe),
            ("scheduler",.schedulerRecipe(type:"scheduler_step_lr",configurationVersion:1,config:try SchedulerConfiguration(componentID:"scheduler_step_lr",values:["step_size":1,"gamma":.number(0.5)]).values),.schedulerRecipe),
            ("initialization",.initializationRecipe(type:"initialization",configurationVersion:1,config:initial),.initializationRecipe)]
        for (role,payload,kind) in controls {
            let address=WorkflowNodeAddress(domain:.control,localID:role)
            value.nodes.append(.init(address:address,payload:payload))
            value.edges.append(.init(from:.init(address:address,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:role),kind:kind))
        }
        return value
    }
    func testAllSixMigrateToCanonicalInlineRecipesPreservingExecutionAndObjectiveContext()throws {
        let original=try allLegacyRecipes(),before=try WorkflowCompiler.compileHistoricalEvidence(original,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13)
        let value=try WorkflowTrainRecipes.migrateLegacyRecipes(original),after=try compile(value)
        XCTAssertEqual(after.inlineTrainingRecipeRoles,WorkflowTrainRecipes.roleKeys)
        XCTAssertEqual(after.optimizerRecipe?.configuration,before.optimizerRecipe?.configuration)
        XCTAssertEqual(after.schedulerRecipe?.configuration,before.schedulerRecipe?.configuration)
        XCTAssertEqual(after.lossRecipe?.configuration,before.lossRecipe?.configuration)
        XCTAssertEqual(after.lossRecipe?.targetMapping,before.lossRecipe?.targetMapping)
        XCTAssertEqual(after.initializationRecipe?.configuration,before.initializationRecipe?.configuration)
        XCTAssertEqual(after.gradientPolicyRecipe?.configuration,before.gradientPolicyRecipe?.configuration)
        XCTAssertEqual(after.trainabilityRecipe?.configuration,before.trainabilityRecipe?.configuration)
        XCTAssertEqual(after.model.graph,before.model.graph);XCTAssertEqual(after.model.semanticSHA256,before.model.semanticSHA256)
        XCTAssertEqual(after.settings,before.settings);XCTAssertEqual(after.dataflow,before.dataflow)
        XCTAssertEqual(value.edges.filter {$0.to.address == WorkflowExamples.trainAddress}.map(\.to.port).sorted(),["configuration","data","model"])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(value)),value)
        XCTAssertEqual(try WorkflowTrainRecipes.migrateLegacyRecipes(value),value)
        let mapping=try XCTUnwrap(value.nodes.first {if case .trainingData=$0.payload {return true};return false})
        let requirements=try WorkflowMappingRequirements.resolve(value,mapping:mapping.address,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(requirements.count,1);XCTAssertTrue(requirements.allSatisfy {$0.problem == nil})
        let code=try WorkflowCodePlan.compile(value,scope:.training,selectedTargets:[WorkflowExamples.trainAddress],modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(code.document.modelSourceVersions,value.modelSourceVersions,"Code retains model provenance on its complete captured source.")
        let projection=try XCTUnwrap(code.trainings.first)
        XCTAssertEqual(projection.recipeSources["optimizer_recipe"],WorkflowExamples.trainAddress)
        XCTAssertEqual(projection.recipeSources["objective_recipe"],WorkflowExamples.trainAddress)
    }
    func testSharedRecipesMigrateToEachTrainAndOrphansConflictsRefuse()throws {
        var value=try allLegacyRecipes()
        let second=WorkflowNodeAddress(domain:.control,localID:"secondTrain")
        value.nodes.append(.init(address:second,payload:.train(config:[:])))
        value.edges += value.edges.filter {$0.to.address == WorkflowExamples.trainAddress}.map {.init(id:$0.id+"-second",from:$0.from,to:.init(address:second,port:$0.to.port),kind:$0.kind)}
        let migrated=try WorkflowTrainRecipes.migrateLegacyRecipes(value)
        let first=try XCTUnwrap(migrated.nodes.first {$0.address == WorkflowExamples.trainAddress}),other=try XCTUnwrap(migrated.nodes.first {$0.address == second})
        XCTAssertEqual(first.payload,other.payload)
        var conflict=try allLegacyRecipes()
        let index=try XCTUnwrap(conflict.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        conflict.nodes[index].payload = .train(config:["optimizer":try OptimizerConfiguration(componentID:"optimizer_sgd",values:[:]).recipeValue])
        XCTAssertThrowsError(try WorkflowTrainRecipes.migrateLegacyRecipes(conflict))
        var orphan=try allLegacyRecipes();orphan.edges.removeAll {$0.to.port == "initialization"}
        XCTAssertThrowsError(try WorkflowTrainRecipes.migrateLegacyRecipes(orphan))
    }
    func testMalformedEnvelopesAndSchedulerWithoutOptimizerRefuse()throws {
        for (key,value) in [("optimizer",JSONValue.null),("scheduler",.object([:])),("objective",try OptimizerConfiguration(componentID:"optimizer_sgd",values:[:]).recipeValue)] {
            XCTAssertThrowsError(try WorkflowCodec.validate(document([key:value])),key)
        }
        let scheduler=try SchedulerConfiguration(componentID:"scheduler_step_lr",values:["step_size":1,"gamma":.number(0.5)])
        XCTAssertThrowsError(try compile(document(["scheduler":scheduler.recipeValue])))
        let omitted=try compile(document([:]))
        XCTAssertTrue(omitted.inlineTrainingRecipeRoles.isEmpty)
        XCTAssertNil(omitted.optimizerRecipe);XCTAssertNil(omitted.schedulerRecipe)
    }
    func testOldCompositeMigrationRetainsSourceTwoModelSemantics()throws {
        var old=try WorkflowExamples.preparedParallelClassifier(device:"cpu");old.schemaVersion=6
        let address=WorkflowNodeAddress(domain:.control,localID:"legacyOptimizer")
        old.nodes.append(.init(address:address,payload:.optimizerRecipe(type:"optimizer_sgd",configurationVersion:1,config:["lr":.number(0.02)])))
        old.edges.append(.init(from:.init(address:address,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"optimizer"),kind:.optimizerRecipe))
        let before=try WorkflowCompiler.compileHistoricalEvidence(old,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13),migrated=try WorkflowTrainRecipes.migrateLegacyRecipes(old),after=try compile(migrated)
        XCTAssertEqual(migrated.schemaVersion,14)
        XCTAssertEqual(migrated.modelSourceVersions?["model"],2)
        XCTAssertEqual(after.model.graph,before.model.graph)
        XCTAssertEqual(after.model.semanticSHA256,before.model.semanticSHA256)
        XCTAssertEqual(after.optimizerRecipe?.configuration,before.optimizerRecipe?.configuration)
    }
    func testOptimizerExperimentAxisChangesInlineRecipeWithoutRenamingAxis()throws {
        let source=try WorkflowTrainRecipes.migrateLegacyRecipes(allLegacyRecipes())
        let plan=try WorkflowExperimentPlan.build(source,train:WorkflowExamples.trainAddress,axes:[.init(field:"optimizer.lr",values:[.number(0.01),.number(0.02)])],budget:.init(maximumRuns:2,maximumTotalEpochs:2),modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(plan.variants.map {$0.execution.optimizerRecipe?.configuration.learningRate},[0.01,0.02])
        XCTAssertTrue(plan.variants.allSatisfy {$0.execution.inlineTrainingRecipeRoles.contains("optimizer")})
    }

}
