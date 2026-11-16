import XCTest
@testable import KwipMLCore

final class WorkflowCodePlanTests:XCTestCase {
    func testImportedCausalTrainingExportsCapturedTaskWithoutPrivateModelPaths()throws {
        var document=try FineTuningPracticeExample.document()
        let plan=try compile(document,scope:.training,targets:[FineTuningPracticeExample.train])
        let training=try XCTUnwrap(plan.trainings.first)
        XCTAssertEqual(training.backendID,"pretrainedCausal");XCTAssertNil(training.graph);XCTAssertNil(training.settings)
        XCTAssertEqual(training.configuration?["model"],.string(FineTuningPracticeExample.model.localID))
        try plan.validate()
        XCTAssertThrowsError(try compile(document),"Generate / Save consumers must never be silently omitted")
        document.nodes.removeAll{$0.address==FineTuningPracticeExample.generate || $0.address==FineTuningPracticeExample.save}
        document.edges.removeAll{$0.to.address==FineTuningPracticeExample.generate || $0.to.address==FineTuningPracticeExample.save}
        try compile(document).validate()
    }
    private func registry()throws->DataFlowOperatorRegistry {
        try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))
    }
    private func compile(_ document:WorkflowDocument,scope:WorkflowCodePlan.Scope = .workflow,targets:[WorkflowNodeAddress] = [])throws->WorkflowCodePlan {
        try .compile(document,scope:scope,selectedTargets:targets,modelRegistry:.builtIn(),dataRegistry:registry())
    }
    func testDataOnlyDoesNotRequireModelAndCapturesLabelsWithoutGrants()throws {
        let source=WorkflowNodeAddress(domain:.data,localID:"source")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:source,name:"My local table",payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")]))])
        let plan=try compile(document)
        XCTAssertEqual(plan.document,document)
        XCTAssertEqual(plan.preparation?.graph.nodes.map(\.id),["source"])
        XCTAssertEqual(plan.preparation?.nodeLabels,["source":"My local table"])
        XCTAssertTrue(plan.models.isEmpty);XCTAssertTrue(plan.trainings.isEmpty)
        let encoded=try plan.encoded(),text=String(decoding:encoded,as:UTF8.self)
        for privateKey in ["sourceBindings","bookmark","datasetCSV","preparedTraining","workingDirectory"] {XCTAssertFalse(text.contains(privateKey))}
        XCTAssertEqual(try JSONDecoder().decode(WorkflowCodePlan.self,from:encoded),plan)
        XCTAssertEqual(try plan.contentIdentity,try plan.contentIdentity)
    }
    func testTrainingProjectionUsesExistingModelMappingAndSelectedRecipes()throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        document.schemaVersion=13
        let dataset=WorkflowNodeAddress(domain:.control,localID:"rows")
        let loader=WorkflowNodeAddress(domain:.control,localID:"batches")
        document.nodes.append(.init(address:dataset,payload:.datasetRecipe(type:DatasetConfiguration.componentID,configurationVersion:1,config:DatasetConfiguration.defaults)))
        document.nodes.append(.init(address:loader,payload:.batchRecipe(type:DataLoaderConfiguration.componentID,configurationVersion:1,config:DataLoaderConfiguration.defaults)))
        document.edges.append(.init(from:.init(address:dataset,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"dataset"),kind:.datasetRecipe))
        document.edges.append(.init(from:.init(address:loader,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"loader"),kind:.batchRecipe))
        let historical=try WorkflowCompiler.compileHistoricalEvidence(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry(),contractVersion:12)
        XCTAssertEqual(historical.datasetRecipe?.source,dataset)
        XCTAssertEqual(historical.loaderRecipe?.source,loader)
        document=try RetiredTrainingRecipeMigration.migrate(document)
        let plan=try compile(document),training=try XCTUnwrap(plan.trainings.first)
        let expected=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry(),contractVersion:12)
        XCTAssertEqual(training.id,"train");XCTAssertEqual(training.backendID,"pytorch")
        XCTAssertEqual(training.settings?.epochs,expected.settings.epochs)
        XCTAssertEqual(training.prepared.featureColumns,expected.featureColumns)
        XCTAssertEqual(training.prepared.trainingNode,expected.training.node)
        XCTAssertEqual(training.recipes["dataset_recipe"],expected.datasetRecipe?.configuration.recipeValue)
        XCTAssertEqual(training.recipes["loader_recipe"],expected.loaderRecipe?.configuration.recipeValue)
        XCTAssertEqual(training.recipeSources["dataset_recipe"],WorkflowExamples.trainAddress)
        XCTAssertEqual(training.recipeSources["loader_recipe"],WorkflowExamples.trainAddress)
        XCTAssertEqual(expected.datasetRecipe?.configuration,historical.datasetRecipe?.configuration)
        XCTAssertEqual(expected.loaderRecipe?.configuration,historical.loaderRecipe?.configuration)
        XCTAssertEqual(expected.settings,historical.settings)
        XCTAssertEqual(expected.model,historical.model)
        XCTAssertEqual(expected.featureColumns,historical.featureColumns)
        XCTAssertFalse(document.nodes.contains { $0.address == dataset || $0.address == loader })
        XCTAssertNil(training.graph?.execution)
        XCTAssertEqual(plan.models.count,1)
    }
    func testWholeWorkflowCoversEveryTrainAndRefusesInvalidSecondTerminal()throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let second=WorkflowNodeAddress(domain:.control,localID:"second")
        document.nodes.append(.init(address:second,name:"Second training",payload:.train(config:[:])))
        document.edges += document.edges.filter {$0.to.address==WorkflowExamples.trainAddress}.map {.init(from:$0.from,to:.init(address:second,port:$0.to.port),kind:$0.kind)}
        XCTAssertEqual(try compile(document).trainings.map(\.id),["train","second"])
        document.edges.removeAll {$0.to.address==second && $0.to.port=="model"}
        XCTAssertThrowsError(try compile(document))
        let selected=try compile(document,scope:.training,targets:[WorkflowExamples.trainAddress])
        XCTAssertEqual(selected.trainings.map(\.id),["train"])
    }
    func testPreparationScopeRetainsAllDataBranchesInMixedCanvas()throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        document.nodes.append(.init(address:.init(domain:.data,localID:"otherSource"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(";")])))
        let plan=try compile(document,scope:.preparation)
        XCTAssertEqual(Set(plan.preparation!.graph.nodes.map(\.id)),Set(document.nodes.filter {$0.address.domain == .data}.map {$0.address.localID}))
        XCTAssertTrue(plan.trainings.isEmpty);XCTAssertTrue(plan.models.isEmpty)
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(document,registry:registry()))
    }
    func testAbsentScopeAndUnknownSelectionRefuseInsteadOfFirstFallback()throws {
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:.init(domain:.data,localID:"source"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")]))])
        XCTAssertThrowsError(try compile(document,scope:.model))
        XCTAssertThrowsError(try compile(document,scope:.training))
        XCTAssertThrowsError(try compile(document,scope:.workflow,targets:[.init(domain:.control,localID:"missing")]))
    }
    func testBackendTerminalsUseTheirExistingSharedCompiler()throws {
        for document in [try WorkflowExamples.preparedSklearnClassifier(),try WorkflowExamples.preparedTensorFlowClassifier()] {
            let plan=try compile(document),training=try XCTUnwrap(plan.trainings.first)
            XCTAssertEqual(plan.trainings.count,1)
            let terminal=try XCTUnwrap(document.nodes.first {if case .estimator=$0.payload {return true};return false})
            let expected=try WorkflowEstimatorCompiler.compile(document,estimator:terminal.address,modelRegistry:.builtIn(),dataRegistry:registry())
            XCTAssertEqual(training.backendID,expected.configuration.backendID)
            XCTAssertEqual(training.configuration,expected.configuration.settings)
            XCTAssertEqual(training.prepared.featureColumns,expected.configuration.featureColumns)
            XCTAssertEqual(training.prepared.validationNode,expected.validation.node)
            XCTAssertEqual(training.trainingMode,.fresh)
            XCTAssertNil(training.settings)
            try plan.validate()
        }
    }
    func testObjectiveProjectionPreservesTargetMappingAndScalarization()throws {
        let document=try WorkflowExamples.preparedMSERegression(device:"cpu")
        let training=try XCTUnwrap(compile(document).trainings.first)
        let expected=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry(),contractVersion:12)
        XCTAssertEqual(training.recipes["objective_recipe"],expected.lossRecipe?.configuration.recipeValue)
        XCTAssertEqual(training.recipes["objective_scalarization"],.string(expected.objectiveScalarization.rawValue))
        XCTAssertNotNil(training.recipes["target_mapping"])
        XCTAssertNil(training.prepared.targetColumn)
        XCTAssertEqual(training.prepared.featureColumns,expected.lossRecipe?.targetMapping.inputColumns)
    }
    func testContentIdentityIgnoresGeometryAndRevisionButIncludesNamesAndSettings()throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let original=try compile(document)
        document.revision += 1
        for index in document.nodes.indices {document.nodes[index].x += 17;document.nodes[index].y -= 9}
        for definition in document.definitions.indices {
            for node in document.definitions[definition].body.nodes.indices {document.definitions[definition].body.nodes[node].x += 41}
        }
        let moved=try compile(document)
        XCTAssertNotEqual(try original.encoded(),try moved.encoded())
        XCTAssertEqual(try original.contentIdentity,try moved.contentIdentity)
        document.nodes[0].name="Updated source label"
        XCTAssertNotEqual(try original.contentIdentity,try compile(document).contentIdentity)
        let setup=try XCTUnwrap(document.nodes.firstIndex {if case .runSetup=$0.payload {return true};return false})
        if case .runSetup(var values)=document.nodes[setup].payload {values["epochs"] = .integer(4);document.nodes[setup].payload = .runSetup(config:values)}
        XCTAssertNotEqual(try moved.contentIdentity,try compile(document).contentIdentity)
    }
}
