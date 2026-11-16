import XCTest
@testable import KwipMLCore

final class WorkflowTrainDataLoadingTests:XCTestCase {
    private var registry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    private var dataset:[String:JSONValue] {["kind":.string("subset"),"indices":.array([4,1,3,0]),"max_rows":100000]}
    private var batching:[String:JSONValue] {var values=DataLoaderConfiguration.defaults;values["batch_size"]=3;values["shuffle"]=true;values["seed"]=99;return values}
    private func document(_ config:[String:JSONValue])throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedParallelClassifier(device:"cpu");doc.schemaVersion=13
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        doc.nodes[index].payload = .train(config:config)
        return doc
    }
    private func compile(_ doc:WorkflowDocument)throws->WorkflowExecutionPlan {try WorkflowCompiler.compile(doc,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:12)}
    func testInlineSettingsNormalizeToExistingVersionedRecipesAtTrainOrigin()throws {
        let doc=try document(["datasetAccess":.object(dataset),"batching":.object(batching)]),plan=try compile(doc)
        XCTAssertEqual(plan.datasetRecipe?.configuration.values,dataset)
        XCTAssertEqual(plan.loaderRecipe?.configuration.values,try DataLoaderConfiguration(values:batching).values)
        XCTAssertEqual(plan.datasetRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(plan.loaderRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(plan.settings.batchSize,3)
        XCTAssertEqual(plan.inlineDataLoadingRoles,["dataset","loader"])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
        let projection=try WorkflowCodePlan.compile(doc,scope:.training,selectedTargets:[WorkflowExamples.trainAddress],modelRegistry:.builtIn(),dataRegistry:registry)
        let training=try XCTUnwrap(projection.trainings.first)
        XCTAssertEqual(training.recipes["dataset_recipe"],plan.datasetRecipe?.configuration.recipeValue)
        XCTAssertEqual(training.recipeSources["dataset_recipe"],WorkflowExamples.trainAddress)
        XCTAssertEqual(training.recipeSources["loader_recipe"],WorkflowExamples.trainAddress)
    }
    func testOmittedSettingsRetainLegacyNilAndRunSetupBatchSize()throws {
        let plan=try compile(document([:]))
        XCTAssertNil(plan.datasetRecipe);XCTAssertNil(plan.loaderRecipe)
        XCTAssertTrue(plan.inlineDataLoadingRoles.isEmpty)
        XCTAssertEqual(plan.settings.batchSize,32)
    }
    func testIndividualInlineRolesAndLegacyConnectedOtherRole()throws {
        let datasetOnly=try compile(document(["datasetAccess":.object(dataset)]))
        XCTAssertNotNil(datasetOnly.datasetRecipe);XCTAssertNil(datasetOnly.loaderRecipe)
        XCTAssertEqual(datasetOnly.settings.batchSize,32)
        let batchingOnly=try compile(document(["batching":.object(batching)]))
        XCTAssertNil(batchingOnly.datasetRecipe);XCTAssertNotNil(batchingOnly.loaderRecipe)
        var doc=try document(["datasetAccess":.object(dataset)])
        let loader=WorkflowNodeAddress(domain:.control,localID:"legacyLoader")
        doc.nodes.append(.init(address:loader,payload:.batchRecipe(type:DataLoaderConfiguration.componentID,configurationVersion:1,config:batching)))
        doc.edges.append(.init(from:.init(address:loader,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"loader"),kind:.batchRecipe))
        XCTAssertThrowsError(try compile(doc))
        let mixed=try WorkflowCompiler.compileHistoricalEvidence(doc,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:12)
        XCTAssertEqual(mixed.inlineDataLoadingRoles,["dataset"])
        XCTAssertEqual(mixed.datasetRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(mixed.loaderRecipe?.source,loader)
    }
    func testConnectedAndInlineSameRoleIsExplicitConflict()throws {
        for (role,kind) in [("dataset",WorkflowEdgeKind.datasetRecipe),("loader",.batchRecipe)] {
            let key=role=="dataset" ? "datasetAccess":"batching"
            var doc=try document([key:.object(role=="dataset" ? dataset:batching)])
            let source=WorkflowNodeAddress(domain:.control,localID:"legacy")
            let payload:WorkflowNodePayload=role=="dataset" ? .datasetRecipe(type:DatasetConfiguration.componentID,configurationVersion:1,config:dataset):.batchRecipe(type:DataLoaderConfiguration.componentID,configurationVersion:1,config:batching)
            doc.nodes.append(.init(address:source,payload:payload));doc.edges.append(.init(from:.init(address:source,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:role),kind:kind))
            XCTAssertThrowsError(try RetiredTrainingRecipeMigration.migrate(doc)) {error in
                XCTAssertTrue(error.localizedDescription.contains("both inline and connected"),error.localizedDescription)
            }
        }
    }
    func testMalformedInlineValuesAndUnchangedResourceBoundsAreRejected()throws {
        let bad:[String:JSONValue]=["datasetAccess":.null]
        XCTAssertThrowsError(try WorkflowCodec.validate(document(bad)))
        XCTAssertThrowsError(try WorkflowCodec.validate(document(["batching":.object([:])])))
        var oversized=DatasetConfiguration.defaults;oversized["max_rows"]=100001
        XCTAssertThrowsError(try WorkflowCodec.validate(document(["datasetAccess":.object(oversized)])))
        var wrong=DataLoaderConfiguration.defaults;wrong["num_workers"]=1
        XCTAssertThrowsError(try WorkflowCodec.validate(document(["batching":.object(wrong)])))
        var unknown=DatasetConfiguration.defaults;unknown["path"] = .string("/private/source")
        XCTAssertThrowsError(try WorkflowCodec.validate(document(["datasetAccess":.object(unknown)])))
        var old=try document(["datasetAccess":.object(dataset)]);old.schemaVersion=12
        XCTAssertThrowsError(try WorkflowCodec.validate(old))
    }
    func testBoundedStreamShuffleCompatibilityStillAppliesToInlineControls()throws {
        let stream:[String:JSONValue]=["kind":.string("boundedIterable"),"indices":.array([]),"max_rows":20]
        XCTAssertThrowsError(try compile(document(["datasetAccess":.object(stream),"batching":.object(batching)])))
        var ordered=batching;ordered["shuffle"] = .bool(false)
        XCTAssertEqual(try compile(document(["datasetAccess":.object(stream),"batching":.object(ordered)])).datasetRecipe?.configuration.maximumRows,20)
    }

    func testEditingAddsVersion13AtomicallyWithoutDroppingOtherTrainSettings()throws {
        var old=try document(["trainingMode":.string("fineTune")]);old.schemaVersion=12
        let config:[String:JSONValue]=["trainingMode":.string("fineTune"),"datasetAccess":.object(dataset),"batching":.object(batching)]
        let edited=try WorkflowEditing.apply(.configure(address:WorkflowExamples.trainAddress,config:config),to:old,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(edited.document.schemaVersion,13)
        let node=try XCTUnwrap(edited.document.nodes.first(where:{$0.address==WorkflowExamples.trainAddress}))
        guard case .train(let actual)=node.payload else {return XCTFail("Train missing")}
        XCTAssertEqual(actual,config)
        XCTAssertEqual(old.schemaVersion,12,"Editing returns a new reversible document.")
        XCTAssertEqual(try compile(edited.document).trainingMode,.fineTune)
    }
    func testEveryExistingDatasetKindAndBatchingConfigurationNormalizesWithoutNewEnvelope()throws {
        for kind in ["indexed","tensor","boundedIterable","subset"] {
            var values=DatasetConfiguration.defaults;values["kind"] = .string(kind)
            if kind=="subset" {values["indices"] = .array([3,0])}
            if kind=="boundedIterable" {values["max_rows"] = 20}
            let plan=try compile(document(["datasetAccess":.object(values)]))
            XCTAssertEqual(plan.datasetRecipe?.configuration.recipeValue,try DatasetConfiguration(values:values).recipeValue)
        }
        var values=DataLoaderConfiguration.defaults
        values["batch_size"]=7;values["shuffle"] = .bool(true);values["drop_last"] = .bool(true)
        values["seed"]=123;values["num_workers"]=2;values["prefetch_factor"]=2;values["timeout"] = .number(10);values["collation"] = .string("fixed_stack")
        let plan=try compile(document(["batching":.object(values)]))
        XCTAssertEqual(plan.loaderRecipe?.configuration.values,try DataLoaderConfiguration(values:values).values)
        XCTAssertEqual(plan.settings.batchSize,7)
    }

}
