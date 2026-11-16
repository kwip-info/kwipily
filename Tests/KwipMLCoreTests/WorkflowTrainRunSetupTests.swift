import XCTest
@testable import KwipMLCore

final class WorkflowTrainRunSetupTests:XCTestCase {
    private var registry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    private func source()throws->WorkflowDocument {try WorkflowExamples.preparedParallelClassifier(device:"cpu")}
    private func compile(_ value:WorkflowDocument,version:Int=14)throws->WorkflowExecutionPlan {
        try WorkflowCompiler.compile(value,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:version)
    }
    func testExactUpgradePreservesSettingsModelDataAndOriginalBytes()throws {
        var original=try source();original.schemaVersion=max(original.schemaVersion,4)
        let setup=try XCTUnwrap(original.nodes.firstIndex {WorkflowTrainRunSetup.isLegacyNode($0)})
        guard case .runSetup(var values)=original.nodes[setup].payload else {return XCTFail("Missing legacy setup")}
        values["preparationCachePolicy"] = .string("materialize")
        values["experiment"] = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8])])]),"budget":.object(["maximumRuns":2,"maximumTotalEpochs":100])])
        values["gradientCapture"] = .object(["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":2,"maxCaptures":128])
        original.nodes[setup].payload = .runSetup(config:values)
        let bytes=try WorkflowCodec.encode(original),before=try compile(original,version:12)
        let upgraded=try WorkflowTrainRunSetup.migrate(original),after=try compile(upgraded)
        XCTAssertEqual(upgraded.schemaVersion,16)
        XCTAssertFalse(upgraded.nodes.contains(where:WorkflowTrainRunSetup.isLegacyNode))
        XCTAssertFalse(upgraded.edges.contains {$0.kind == .runConfiguration})
        XCTAssertEqual(try WorkflowTrainRunSetup.resolve(in:upgraded,train:WorkflowExamples.trainAddress).configuration,values)
        XCTAssertEqual(after.settings,before.settings)
        XCTAssertEqual(after.model.graph,before.model.graph)
        XCTAssertEqual(after.model.semanticSHA256,before.model.semanticSHA256)
        XCTAssertEqual(after.dataflow,before.dataflow)
        XCTAssertEqual(after.preparationCachePolicy,before.preparationCachePolicy)
        XCTAssertEqual(try WorkflowCodec.encode(original),bytes)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(upgraded)),upgraded)
        XCTAssertEqual(try WorkflowTrainRunSetup.migrate(upgraded),upgraded)
        XCTAssertThrowsError(try compile(upgraded,version:13))
        let code=try WorkflowCodePlan.compile(upgraded,scope:.training,selectedTargets:[WorkflowExamples.trainAddress],modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(code.trainings.count,1)
    }
    func testSharedSetupIsCopiedToEveryTrainWithoutChangingOtherTrainFields()throws {
        var original=try source();original.schemaVersion=max(3,original.schemaVersion)
        let second=WorkflowNodeAddress(domain:.control,localID:"secondTrain")
        original.nodes.append(.init(address:second,payload:.train(config:["trainingMode":.string("fresh") ])))
        original.edges += original.edges.filter {$0.to.address==WorkflowExamples.trainAddress}.map {
            .init(id:$0.id+"-second",from:$0.from,to:.init(address:second,port:$0.to.port),kind:$0.kind)
        }
        let before=try WorkflowTrainRunSetup.resolve(in:original,train:second).configuration
        let upgraded=try WorkflowTrainRunSetup.migrate(original)
        XCTAssertEqual(try WorkflowTrainRunSetup.resolve(in:upgraded,train:second).configuration,before)
        XCTAssertEqual(try WorkflowTrainRunSetup.resolve(in:upgraded,train:WorkflowExamples.trainAddress).configuration,before)
        guard case .train(let config)=try XCTUnwrap(upgraded.nodes.first {$0.address==second}).payload else {return XCTFail("Missing Train")}
        XCTAssertEqual(config["trainingMode"],.string("fresh"))
        XCTAssertEqual(upgraded.nodes.count,original.nodes.count-1)
    }
    func testMissingUnusedMultipleOrUnsupportedAuthoritiesRefuseWithoutMutation()throws {
        let baseline=try source()
        var missing=baseline;missing.edges.removeAll {$0.kind == .runConfiguration}
        var orphan=baseline
        orphan.nodes.append(.init(address:.init(domain:.control,localID:"unusedSetup"),payload:.runSetup(config:["seed":71])))
        var multiple=baseline
        let existing=try XCTUnwrap(multiple.edges.first {$0.kind == .runConfiguration})
        multiple.nodes.append(.init(address:.init(domain:.control,localID:"otherSetup"),payload:.runSetup(config:["seed":72])))
        multiple.edges.append(.init(from:.init(address:.init(domain:.control,localID:"otherSetup"),port:"configuration"),to:existing.to,kind:.runConfiguration))
        var unsupported=baseline
        let index=try XCTUnwrap(unsupported.edges.firstIndex {$0.kind == .runConfiguration})
        unsupported.edges[index] = .init(from:existing.from,to:.init(address:existing.to.address,port:"data"),kind:.runConfiguration)
        for candidate in [missing,orphan,multiple,unsupported] {
            let encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys]
            let bytes=try encoder.encode(candidate)
            XCTAssertThrowsError(try WorkflowTrainRunSetup.migrate(candidate))
            XCTAssertEqual(try encoder.encode(candidate),bytes)
        }
    }
    func testCurrentSchemaCannotSilentlyIgnoreStandaloneRunSettings()throws {
        let legacy=try source()
        var current=try WorkflowTrainRunSetup.migrate(legacy)
        current.nodes.append(.init(address:.init(domain:.control,localID:"unusedSettings"),payload:.runSetup(config:["seed":71])))
        let bytes=try WorkflowCodec.encode(current)
        XCTAssertThrowsError(try compile(current)) {error in
            XCTAssertTrue(error.localizedDescription.contains("standalone Run setup"))
        }
        XCTAssertEqual(try WorkflowCodec.encode(current),bytes)
        XCTAssertThrowsError(try WorkflowCodePlan.compile(current,scope:.training,selectedTargets:[WorkflowExamples.trainAddress],modelRegistry:.builtIn(),dataRegistry:registry))
        let ordinaryLegacy=try compile(legacy,version:12)
        let historicalLegacy=try WorkflowCompiler.compileHistoricalEvidence(legacy,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:12)
        XCTAssertEqual(ordinaryLegacy,historicalLegacy)
    }
    func testInlineShapeIsStrictAndSettingPreservesUnrelatedConfiguration()throws {
        let other:[String:JSONValue]=["trainingMode":.string("fresh"),"batching":.object(DataLoaderConfiguration.defaults)]
        let settings:[String:JSONValue]=["seed":19,"learningRate":.number(0.025)]
        let config=try WorkflowTrainRunSetup.setting(settings,in:other)
        XCTAssertEqual(config["batching"],other["batching"])
        XCTAssertEqual(config["trainingMode"],other["trainingMode"])
        XCTAssertEqual(try WorkflowTrainRunSetup.configuration(config,schemaVersion:16),settings)
        XCTAssertThrowsError(try WorkflowTrainRunSetup.configuration(config,schemaVersion:15))
        XCTAssertThrowsError(try WorkflowTrainRunSetup.configuration([:],schemaVersion:16))
        for bad:JSONValue in [.null,.array([]),.object(["unexpected":1]),.object(["gradientCapture":.object(["unexpected":1])]),.object(["epochs":0])] {
            XCTAssertThrowsError(try WorkflowTrainRunSetup.configuration(["runSetup":bad],schemaVersion:16))
        }
        XCTAssertEqual(try WorkflowTrainRunSetup.configuration(["runSetup":.object([:])],schemaVersion:16),[:])
    }
    func testExperimentBatchAxisUsesEffectiveInlineBatchingWithoutChangingFallback()throws {
        var source=try WorkflowTrainRunSetup.migrate(source())
        let index=try XCTUnwrap(source.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        guard case .train(var config)=source.nodes[index].payload else {return XCTFail("Missing Train")}
        var batching=DataLoaderConfiguration.defaults;batching["batch_size"]=7
        config["batching"] = .object(batching)
        source.nodes[index].payload = .train(config:config)
        let bytes=try WorkflowCodec.encode(source)
        let plan=try WorkflowExperimentPlan.build(source,train:WorkflowExamples.trainAddress,axes:[.init(field:"batchSize",values:[16,32])],budget:.init(maximumRuns:2,maximumTotalEpochs:100),modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(plan.variants.map {$0.execution.settings.batchSize},[16,32])
        XCTAssertEqual(plan.variants.map {$0.execution.loaderRecipe?.configuration.batchSize},[16,32])
        for (variant,size) in zip(plan.variants,[16,32]) {
            guard case .train(let next)=try XCTUnwrap(variant.document.nodes.first {$0.address==WorkflowExamples.trainAddress}).payload else {return XCTFail("Missing variant Train")}
            var expected=config,expectedBatching=batching;expectedBatching["batch_size"] = .integer(Int64(size))
            expected["batching"] = .object(expectedBatching)
            XCTAssertEqual(next,expected,"Only the effective batch size changes; fallback settings and all unrelated controls stay exact.")
        }
        XCTAssertEqual(try WorkflowCodec.encode(source),bytes)
    }
    func testInlineExperimentsChangeOnlyTrainSettingsAndRetainModelData()throws {
        let upgraded=try WorkflowTrainRunSetup.migrate(source()),bytes=try WorkflowCodec.encode(upgraded)
        let plan=try WorkflowExperimentPlan.build(upgraded,train:WorkflowExamples.trainAddress,axes:[.init(field:"seed",values:[7,8]),.init(field:"batchSize",values:[16,32])],budget:.init(maximumRuns:4,maximumTotalEpochs:100),modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(plan.variants.map {[$0.execution.settings.seed,$0.execution.settings.batchSize]},[[7,16],[7,32],[8,16],[8,32]])
        XCTAssertTrue(plan.variants.allSatisfy {$0.execution.model.semanticSHA256==plan.variants[0].execution.model.semanticSHA256})
        XCTAssertTrue(plan.variants.allSatisfy {$0.execution.dataflow==plan.variants[0].execution.dataflow})
        XCTAssertEqual(try WorkflowCodec.encode(upgraded),bytes)
    }
}
