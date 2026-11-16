import XCTest
@testable import KwipMLCore

final class WorkflowTrainedArtifactBoundaryTests:XCTestCase {
    private var registry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    func testSchema17TrainPublishesTrainedModelButRetained16DoesNot()throws {
        var document=try WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        let train=try XCTUnwrap(document.nodes.first {$0.address==WorkflowExamples.trainAddress})
        XCTAssertEqual(try WorkflowPortContracts.resolve(train,document:document,modelRegistry:.builtIn(),dataRegistry:registry).outputs.map(\.id),["predictions"])
        document.schemaVersion=17
        let ports=try WorkflowPortContracts.resolve(train,document:document,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(ports.outputs.map(\.id),["model","predictions"])
        XCTAssertEqual(ports.outputs.map(\.kind),[.frozenModel,.table])
    }
    func testEvaluateTrainModelIsArtifactBoundaryAndDoesNotRequireEditableTrainingAncestors()throws {
        var document=try WorkflowExamples.preparedGroupSplitClassifier(device:"cpu")
        document.nodes.removeAll {$0.address.domain != .data}
        let retained=Set(document.nodes.map(\.address));document.edges.removeAll {!retained.contains($0.from.address) || !retained.contains($0.to.address)}
        document.schemaVersion=17
        let evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        document.nodes += [.init(address:WorkflowExamples.trainAddress,payload:.train(config:["runSetup":.object([:])])),.init(address:evaluate,payload:.evaluateTest(config:[:]))]
        document.edges += [.init(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:.init(domain:.data,localID:"split"),port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        let plan=try WorkflowEvaluationCompiler.compile(document,evaluate:evaluate,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(plan.frozenModel,WorkflowExamples.trainAddress)
        XCTAssertEqual(plan.testLineage.role,"test")
        XCTAssertEqual(plan.canonicalTestEndpoint,.init(node:"split",port:"test"))
    }
    func testSaveModelConfigurationAndPlanAreStrictArtifactOnlyContracts()throws {
        XCTAssertEqual(try WorkflowSaveModelConfiguration.decode([:]).format,.package)
        XCTAssertEqual(try WorkflowSaveModelConfiguration.decode(["format":.string("checkpoint")]).format,.checkpoint)
        for bad:[String:JSONValue] in [["format":.null],["format":.string("onnx")],["path":.string("/private/source")]] {XCTAssertThrowsError(try WorkflowSaveModelConfiguration.decode(bad))}
        var document=WorkflowDocument(projectID:UUID().uuidString)
        document.schemaVersion=17
        let train=WorkflowExamples.trainAddress,save=WorkflowNodeAddress(domain:.control,localID:"save")
        document.nodes=[.init(address:train,payload:.train(config:["runSetup":.object([:])])),.init(address:save,payload:.saveModel(config:["format":.string("checkpoint")]))]
        document.edges=[.init(from:.init(address:train,port:"model"),to:.init(address:save,port:"model"),kind:.frozenModel)]
        let plan=try WorkflowSaveModelPlan.compile(document,save:save)
        XCTAssertEqual(plan.modelSource,.train(.init(address:train,port:"model")))
        XCTAssertEqual(plan.configuration.format,.checkpoint)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        XCTAssertThrowsError(try WorkflowCodePlan.compile(document,scope:.workflow,selectedTargets:[],modelRegistry:.builtIn(),dataRegistry:registry))
        var malformed=document;malformed.edges[0].from.port="data"
        XCTAssertThrowsError(try WorkflowSaveModelPlan.compile(malformed,save:save))
        malformed=document;malformed.edges[0].kind = .modelDefinition
        XCTAssertThrowsError(try WorkflowSaveModelPlan.compile(malformed,save:save))
        malformed=document;malformed.edges.append(document.edges[0]);malformed.edges[1].id=UUID().uuidString
        XCTAssertThrowsError(try WorkflowSaveModelPlan.compile(malformed,save:save))
        var saved=document
        saved.nodes[0].payload = .frozenModel(config:[:])
        XCTAssertEqual(try WorkflowSaveModelPlan.compile(saved,save:save).modelSource,.savedCheckpoint(.init(address:train,port:"model")))
    }
    func testSavePlacementAndTrainConnectionPromoteOnlyExplicitAuthoringAction()throws {
        let legacy=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),original=try WorkflowCodec.encode(legacy)
        let save=WorkflowNodeAddress(domain:.control,localID:"save")
        let inserted=try WorkflowEditing.apply(.insert(.init(address:save,payload:.saveModel(config:[:]))),to:legacy,modelRegistry:.builtIn(),dataRegistry:registry).document
        XCTAssertEqual(inserted.schemaVersion,18)
        XCTAssertFalse(inserted.nodes.contains(where:WorkflowTrainRunSetup.isLegacyNode))
        let connected=try WorkflowEditing.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:save,port:"model")),to:inserted,modelRegistry:.builtIn(),dataRegistry:registry).document
        XCTAssertTrue(try WorkflowSaveModelPlan.compile(connected,save:save).modelSource.isTrain)
        XCTAssertEqual(try WorkflowCodec.encode(legacy),original)
        let migrated16=try WorkflowTrainRunSetup.migrate(legacy)
        _ = try WorkflowCompiler.compile(migrated16,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:14)
        XCTAssertThrowsError(try WorkflowCompiler.compile(connected,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:14))
        _ = try WorkflowCompiler.compile(connected,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:15)
    }
    func testSchema16AuthoringShowsModelPortAndConnectionExplicitlyPromotesSource()throws {
        var document=try WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        let train=try XCTUnwrap(document.nodes.first {$0.address==WorkflowExamples.trainAddress})
        let authoring=try WorkflowPortContracts.resolveForAuthoring(train,document:document,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(authoring.outputs.map(\.id),["model","predictions"])
        XCTAssertEqual(try WorkflowPortContracts.resolve(train,document:document,modelRegistry:.builtIn(),dataRegistry:registry).outputs.map(\.id),["predictions"])
        XCTAssertTrue(try WorkflowPortContracts.resolvePlanPorts(train,document:document,modelRegistry:.builtIn(),dataRegistry:registry,historicalEvidence:true).outputs.isEmpty)
        let visible=try WorkflowCanvasProjector.project(document,modelRegistry:.builtIn(),dataRegistry:registry,inlineTrainControls:true)
        XCTAssertEqual(visible.portDescriptor(address:train.address,port:"model",direction:.output)?.semanticKind,.frozenModel)
        let retained=try WorkflowCanvasProjector.project(document,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertNil(retained.portDescriptor(address:train.address,port:"model",direction:.output))
        let evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        document.nodes.append(.init(address:evaluate,payload:.evaluateTest(config:[:])))
        let original=try WorkflowCodec.encode(document)
        let connected=try WorkflowEditing.apply(.connect(from:.init(address:train.address,port:"model"),to:.init(address:evaluate,port:"model")),to:document,modelRegistry:.builtIn(),dataRegistry:registry).document
        XCTAssertEqual(connected.schemaVersion,18)
        XCTAssertEqual(try WorkflowTrainedModelSource.resolve(in:connected,consumer:evaluate),.train(.init(address:train.address,port:"model")))
        XCTAssertEqual(try WorkflowCodec.encode(document),original)
    }
}
