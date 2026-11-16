import XCTest
@testable import KwipMLCore

final class InitializationWorkflowCompilerTests:XCTestCase {
    private let recipeAddress=WorkflowNodeAddress(domain:.control,localID:"starting-settings")
    private func registries()throws->(ComponentRegistry,DataFlowOperatorRegistry) {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try (.builtIn(),.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func compile(_ document:WorkflowDocument,version:Int=8)throws->WorkflowExecutionPlan {
        let (model,data)=try registries()
        return try WorkflowCompiler.compileHistoricalEvidence(document,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:version)
    }
    private func controls(owner:String,parameter:String="weight",seed:Int64=42)->[String:JSONValue] {
        ["seed":.integer(seed),"rules":.array([.object(["owner":.object(["node":.string(owner),"parameter":.string(parameter)]),"algorithm":.string("zeros"),"settings":.object([:])])])]
    }
    private func fixture()throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let plan=try compile(document,version:2)
        let owner=try XCTUnwrap(plan.model.graph.nodes.first {$0.type == "linear"})
        document.schemaVersion=9
        document.nodes.append(.init(address:recipeAddress,name:"Starting settings",payload:.initializationRecipe(type:"initialization",configurationVersion:1,config:controls(owner:owner.id))))
        document.edges.append(.init(id:"starting-wire",from:.init(address:recipeAddress,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"initialization"),kind:.initializationRecipe))
        return document
    }
    func testQualifiedInitializationCompilesWithImplicitAdamAndPreservesExpandedModel()throws {
        let old=try WorkflowExamples.preparedParallelClassifier(device:"cpu"),baseline=try compile(old,version:2)
        let document=try fixture(),plan=try compile(document)
        let recipe=try XCTUnwrap(plan.initializationRecipe)
        XCTAssertEqual(recipe.source,recipeAddress)
        XCTAssertEqual(recipe.configuration.seed,42)
        XCTAssertNil(plan.optimizerRecipe)
        XCTAssertEqual(plan.settings.learningRate,baseline.settings.learningRate)
        XCTAssertEqual(plan.model.graph,baseline.model.graph)
        let selected=try XCTUnwrap(recipe.configuration.rules.first)
        XCTAssertNotNil(plan.model.origins[selected.owner.node])
        XCTAssertFalse(plan.model.graph.nodes.contains {$0.type == "initialization"})
        XCTAssertNil(try compile(old,version:2).initializationRecipe)
    }
    func testSchemaCompilerAndCanonicalCodecBoundaries()throws {
        let document=try fixture()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        for version in [2,3,4,5,6,7,17] {XCTAssertThrowsError(try compile(document,version:version))}
        let supportedBaseline=try compile(document,version:8)
        for supportedVersion in [11,12,13,14,15,16] {
            let supported=try compile(document,version:supportedVersion)
            XCTAssertEqual(supported.initializationRecipe,supportedBaseline.initializationRecipe)
            XCTAssertEqual(supported.model.graph,supportedBaseline.model.graph)
        }
        var lower=document;lower.schemaVersion=8
        XCTAssertThrowsError(try WorkflowCodec.encode(lower))
        var future=document;future.schemaVersion=18
        XCTAssertThrowsError(try WorkflowCodec.encode(future))
        let old=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let migrated=try WorkflowCodec.migrateToVersion9(old)
        XCTAssertEqual(migrated.schemaVersion,9)
        XCTAssertEqual(migrated.nodes,old.nodes);XCTAssertEqual(migrated.edges,old.edges)
        XCTAssertNil(try compile(migrated).initializationRecipe)
    }
    func testInvalidSelectedNodeDuplicateWireAndWrongPortKindRefuse()throws {
        let source=try fixture()
        for invalid in ["not-in-model","observations","model"] {
            var document=source
            document.nodes[document.nodes.count-1].payload = .initializationRecipe(type:"initialization",configurationVersion:1,config:controls(owner:invalid))
            XCTAssertThrowsError(try compile(document),invalid)
        }
        var duplicate=source
        duplicate.edges.append(.init(id:"second-wire",from:.init(address:recipeAddress,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"initialization"),kind:.initializationRecipe))
        XCTAssertThrowsError(try compile(duplicate))
        var wrong=source;wrong.edges[wrong.edges.count-1].kind = .optimizerRecipe
        XCTAssertThrowsError(try compile(wrong))
        wrong=source;wrong.edges[wrong.edges.count-1].to.port="optimizer"
        XCTAssertThrowsError(try compile(wrong))
        var disconnected=source;disconnected.edges.removeLast()
        XCTAssertNil(try compile(disconnected).initializationRecipe)
    }
    func testCoreDefersNativeParameterMembershipButChecksCanonicalRuleIdentity()throws {
        var document=try fixture()
        let owner=try XCTUnwrap(try compile(document).initializationRecipe?.configuration.rules.first?.owner.node)
        document.nodes[document.nodes.count-1].payload = .initializationRecipe(type:"initialization",configurationVersion:1,config:controls(owner:owner,parameter:"nested.actual_runtime_parameter"))
        XCTAssertEqual(try compile(document).initializationRecipe?.configuration.rules.first?.owner.parameter,"nested.actual_runtime_parameter")
        var bad=controls(owner:owner)
        if case .array(let rules)=bad["rules"] {bad["rules"] = .array(rules+rules)}
        document.nodes[document.nodes.count-1].payload = .initializationRecipe(type:"initialization",configurationVersion:1,config:bad)
        XCTAssertThrowsError(try compile(document))
    }
    func testRetiredInitializationHasNoCurrentPortsOrEditableRoute()throws {
        let (model,data)=try registries(),source=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let node=try XCTUnwrap(try fixture().nodes.last)
        XCTAssertThrowsError(try WorkflowEditing.apply(.insert(node),to:source,modelRegistry:model,dataRegistry:data))
        let historical=try fixture()
        XCTAssertTrue(try WorkflowPortContracts.resolve(node,document:historical,modelRegistry:model,dataRegistry:data).outputs.isEmpty)
        let descriptor=CanvasPortDescriptor(kind:.initializationRecipe,direction:.output)
        XCTAssertEqual(descriptor.glyph,.sliders);XCTAssertEqual(descriptor.shortLabel,"Starting settings recipe")
        XCTAssertThrowsError(try WorkflowEditing.apply(.configure(address:recipeAddress,config:controls(owner:"anything",seed:19)),to:historical,modelRegistry:model,dataRegistry:data))
    }
    func testExperimentVariantsPreserveSelectedInitializationRecipe()throws {
        let document=try RetiredTrainingRecipeMigration.migrate(fixture()),baseline=try compile(document,version:13),(model,data)=try registries()
        let variants=try WorkflowExperimentPlan.build(document,train:WorkflowExamples.trainAddress,axes:[.init(field:"learningRate",values:[0.01,0.05])],budget:.init(maximumRuns:2,maximumTotalEpochs:6),modelRegistry:model,dataRegistry:data)
        XCTAssertEqual(variants.variants.count,2)
        XCTAssertEqual(variants.variants.map {$0.execution.initializationRecipe},[baseline.initializationRecipe,baseline.initializationRecipe])
        XCTAssertEqual(document,variants.source)
    }
}
