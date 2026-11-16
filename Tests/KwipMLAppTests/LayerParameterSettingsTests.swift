import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LayerParameterSettingsTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func fixture()throws->(WorkflowDocument,ComponentRegistry,DataFlowOperatorRegistry) {
        (try WorkflowExamples.preparedParallelClassifier(device:"cpu"),try .builtIn(),try .load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testLayerSettingsCompileToExistingRecipesWithoutChangingExecutableModel()throws {
        var (document,registry,data)=try fixture()
        let baseline=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:12)
        let index=try XCTUnwrap(document.nodes.firstIndex {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false})
        document=try WorkflowEditing.apply(.setLayerParameters(address:document.nodes[index].address,settings:["weight":.object(["initialization":.object(["algorithm":.string("zeros"),"settings":.object([:])]),"trainability":.string("frozen")])]),to:document,modelRegistry:registry,dataRegistry:data).document
        let restored=try WorkflowCodec.decode(WorkflowCodec.encode(document))
        XCTAssertEqual(restored,document)
        let plan=try WorkflowCompiler.compile(restored,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:13)
        XCTAssertEqual(plan.model.graph,baseline.model.graph)
        XCTAssertEqual(plan.model.semanticSHA256,baseline.model.semanticSHA256)
        XCTAssertEqual(plan.model.origins,baseline.model.origins)
        let rule=try XCTUnwrap(plan.initializationRecipe?.configuration.rules.first)
        XCTAssertEqual(rule.owner.node,document.nodes[index].address.localID)
        XCTAssertEqual(rule.algorithm,"zeros")
        XCTAssertEqual(plan.initializationRecipe?.configuration.seed,Int64(plan.settings.seed))
        XCTAssertEqual(plan.trainabilityRecipe?.configuration.owners.first,.object(["node":.string(rule.owner.node),"parameter":.string("weight"),"mode":.string("frozen")]))
        XCTAssertThrowsError(try WorkflowCompiler.compile(restored,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:12))
    }
    func testLegacyOverridesRetainExactSeedAndSelectedParameterBehavior()throws {
        var (document,registry,data)=try fixture()
        document=try WorkflowTrainRecipes.migrateAuthoringSchema(document)
        let index=try XCTUnwrap(document.nodes.firstIndex {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false})
        let id=document.nodes[index].address.localID
        let legacy=try InitializationConfiguration(values:["seed":97,"rules":.array([.object(["owner":.object(["node":.string(id),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0.3,"std":0.02])])])])
        let train=try XCTUnwrap(document.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        document.nodes[train].payload = .train(config:["initialization":legacy.recipeValue])
        let before=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:12)
        document.nodes[index].parameterSettings=["weight":.object(["initialization":.object(["algorithm":.string("zeros"),"settings":.object([:])])])]
        document.schemaVersion=15
        let after=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:13)
        XCTAssertEqual(after.initializationRecipe,before.initializationRecipe)
        XCTAssertEqual(after.model,before.model)
    }
    func testParameterSettingsRejectMalformedPoliciesAndNonLayerOwnership()throws {
        var (document,_,_)=try fixture()
        document.schemaVersion=15
        document.nodes[0].parameterSettings=["weight":.object(["trainability":.string("frozen")])]
        XCTAssertThrowsError(try WorkflowCodec.encode(document))
    }
    func testLayerEditorApplyUndoAndStaleCallbacksPreserveGeometry()throws {
        let(document,registry,data)=try fixture()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:data)
        let address=try XCTUnwrap(document.nodes.first {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false}).address
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        let editor=LayerParametersInspectorContext(context:context)
        XCTAssertEqual(editor.choices.map(\.id),["weight","bias"])
        XCTAssertTrue(editor.setAlgorithm("zeros",parameter:"weight"))
        XCTAssertTrue(editor.setMode("frozen",parameter:"bias"))
        XCTAssertFalse(editor.setAlgorithm("xavier_uniform",parameter:"bias"))
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        let geometry=context.configuration
        XCTAssertTrue(editor.apply())
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(context.configuration,geometry)
        XCTAssertEqual(owner.document.schemaVersion,15)
        let reopened=try library.loadSource(id:document.projectID)
        guard case .workflow(let restored)=reopened else {return XCTFail("Missing saved workflow")}
        XCTAssertEqual(restored.nodes.first {$0.address==address}?.parameterSettings,editor.draft)
        XCTAssertTrue(editor.setMode("frozen",parameter:"weight"))
        editor.revert();XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(context.isCurrent)
        XCTAssertFalse(editor.setMode("trainable",parameter:"weight"))
        XCTAssertNil(owner.document.nodes.first {$0.address==address}?.parameterSettings)
    }
    func testNestedBlockSettingsRoundTripAndExpandToExactOwners()throws {
        var(document,registry,data)=try fixture()
        let d=try XCTUnwrap(document.definitions.indices.first)
        let n=try XCTUnwrap(document.definitions[d].body.nodes.firstIndex {$0.type=="linear"})
        document=try WorkflowTrainRecipes.migrateAuthoringSchema(document)
        document.schemaVersion=15
        document.definitions[d].body.nodes[n].parameterSettings=["bias":.object(["trainability":.string("frozen")])]
        let restored=try WorkflowCodec.decode(WorkflowCodec.encode(document))
        let plan=try WorkflowCompiler.compile(restored,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:13)
        let row=try XCTUnwrap(plan.trainabilityRecipe?.configuration.owners.first)
        guard case .object(let flag)=row else {return XCTFail("Missing flag")}
        let id=try XCTUnwrap(flag["node"]?.stringValue)
        XCTAssertEqual(plan.model.origins[id]?.localNodeID,document.definitions[d].body.nodes[n].id)
        XCTAssertFalse(plan.model.origins[id]?.instancePath.isEmpty ?? true)
        XCTAssertEqual(flag["parameter"],.string("bias"))
        XCTAssertTrue(plan.model.graph.nodes.allSatisfy {$0.parameterSettings == nil})
    }

    func testCompilerRejectsMissingParameterAndBiasFanInitializer()throws {
        var(document,registry,data)=try fixture()
        document=try WorkflowTrainRecipes.migrateAuthoringSchema(document);document.schemaVersion=15
        let index=try XCTUnwrap(document.nodes.firstIndex {if case .modelPrimitive("linear",_,_)=$0.payload {return true};return false})
        for settings:[String:JSONValue] in [
            ["no_such_weight":.object(["trainability":.string("inherit")])],
            ["bias":.object(["initialization":.object(["algorithm":.string("xavier_uniform"),"settings":.object(["gain":1])])])]
        ] {
            document.nodes[index].parameterSettings=settings
            XCTAssertThrowsError(try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:13))
        }
    }
    func testLayerParameterInventoryCoversConfiguredNativeFamiliesAndDisabledBias()throws {
        let registry=try ComponentRegistry.builtIn()
        let linear=try LayerParameterInventory.choices(type:"linear",config:["in_features":3,"out_features":2,"bias":false],version:2,registry:registry)
        XCTAssertEqual(linear.map(\.id),["weight"]);XCTAssertEqual(linear.first?.shape,[2,3])
        let relu=try LayerParameterInventory.choices(type:"relu",config:[:],version:2,registry:registry)
        XCTAssertTrue(relu.isEmpty)
        for type in ["linear","lazy_linear","bilinear","prelu","embedding","embedding_bag","layer_norm","batch_norm1d","batch_norm2d","instance_norm1d","instance_norm2d","group_norm","conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d","depthwise_conv2d","rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell","multihead_attention","transformer","transformer_encoder","transformer_decoder"] {
            let definition=try XCTUnwrap(registry.component(type:type),type)
            let config=Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)})
            let choices=try LayerParameterInventory.choices(type:type,config:config,version:2,registry:registry)
            XCTAssertEqual(Set(choices.map(\.id)).count,choices.count,type)
            // Native defaults can disable affine parameters; inventory must still parse every family.
        }
    }

}
