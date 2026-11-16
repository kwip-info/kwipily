import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AttentionInspectorDraftTests:XCTestCase {
    func testOptionalWidthsScaleAndInvalidNumericDrafts()throws {
        var draft=try AttentionInspectorDraft(componentID:"multihead_attention",values:[:])
        draft.optionalEnabled["kdim"]=true;draft.setText("3",key:"kdim")
        XCTAssertEqual(try draft.configuration()["kdim"],3)
        draft.setText("3.5",key:"kdim");XCTAssertThrowsError(try draft.configuration())
        draft.optionalEnabled["kdim"]=false;XCTAssertEqual(try draft.configuration()["kdim"],.null)
        draft.setText("3",key:"num_heads");XCTAssertThrowsError(try draft.configuration())
        var sdpa=try AttentionInspectorDraft(componentID:"scaled_dot_product_attention",values:[:])
        sdpa.optionalEnabled["scale"]=true;sdpa.setText("1e-2",key:"scale")
        XCTAssertEqual(try sdpa.configuration()["scale"],.number(0.01))
        for value in ["0","nan","-1","1e-100"] {sdpa.setText(value,key:"scale");XCTAssertThrowsError(try sdpa.configuration())}
    }
    func testMaskRowsPreservePairedCoordinatesAndRejectDuplicates()throws {
        var draft=try AttentionInspectorDraft(componentID:"attention_mask",values:["mode":.string("blocked"),"blocked_queries":.array([0,1]),"blocked_keys":.array([2,3]),"padding_keys":.array([4])])
        XCTAssertEqual(draft.blockedPairs.count,2)
        draft.blockedPairs[1].key="2";draft.blockedPairs[1].query="0"
        XCTAssertThrowsError(try draft.configuration())
        draft.blockedPairs.remove(at:1);draft.paddingPositions[0].key="-1"
        XCTAssertThrowsError(try draft.configuration())
        draft.paddingPositions[0].key="4"
        XCTAssertEqual(try draft.configuration()["blocked_keys"],.array([2]))
        draft.values["mode"] = .string("causal")
        XCTAssertThrowsError(try draft.configuration(),"Changing mode must not silently drop authored pairs")
    }
    func testEncoderDraftRetainsDefaultsAndRejectsInvalidHeads()throws {
        var draft=try AttentionInspectorDraft(componentID:"transformer_encoder",values:[:])
        XCTAssertEqual(try draft.configuration()["activation"],.string("relu"))
        draft.values["final_norm"]=true;draft.values["norm_first"]=true
        draft.setText("2",key:"num_layers");draft.setText("1e-4",key:"layer_norm_eps")
        XCTAssertEqual(try draft.configuration()["num_layers"],2)
        draft.setText("3",key:"nhead");XCTAssertThrowsError(try draft.configuration())
    }
    func testAttentionParametersHaveHumanLayerNames() {
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("in_proj_weight"),"Query/key/value projection weights")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("decoder.layers.1.multihead_attn.out_proj.bias"),"Decoder layer 2 · Cross-attention · Output projection biases")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("layers.0.norm1.weight"),"Layer 1 · Normalization 1 · Weights")
    }

    func testDecoderMasksRetainIndependentRolesAndRejectMemoryCausal()throws {
        var draft=try AttentionInspectorDraft(componentID:"transformer_decoder",values:[:])
        draft.values["target_mask_mode"] = .string("causal")
        draft.values["memory_mask_mode"] = .string("supplied")
        draft.values["target_padding_mode"] = .string("supplied")
        draft.values["memory_padding_mode"] = .string("supplied")
        let c=try AttentionConfiguration(componentID:draft.componentID,values:draft.configuration())
        XCTAssertEqual(c.configuredInputs,["target","memory","memory_mask","tgt_key_padding_mask","memory_key_padding_mask"])
        draft.values["memory_mask_mode"] = .string("causal");XCTAssertThrowsError(try draft.configuration())
    }

    func testCombinedTransformerRetainsSourceTargetMasksAndFixedFinalNorm()throws {
        var draft=try AttentionInspectorDraft(componentID:"transformer",values:[:])
        draft.setText("2",key:"num_encoder_layers");draft.setText("3",key:"num_decoder_layers")
        draft.values["source_mask_mode"] = .string("supplied")
        draft.values["target_mask_mode"] = .string("causal")
        draft.values["memory_mask_mode"] = .string("supplied")
        let c=try AttentionConfiguration(componentID:draft.componentID,values:draft.configuration())
        XCTAssertEqual(c.configuredInputs,["source","target","src_mask","memory_mask"])
        XCTAssertTrue(c.parameterShapes.contains {$0.name=="encoder.norm.weight"})
        XCTAssertTrue(c.parameterShapes.contains {$0.name=="decoder.layers.2.multihead_attn.in_proj_weight"})
        XCTAssertNil(c.values["final_norm"])
        draft.values["final_norm"]=false;XCTAssertThrowsError(try draft.configuration())
    }

    @MainActor func testWorkflowEditUndoStaleSelectionAndOriginalSourceAcrossAllTools()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for type in AttentionConfiguration.componentIDs.sorted() {
            let address=WorkflowNodeAddress(domain:.model,localID:"layer")
            let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing",payload:.modelPrimitive(type:type,configurationVersion:1,config:[:]))])
            _=try library.saveSource(.workflow(original))
            let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
            defer {owner.invalidate()}
            XCTAssertEqual(owner.document,original)
            XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
            let context=try WorkflowModelNodeContext(owner:owner,address:address)
            var next=try AttentionConfiguration(componentID:type,values:[:]).values
            if type=="rotary_position" {next["start_position"]=2} else if type=="attention_mask" {next["padding_keys"] = .array([1])} else {next["dropout"] = .number(0.2)}
            XCTAssertTrue(context.setAttention(configuration:next,expected:[:]),context.statusMessage)
            XCTAssertFalse(context.setAttention(configuration:next,expected:[:]))
            XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,original.nodes)
            XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
            XCTAssertFalse(context.setAttention(configuration:next,expected:[:]))
            XCTAssertEqual(owner.document.nodes,original.nodes)
            let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
            let entry=try XCTUnwrap(canvas.creationEntries.first {$0.iconType==type})
            XCTAssertEqual(canvas.creationEntries.filter {$0.iconType==type}.count,1)
            XCTAssertEqual(entry.component.category,"Layers")
            XCTAssertNotNil(ComponentLessonDestination.entryID(type:type))
            canvas.invalidate()
        }
    }

}
