import XCTest
@testable import KwipMLCore
final class TransformerBlockTests:XCTestCase {
    func testCompositionRetainsAllConfiguredMaskBindingsAndFinalNorms()throws {
        let registry=try ComponentRegistry.builtIn()
        let values:[String:JSONValue]=["d_model":4,"nhead":2,"num_encoder_layers":2,"num_decoder_layers":3,"dim_feedforward":7,"dropout":0,"norm_first":true,"source_mask_mode":.string("supplied"),"source_padding_mode":.string("supplied"),"target_mask_mode":.string("causal"),"memory_mask_mode":.string("supplied"),"target_padding_mode":.string("supplied"),"memory_padding_mode":.string("supplied")]
        let definition=try TransformerBlock.definition(values:values)
        try CompositeDefinitionValidation.validate(definition,definitions:[definition],registry:registry)
        XCTAssertEqual(definition.body.nodes.map(\.type),["transformer_encoder","transformer_decoder"])
        XCTAssertEqual(definition.body.nodes.map {$0.config["final_norm"]},[true,true])
        XCTAssertEqual(definition.body.nodes[0].config["num_layers"],2)
        XCTAssertEqual(definition.body.nodes[1].config["num_layers"],3)
        XCTAssertEqual(Set(definition.inputs.map(\.id)),["source","target","src_mask","src_key_padding_mask","memory_mask","tgt_key_padding_mask","memory_key_padding_mask"])
        XCTAssertEqual(definition.body.edges.count,1)
        XCTAssertEqual(definition.body.edges[0].targetPort,"memory")
        XCTAssertEqual(definition.composition?.kind,.graph)
        XCTAssertTrue(definition.parameters.isEmpty,"Members remain freely editable rather than parent-bound")
    }
    func testFreshExampleIsEditableWithoutReinterpretingLegacyPrimitive()throws {
        let old=try WorkflowExamples.preparedTransformerClassifier(device:"cpu")
        let document=try WorkflowExamples.preparedEditableTransformerClassifier(device:"cpu")
        XCTAssertTrue(old.nodes.contains {if case .modelPrimitive("transformer",_,_)=$0.payload {return true};return false})
        let node=try XCTUnwrap(document.nodes.first {if case .modelBlock(let pin,_)=$0.payload {return pin==TransformerBlock.reference};return false})
        XCTAssertEqual(document.edges,old.edges)
        let registry=try ComponentRegistry.builtIn()
        var edit=try WorkflowInlineContainer.definition(in:document,root:node.address).body.nodes[0].config;edit["dim_feedforward"]=5
        let changed=try WorkflowInlineContainer.apply(.setAttention(childID:"encoder",configuration:edit),to:document,root:node.address,registry:registry,idProvider:{"edited-transformer"})
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:changed,root:node.address).body.nodes[0].config["dim_feedforward"],5)
        XCTAssertEqual(document.definitions.first?.body.nodes[0].config["dim_feedforward"],3)
    }
    func testCapturedSourceRoundTripAndIndependentExpansionRemainExact()throws {
        let registry=try ComponentRegistry.builtIn()
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try WorkflowExamples.preparedEditableTransformerClassifier(device:"cpu")
        let train=try XCTUnwrap(document.nodes.first {if case .train=$0.payload {return true};return false})
        let plan=try WorkflowCompiler.compile(document,train:train.address,modelRegistry:registry,dataRegistry:data,contractVersion:15)
        let source=try XCTUnwrap(plan.model.provenance)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let bytes=try encoder.encode(source)
        let decoded=try JSONDecoder().decode(CompositeSourceSnapshot.self,from:bytes)
        XCTAssertEqual(source,decoded)
        XCTAssertEqual(try encoder.encode(decoded),bytes)
        XCTAssertNoThrow(try decoded.verifiedExpansion(registry:registry,context:.preparedWorkflow))
        var changed=decoded
        changed.sourceCompositeDocument.definitions[0].body.nodes[0].config["dropout"] = .number(0.25)
        XCTAssertNotEqual(changed,source)
        XCTAssertThrowsError(try changed.verifiedExpansion(registry:registry,context:.preparedWorkflow))
    }
}
