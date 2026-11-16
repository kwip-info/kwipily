import XCTest
@testable import KwipMLCore
final class AttentionGraphIntegrationTests:XCTestCase {
    func testConditionalMaskAndNamedWeightsPorts()throws {
        let definition=BundledAttentionComponents.definition("multihead_attention")
        XCTAssertEqual(definition.configuredInputs([:]),["query","key","value"])
        XCTAssertEqual(definition.configuredInputs(["attention_mask_mode":.string("supplied"),"padding_mask_mode":.string("supplied")]),["query","key","value","attn_mask","key_padding_mask"])
        XCTAssertEqual(definition.configuredOutputs(["return_weights":false]),["out"])
        XCTAssertEqual(definition.configuredOutputs([:]),["out","weights"])
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:.init(id:"op",type:"transformer_encoder"),schemaVersion:1,registry:ComponentRegistry.builtIn()))
    }
    func testRealMaskGeometryAndTypedProductionGraph()throws {
        let registry=try ComponentRegistry.builtIn()
        let nodes:[GraphNode]=[
            .init(id:"input",type:"input",config:["features":2]),
            .init(id:"sequence",type:"reshape",config:["shape":.array([1,2])]),
            .init(id:"mask",type:"attention_mask"),
            .init(id:"op",type:"multihead_attention",config:["embed_dim":2,"num_heads":1,"attention_mask_mode":.string("supplied"),"padding_mask_mode":.string("supplied")]),
            .init(id:"flat",type:"flatten",config:["start_dim":1,"end_dim":-1]),
            .init(id:"output",type:"output")]
        let pairs:[(String,String,String,String)]=[("input","out","sequence","in"),("sequence","out","mask","query"),("sequence","out","mask","key"),("sequence","out","op","query"),("sequence","out","op","key"),("sequence","out","op","value"),("mask","mask","op","attn_mask"),("mask","padding","op","key_padding_mask"),("op","out","flat","in"),("flat","out","output","in")]
        let edges=pairs.enumerated().map {i,p in GraphEdge(id:"e\(i)",source:p.0,sourcePort:p.1,target:p.2,targetPort:p.3)}
        let graph=GraphDocument(name:"Real masks",nodes:nodes,edges:edges)
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.issues.isEmpty,"\(result.issues)")
        XCTAssertEqual(result.outputTensorContracts[.init("mask","mask")]?.dtype,.int64)
        XCTAssertEqual(result.outputTensorContracts[.init("mask","padding")]?.layout.sample.dimensions,[1])
        XCTAssertEqual(result.outputTensorContracts[.init("op","weights")]?.layout.sample.dimensions,[1,1])
        var wrong=graph;wrong.edges[6].source="sequence";wrong.edges[6].sourcePort="out"
        XCTAssertFalse(GraphValidator.validate(wrong,registry:registry).issues.isEmpty)
        var disabled=graph;disabled.nodes[3].config["return_weights"]=false
        XCTAssertNil(GraphValidator.validate(disabled,registry:registry).outputTensorContracts[.init("op","weights")])
    }
}
