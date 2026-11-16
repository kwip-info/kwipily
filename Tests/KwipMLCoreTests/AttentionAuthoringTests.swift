import XCTest
@testable import KwipMLCore
final class AttentionAuthoringTests:XCTestCase {
    func testRemovingConnectedMasksOrWeightsRefusesWithoutChangingSource()throws {
        let registry=try ComponentRegistry.builtIn()
        let node=GraphNode(id:"a",type:"multihead_attention",config:["attention_mask_mode":.string("supplied")])
        let next=try AttentionConfiguration(componentID:node.type,values:[:]).values
        XCTAssertThrowsError(try AttentionConfiguration.applying(next,to:node,connectedInputPorts:["attn_mask"],registry:registry))
        var quiet=next;quiet["return_weights"]=false
        XCTAssertThrowsError(try AttentionConfiguration.applying(quiet,to:node,connectedOutputPorts:["weights"],registry:registry))
        XCTAssertEqual(node.config,["attention_mask_mode":.string("supplied")])
        XCTAssertEqual(try AttentionConfiguration.applying(quiet,to:node,registry:registry).config["return_weights"],false)
    }
    func testRemovingAuthoredProjectionPolicyNeedsExplicitRepair()throws {
        let registry=try ComponentRegistry.builtIn()
        let node=GraphNode(id:"a",type:"multihead_attention",config:[:],parameterSettings:["in_proj_weight":.object(["trainability":.string("frozen")])])
        var values=try AttentionConfiguration(componentID:node.type,values:[:]).values;values["kdim"]=3
        XCTAssertThrowsError(try AttentionConfiguration.applying(values,to:node,registry:registry))
        values["kdim"] = .null;values["dropout"] = .number(0.2)
        XCTAssertEqual(try AttentionConfiguration.applying(values,to:node,registry:registry).parameterSettings,node.parameterSettings)
    }
    func testAllSixCanonicalGraphEditsKeepIdentityAndUndoSource()throws {
        let registry=try ComponentRegistry.builtIn()
        for type in AttentionConfiguration.componentIDs.sorted() {
            let node=GraphNode(id:"original",type:type,name:"Mine",config:[:])
            let state=GraphEditorState(document:.init(schemaVersion:2,nodes:[node],edges:[]))
            var values=try AttentionConfiguration(componentID:type,values:[:]).values
            if type=="rotary_position" {values["start_position"]=2} else if type=="attention_mask" {values["padding_keys"] = .array([1])} else {values["dropout"] = .number(0.2)}
            let result=try GraphCommandEngine.apply(.setAttention(nodeID:node.id,configuration:values),to:state,registry:registry)
            XCTAssertEqual(result.before,state);XCTAssertEqual(result.after.document.nodes.first?.id,node.id)
            XCTAssertEqual(result.after.document.nodes.first?.name,"Mine")
            XCTAssertEqual(result.after.document.nodes.first?.config,values)
        }
    }
    func testInlineExposedMaskWeightsAndBoundHeadCountAreProtected()throws {
        let registry=try ComponentRegistry.builtIn()
        let config=try AttentionConfiguration(componentID:"multihead_attention",values:["attention_mask_mode":.string("supplied")]).values
        let field=try XCTUnwrap(registry.component(type:"multihead_attention")?.fields.first {$0.key=="num_heads"})
        let definition=CompositeDefinition(reference:.init("attention-container"),title:"Attention",body:.init(nodes:[.init(id:"attention",type:"multihead_attention",config:config)],configurationVersion:2),inputs:["query","key","value","attn_mask"].map {.init($0,targets:[.init("attention",$0)])},outputs:[.init("out",source:.init("attention","out")),.init("weights",source:.init("attention","weights"))],parameters:[.init(field,targets:[.init("attention","num_heads")])],composition:.init(kind:.graph,children:[.init(localID:"attention")]))
        let root=WorkflowNodeAddress(domain:.model,localID:"group")
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:[:]))],definitions:[definition]);document.schemaVersion=11
        for (key,value) in [("attention_mask_mode",JSONValue.string("none")),("return_weights",.bool(false)),("num_heads",.integer(1))] {
            var next=config;next[key]=value
            XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setAttention(childID:"attention",configuration:next),to:document,root:root,registry:registry),key)
        }
        var next=config;next["dropout"] = .number(0.1)
        let result=try WorkflowInlineContainer.apply(.setAttention(childID:"attention",configuration:next),to:document,root:root,registry:registry)
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:result,root:root).body.nodes.first?.config["dropout"],.number(0.1))
        XCTAssertTrue(result.definitions.contains(definition))
    }

}
