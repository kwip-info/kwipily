import XCTest
@testable import KwipMLCore

final class BilinearContainerEditingTests:XCTestCase {
    func testInlineDraftRespectsBoundWidthAndRetainsOriginalDefinitionAndPolicies()throws {
        let registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        let config:[String:JSONValue]=["in1_features":3,"in2_features":5,"out_features":2,"bias":true]
        let field=try XCTUnwrap(registry.component(type:"bilinear")?.fields.first {$0.key=="in1_features"})
        let policy:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        let definition=CompositeDefinition(reference:.init("bilinear-fixture"),title:"Interactions",body:.init(nodes:[.init(id:"mix",type:"bilinear",config:config,parameterSettings:policy)]),inputs:[.init("a",targets:[.init("mix","a")]),.init("b",targets:[.init("mix","b")])],outputs:[.init("out",source:.init("mix","out"))],parameters:[.init(field,targets:[.init("mix","in1_features")])],composition:.init(kind:.graph,children:[.init(localID:"mix")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:["in1_features":4]))],definitions:[definition]);document.schemaVersion=15
        let proxy=try WorkflowInlineContainer.node(in:document,root:root,childID:"mix",registry:registry)
        guard case .modelPrimitive(_,_,var values)=proxy.payload else {return XCTFail("Expected actual Bilinear")}
        XCTAssertEqual(values["in1_features"],4);values["in2_features"]=6
        let changed=try WorkflowInlineContainer.apply(.setBilinear(childID:"mix",configuration:values),to:document,root:root,registry:registry,idProvider:{"private-interactions"})
        let stored=try WorkflowInlineContainer.definition(in:changed,root:root)
        XCTAssertEqual(stored.body.nodes[0].config["in1_features"],3)
        XCTAssertEqual(stored.body.nodes[0].config["in2_features"],6)
        XCTAssertEqual(stored.body.nodes[0].parameterSettings,policy)
        XCTAssertEqual(stored.inputs,definition.inputs);XCTAssertEqual(stored.outputs,definition.outputs)
        XCTAssertEqual(changed.definitions[0],definition)
        values["in1_features"]=7
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setBilinear(childID:"mix",configuration:values),to:document,root:root,registry:registry,idProvider:{"refused"}))
    }
}
