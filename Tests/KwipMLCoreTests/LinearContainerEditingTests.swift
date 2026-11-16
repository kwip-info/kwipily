import XCTest
@testable import KwipMLCore

final class LinearContainerEditingTests:XCTestCase {
    func testWorkflowAndInlineEditsRetainPortsIdentityBindingsAndPolicies()throws {
        let registry=try ComponentRegistry.builtIn()
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let address=WorkflowNodeAddress(domain:.model,localID:"projection")
        let policy:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        var config=try LinearConfiguration.defaults(for:.explicit,registry:registry);config["out_features"]=6;config["bias"]=false
        let node=WorkflowNode(address:address,name:"Projection",x:10,y:20,payload:.modelPrimitive(type:"linear",configurationVersion:2,config:config),parameterSettings:policy)
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[node])
        document.schemaVersion=15
        let changed=try WorkflowEditing.apply(.setLinear(address:address,variant:.inferred),to:document,modelRegistry:registry,dataRegistry:data).document
        XCTAssertEqual(changed.nodes[0].address,address);XCTAssertEqual(changed.nodes[0].parameterSettings,policy)
        let next=try LinearConfiguration.configuration(of:changed.nodes[0],registry:registry)
        XCTAssertEqual(next.values["out_features"],6);XCTAssertEqual(next.values["bias"],false)
        XCTAssertEqual(node.payload,document.nodes[0].payload)
        let container=WorkflowNodeAddress(domain:.model,localID:"container")
        let field=try XCTUnwrap(registry.component(type:"linear")?.fields.first {$0.key=="out_features"})
        let definition=CompositeDefinition(reference:.init("linear-fixture"),title:"Projection",body:.init(nodes:[.init(id:"projection",type:"linear",config:config,parameterSettings:policy)]),inputs:[.init("in",targets:[.init("projection","in")])],outputs:[.init("out",source:.init("projection","out"))],parameters:[.init(field,targets:[.init("projection","out_features")])],composition:.init(kind:.graph,children:[.init(localID:"projection")]))
        var inline=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:container,payload:.modelBlock(reference:definition.reference,overrides:["out_features":12]))],definitions:[definition])
        inline.schemaVersion=15
        let proxy=try WorkflowInlineContainer.node(in:inline,root:container,childID:"projection",registry:registry)
        guard case .modelPrimitive(_,_,var values)=proxy.payload else {return XCTFail("Expected resolved primitive")}
        XCTAssertEqual(values["out_features"],12);values["in_features"]=4
        let edited=try WorkflowInlineContainer.apply(.setLinear(childID:"projection",variant:.explicit,configuration:values),to:inline,root:container,registry:registry,idProvider:{"private-projection"})
        let retained=try WorkflowInlineContainer.definition(in:edited,root:container)
        XCTAssertEqual(retained.body.nodes[0].config["out_features"],6,"Bound raw value remains owned by the definition")
        XCTAssertEqual(retained.body.nodes[0].config["in_features"],4)
        XCTAssertEqual(retained.body.nodes[0].parameterSettings,policy)
        XCTAssertEqual(retained.inputs,definition.inputs);XCTAssertEqual(retained.outputs,definition.outputs)
        XCTAssertEqual(edited.definitions[0],definition)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setLinear(childID:"projection",variant:.inferred),to:inline,root:container,registry:registry,idProvider:{"refused"}))
    }
}
