import XCTest
@testable import KwipMLCore
final class ReshapeAuthoringTests:XCTestCase {
    func testSwitchingPreservesIdentityWiresAndOriginalSource()throws {
        let registry=try ComponentRegistry.builtIn()
        let node=GraphNode(id:"shape",type:"flatten",name:"Features",x:20,y:30,config:["start_dim":1,"end_dim":-1])
        let next=try ReshapeAuthoringConfiguration.switching(node,configurationVersion:2,to:.dimensions,configuration:["shape":.array([2,3])],registry:registry)
        XCTAssertEqual(next.id,node.id);XCTAssertEqual(next.name,node.name);XCTAssertEqual(next.x,20)
        XCTAssertEqual(next.type,"reshape");XCTAssertEqual(node.type,"flatten")
        let state=GraphEditorState(document:.init(schemaVersion:2,nodes:[node],edges:[]))
        let result=try GraphCommandEngine.apply(.setReshape(nodeID:node.id,variant:.dimensions,configuration:["shape":.array([-1])]),to:state,registry:registry)
        XCTAssertEqual(result.before,state);XCTAssertEqual(result.after.document.nodes.first?.id,node.id)
    }
    func testPlannedShapeIsBatchSafeAndRejectsWrongCount()throws {
        let input=try BatchedTensorLayout(sample:[2,3,4])
        XCTAssertEqual(try ReshapeAuthoringConfiguration.outputLayout(variant:.dimensions,configuration:["shape":.array([3,-1])],input:input).sample.dimensions,[3,8])
        XCTAssertEqual(try ReshapeAuthoringConfiguration.outputLayout(variant:.flatten,configuration:["start_dim":1,"end_dim":2],input:input).sample.dimensions,[6,4])
        XCTAssertThrowsError(try ReshapeAuthoringConfiguration.outputLayout(variant:.dimensions,configuration:["shape":.array([5,5])],input:input))
        for (first,last) in [(0,2),(-4,-1),(3,1)] {
            XCTAssertThrowsError(try ReshapeAuthoringConfiguration.outputLayout(variant:.flatten,configuration:["start_dim":.integer(Int64(first)),"end_dim":.integer(Int64(last))],input:input))
        }
    }
    func testKnownIncomingShapeRefusesMismatchAndKeepsWires()throws {
        let registry=try ComponentRegistry.builtIn()
        let graph=GraphDocument(schemaVersion:2,nodes:[.init(id:"input",type:"input",config:["features":6]),.init(id:"shape",type:"reshape",config:["shape":.array([2,3])]),.init(id:"output",type:"output")],edges:[.init(source:"input",target:"shape"),.init(source:"shape",target:"output")])
        let state=GraphEditorState(document:graph)
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setReshape(nodeID:"shape",variant:.dimensions,configuration:["shape":.array([4,4])]),to:state,registry:registry))
        let result=try GraphCommandEngine.apply(.setReshape(nodeID:"shape",variant:.flatten,configuration:["start_dim":-1,"end_dim":-1]),to:state,registry:registry)
        XCTAssertEqual(result.after.document.edges,graph.edges);XCTAssertEqual(result.before,state)
        XCTAssertEqual(result.after.document.nodes.first {$0.id=="shape"}?.config,["start_dim":-1,"end_dim":-1])
    }
    func testInlineBindingsRetainRawValuesAndRefuseMethodReplacement()throws {
        let registry=try ComponentRegistry.builtIn()
        let root=WorkflowNodeAddress(domain:.model,localID:"container")
        let field=try XCTUnwrap(registry.component(type:"flatten")?.fields.first {$0.key=="start_dim"})
        let config:[String:JSONValue]=["start_dim":1,"end_dim":-1]
        let definition=CompositeDefinition(reference:.init("flatten-fixture"),title:"Features",body:.init(nodes:[.init(id:"shape",type:"flatten",config:config)]),inputs:[.init("in",targets:[.init("shape","in")])],outputs:[.init("out",source:.init("shape","out"))],parameters:[.init(field,targets:[.init("shape","start_dim")])],composition:.init(kind:.graph,children:[.init(localID:"shape")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:["start_dim":2]))],definitions:[definition]);document.schemaVersion=15
        let edited=try WorkflowInlineContainer.apply(.setReshape(childID:"shape",variant:.flatten,configuration:["start_dim":2,"end_dim":3]),to:document,root:root,registry:registry,idProvider:{"private"})
        let retained=try WorkflowInlineContainer.definition(in:edited,root:root)
        XCTAssertEqual(retained.body.nodes[0].config,["start_dim":1,"end_dim":3]);XCTAssertEqual(retained.inputs,definition.inputs);XCTAssertEqual(retained.outputs,definition.outputs)
        XCTAssertEqual(edited.definitions[0],definition)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setReshape(childID:"shape",variant:.dimensions),to:document,root:root,registry:registry,idProvider:{"refused"}))
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setReshape(childID:"shape",variant:.flatten,configuration:["start_dim":1,"end_dim":3]),to:document,root:root,registry:registry,idProvider:{"refused"}))
    }

}
