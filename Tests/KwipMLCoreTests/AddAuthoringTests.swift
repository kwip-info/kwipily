import XCTest
@testable import KwipMLCore
final class AddAuthoringTests:XCTestCase {
    func testDefaultScaledAndBroadcastShapesPreserveOriginalContract()throws {
        XCTAssertNotNil(TensorOperationAuthoring.family("add"))
        XCTAssertEqual(try TensorOperationAuthoring.inputPorts("add",[:]),["a","b"])
        let a=try BatchedTensorLayout(sample:[4,8]),same=try BatchedTensorLayout(sample:[4,8]),small=try BatchedTensorLayout(sample:[8])
        XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"add",values:[:],inputs:["a":a,"b":same])["out"],a)
        XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"add",values:["alpha":-0.5,"sample_broadcast":true],inputs:["a":a,"b":small])["out"],a)
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"add",values:["alpha":-0.5],inputs:["a":a,"b":small]))
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"add",values:["sample_broadcast":true],inputs:["a":a,"b":try .init(sample:[3,8])]))
        XCTAssertThrowsError(try TensorOperationAuthoring.canonical("add",["alpha":.number(.infinity)]))
    }
    func testExistingSavedConfigurationAndWiresAreNotReplaced()throws {
        let registry=try ComponentRegistry.builtIn()
        let original=GraphNode(id:"sum",type:"add",name:"Residual",config:[:])
        let a=try BatchedTensorLayout(sample:[4,8])
        let retained=try TensorOperationAuthoring.switching(original,configurationVersion:2,to:"add",configuration:["alpha":1,"sample_broadcast":false],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:["a":a,"b":a],registry:registry)
        XCTAssertEqual(retained,original,"Opening/default reapplication must not materialize omitted values")
        let next=try TensorOperationAuthoring.switching(original,configurationVersion:2,to:"add",configuration:["alpha":-0.5,"sample_broadcast":false],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:["a":a,"b":a],registry:registry)
        XCTAssertEqual(next.id,original.id);XCTAssertEqual(next.name,original.name);XCTAssertEqual(next.config["alpha"],-0.5)
    }
    func testKnownIntegerInputsRefuseScalingAndBroadcastingAtCommit()throws {
        let registry=try ComponentRegistry.builtIn(),a=try TypedTensorContract(sample:[4,8],dtype:.int64)
        let original=GraphNode(id:"sum",type:"add",config:[:])
        for values:[String:JSONValue] in [["alpha":-0.5],["sample_broadcast":true]] {
            XCTAssertThrowsError(try TensorOperationAuthoring.switching(original,configurationVersion:2,to:"add",configuration:values,connectedInputs:["a","b"],connectedOutputs:["out"],inputs:["a":a.layout,"b":a.layout],registry:registry,inputContracts:["a":a,"b":a]))
        }
        XCTAssertNoThrow(try TensorOperationAuthoring.validateKnownTypes(type:"add",values:[:],inputs:["a":a,"b":a]))
        let different=try TypedTensorContract(sample:[4,8],dtype:.float32)
        XCTAssertThrowsError(try TensorOperationAuthoring.validateKnownTypes(type:"add",values:[:],inputs:["a":a,"b":different]))
    }
    func testRawContainerBindingsRetainOverridesAndRefuseDirectReplacement()throws {
        let registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        let field=try XCTUnwrap(registry.component(type:"add")?.fields.first {$0.key=="alpha"})
        let definition=CompositeDefinition(reference:.init("add-fixture"),title:"Residual",body:.init(nodes:[.init(id:"sum",type:"add",config:["alpha":1])]),inputs:[.init("a",targets:[.init("sum","a")]),.init("b",targets:[.init("sum","b")])],outputs:[.init("out",source:.init("sum","out"))],parameters:[.init(field,targets:[.init("sum","alpha")])],composition:.init(kind:.graph,children:[.init(localID:"sum")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:["alpha":-0.5]))],definitions:[definition]);document.schemaVersion=15
        let edited=try WorkflowInlineContainer.apply(.setTensorOperation(childID:"sum",type:"add",configuration:["alpha":-0.5,"sample_broadcast":true]),to:document,root:root,registry:registry,idProvider:{"private-add"})
        let retained=try WorkflowInlineContainer.definition(in:edited,root:root)
        XCTAssertEqual(retained.body.nodes[0].config["alpha"],1);XCTAssertEqual(edited.definitions[0],definition)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setTensorOperation(childID:"sum",type:"add",configuration:["alpha":2,"sample_broadcast":false]),to:document,root:root,registry:registry,idProvider:{"refused"}))
    }

    func testActualGraphCommandRechecksIntegerSourceContracts()throws {
        let registry=try ComponentRegistry.builtIn()
        let graph=GraphDocument(schemaVersion:2,nodes:[.init(id:"a",type:"index_input",config:["shape":.array([4,8])]),.init(id:"b",type:"index_input",config:["shape":.array([4,8])]),.init(id:"sum",type:"add",config:[:])],edges:[.init(source:"a",target:"sum",targetPort:"a"),.init(source:"b",target:"sum",targetPort:"b")])
        let state=GraphEditorState(document:graph)
        for config:[String:JSONValue] in [["alpha":-0.5],["sample_broadcast":true]] {
            XCTAssertThrowsError(try GraphCommandEngine.apply(.setTensorOperation(nodeID:"sum",type:"add",configuration:config),to:state,registry:registry,makeID:{"unused"}))
        }
        XCTAssertEqual(graph.nodes[2].config,[:])
    }

}
