import XCTest
@testable import KwipMLCore
final class TensorOperationAuthoringTests:XCTestCase {
    func testOriginalDefaultsAndIndependentDeclaredGeometry()throws {
        let registry=try ComponentRegistry.builtIn()
        for type in TensorOperationAuthoring.componentIDs {XCTAssertNoThrow(try TensorOperationAuthoring.defaults(type,registry:registry))}
        let input=try BatchedTensorLayout(sample:[2,3,4])
        XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"transpose",values:["dim0":1,"dim1":-1],inputs:["in":input])["out"]?.sample.dimensions,[4,3,2])
        XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"permute",values:["dims":.array([0,3,1,2])],inputs:["in":input])["out"]?.sample.dimensions,[4,2,3])
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"permute",values:["dims":.array([0,1,1,2])],inputs:["in":input]))
        XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"chunk",values:["dim":-1,"chunks":4],inputs:["in":try .init(sample:[6])]).count,3,"Actual chunk count may be fewer than requested")
        let patches=try TensorOperationAuthoring.outputs(type:"unfold",values:["kernel_size":.array([2,3]),"stride":.array([1,2]),"dilation":.array([1,1]),"padding":.array([0,0])],inputs:["in":try .init(sample:[2,4,7])])
        XCTAssertEqual(patches["out"]?.sample.dimensions,[12,9])
        XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"matmul",values:[:],inputs:["a":try .init(sample:[2,3]),"b":try .init(sample:[3,4])])["out"]?.sample.dimensions,[2,4])
    }
    func testChangingMethodCannotEraseConnectedPartsOrOperands()throws {
        let registry=try ComponentRegistry.builtIn()
        let split=GraphNode(id:"parts",type:"split",name:"Slices",config:try TensorOperationAuthoring.defaults("split",registry:registry))
        let input=["in":try BatchedTensorLayout(sample:[7])]
        XCTAssertThrowsError(try TensorOperationAuthoring.switching(split,configurationVersion:2,to:"chunk",configuration:["dim":-1,"chunks":2],connectedInputs:["in"],connectedOutputs:["part2"],inputs:input,registry:registry))
        let next=try TensorOperationAuthoring.switching(split,configurationVersion:2,to:"chunk",configuration:["dim":-1,"chunks":4],connectedInputs:["in"],connectedOutputs:["part2"],inputs:input,registry:registry)
        XCTAssertEqual(next.id,split.id);XCTAssertEqual(next.name,"Slices");XCTAssertEqual(split.type,"split")
        XCTAssertThrowsError(try TensorOperationAuthoring.switching(split,configurationVersion:2,to:"chunk",configuration:["dim":-1,"chunks":4],connectedInputs:["in"],connectedOutputs:["part2"],inputs:[:],registry:registry))
        let einsum=GraphNode(id:"combine",type:"einsum",config:try TensorOperationAuthoring.defaults("einsum",registry:registry))
        XCTAssertThrowsError(try TensorOperationAuthoring.switching(einsum,configurationVersion:2,to:"einsum",configuration:["pattern":.string("diagonal"),"equation":.string("naa->na")],connectedInputs:["a","b"],connectedOutputs:[],inputs:[:],registry:registry))
        XCTAssertThrowsError(try TensorOperationAuthoring.switching(split,configurationVersion:2,to:"gather",configuration:["dim":-1],connectedInputs:[],connectedOutputs:[],inputs:[:],registry:registry))
    }
    func testActualWorkflowShapeAndWiresAreRecheckedBeforeCommit()throws {
        let registry=try ComponentRegistry.builtIn()
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try PreparedWorkflowExample.component("chunk").document(device:"cpu")
        let node=try XCTUnwrap(document.nodes.first {if case .modelPrimitive(let type,_,_)=$0.payload {return type=="chunk"};return false})
        let inputs=TensorOperationAuthoring.incoming(address:node.address,document:document,registry:registry,dataRegistry:data)
        XCTAssertEqual(inputs["in"]?.sample.dimensions,[2,13])
        XCTAssertThrowsError(try WorkflowEditing.apply(.setTensorOperation(address:node.address,type:"chunk",configuration:["dim":-1,"chunks":4]),to:document,modelRegistry:registry,dataRegistry:data))
        let changed=try WorkflowEditing.apply(.setTensorOperation(address:node.address,type:"chunk",configuration:["dim":-1,"chunks":6]),to:document,modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(changed.document.edges,document.edges)
        XCTAssertEqual(try WorkflowCompiler.modelSnapshot(changed.document,definition:document.nodes.first {if case .modelDefinition=$0.payload {return true};return false}!.address,modelRegistry:registry,dataRegistry:data).graph.nodes.first {$0.type=="chunk"}?.config["dim"],-1)
    }
    func testInlineAxisBindingsKeepRawSourceAndRefuseMethodReplacement()throws {
        let registry=try ComponentRegistry.builtIn(),root=WorkflowNodeAddress(domain:.model,localID:"container")
        let field=try XCTUnwrap(registry.component(type:"transpose")?.fields.first {$0.key=="dim0"})
        let definition=CompositeDefinition(reference:.init("transpose-fixture"),title:"Axes",body:.init(nodes:[.init(id:"axes",type:"transpose",config:["dim0":1,"dim1":2])]),inputs:[.init("in",targets:[.init("axes","in")])],outputs:[.init("out",source:.init("axes","out"))],parameters:[.init(field,targets:[.init("axes","dim0")])],composition:.init(kind:.graph,children:[.init(localID:"axes")]))
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:["dim0":-1]))],definitions:[definition]);document.schemaVersion=15
        let edited=try WorkflowInlineContainer.apply(.setTensorOperation(childID:"axes",type:"transpose",configuration:["dim0":-1,"dim1":-1]),to:document,root:root,registry:registry,idProvider:{"private-axes"})
        let retained=try WorkflowInlineContainer.definition(in:edited,root:root)
        XCTAssertEqual(retained.body.nodes[0].config,["dim0":1,"dim1":-1]);XCTAssertEqual(edited.definitions[0],definition)
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setTensorOperation(childID:"axes",type:"permute",configuration:["dims":.array([0,2,1])]),to:document,root:root,registry:registry,idProvider:{"refused"}))
    }

    func testReusableScopeCommitsConfigurationWithItsNewMethod()throws {
        let registry=try ComponentRegistry.builtIn()
        let scope=CompositeGraph(nodes:[.init(id:"axes",type:"transpose",config:["dim0":1,"dim1":2])])
        let edited=try ModelScopeEditing.apply(.setTensorOperation(nodeID:"axes",type:"permute",configuration:["dims":.array([0,2,1])]),to:scope,definitions:[],registry:registry)
        XCTAssertEqual(edited.scope.nodes[0].type,"permute")
        XCTAssertEqual(edited.scope.nodes[0].config,["dims":.array([0,2,1])])
        XCTAssertEqual(scope.nodes[0].config,["dim0":1,"dim1":2])
    }

}
