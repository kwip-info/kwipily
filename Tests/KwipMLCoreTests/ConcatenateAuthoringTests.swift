import XCTest
@testable import KwipMLCore
final class ConcatenateAuthoringTests:XCTestCase {
    func testExistingAxisContractAndTwoOrderedInputs()throws {
        XCTAssertNotNil(TensorOperationAuthoring.family("concat"))
        XCTAssertEqual(try TensorOperationAuthoring.inputPorts("concat",[:]),["a","b"])
        let a=try BatchedTensorLayout(sample:[4,12]),b=try BatchedTensorLayout(sample:[4,8])
        for axis in [-1,2] {XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"concat",values:["dim":.integer(Int64(axis))],inputs:["a":a,"b":b])["out"]?.sample.dimensions,[4,20])}
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"concat",values:["dim":0],inputs:["a":a,"b":b]))
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"concat",values:["dim":1],inputs:["a":a,"b":b]))
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"concat",values:["dim":-1],inputs:["a":a,"b":try .init(sample:[6,8])]))
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"concat",values:[:],inputs:["a":a]))
    }
    func testAtomicEditRetainsOriginalIdentityAndRejectsShapeMismatch()throws {
        let registry=try ComponentRegistry.builtIn()
        let node=GraphNode(id:"join",type:"concat",name:"Joined features",config:[:])
        let inputs=["a":try BatchedTensorLayout(sample:[4,12]),"b":try BatchedTensorLayout(sample:[4,8])]
        let next=try TensorOperationAuthoring.switching(node,configurationVersion:2,to:"concat",configuration:["dim":2],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:inputs,registry:registry)
        XCTAssertEqual(next.id,node.id);XCTAssertEqual(next.name,node.name);XCTAssertEqual(next.config,["dim":2]);XCTAssertEqual(node.config,[:])
        XCTAssertThrowsError(try TensorOperationAuthoring.switching(node,configurationVersion:2,to:"concat",configuration:["dim":1],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:inputs,registry:registry))
    }
    func testExistingWorkflowCompilesAfterAxisEditWithoutRewiring()throws {
        let registry=try ComponentRegistry.builtIn()
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try PreparedWorkflowExample.component("concat").document(device:"cpu")
        let node=try XCTUnwrap(document.nodes.first {if case .modelPrimitive("concat",_,_)=$0.payload {return true};return false})
        let changed=try WorkflowEditing.apply(.setTensorOperation(address:node.address,type:"concat",configuration:["dim":1]),to:document,modelRegistry:registry,dataRegistry:data).document
        XCTAssertEqual(changed.edges,document.edges)
        XCTAssertEqual(changed.nodes.map(\.address),document.nodes.map(\.address))
        let boundary=try XCTUnwrap(document.nodes.first {WorkflowModelBoundary.isVisibleDefinition($0,in:document)})
        let snapshot=try WorkflowCompiler.modelSnapshot(changed,definition:boundary.address,modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(snapshot.graph.nodes.first {$0.type=="concat"}?.config["dim"],1)
        XCTAssertThrowsError(try WorkflowEditing.apply(.setTensorOperation(address:node.address,type:"concat",configuration:["dim":0]),to:document,modelRegistry:registry,dataRegistry:data))
    }

}
