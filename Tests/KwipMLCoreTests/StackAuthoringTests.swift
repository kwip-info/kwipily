import XCTest
@testable import KwipMLCore
final class StackAuthoringTests:XCTestCase {
    func testInsertionUsesOutputRankAndPreservesTwoOrderedInputs()throws {
        XCTAssertNotNil(TensorOperationAuthoring.family("stack"))
        XCTAssertEqual(try TensorOperationAuthoring.inputPorts("stack",[:]),["a","b"])
        let input=try BatchedTensorLayout(sample:[4,8])
        for (axis,shape) in [(1,[2,4,8]),(2,[4,2,8]),(-2,[4,2,8]),(-1,[4,8,2])] {
            XCTAssertEqual(try TensorOperationAuthoring.outputs(type:"stack",values:["dim":.integer(Int64(axis))],inputs:["a":input,"b":input])["out"],try BatchedTensorLayout(sample:shape))
        }
        for axis in [0,-4,4] {XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"stack",values:["dim":.integer(Int64(axis))],inputs:["a":input,"b":input]))}
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"stack",values:[:],inputs:["a":input,"b":try .init(sample:[4,6])]))
        let maximum=try BatchedTensorLayout(sample:[1,1,1,1,1,1,1])
        XCTAssertThrowsError(try TensorOperationAuthoring.outputs(type:"stack",values:[:],inputs:["a":maximum,"b":maximum]))
    }
    func testLegacyDefaultsAtomicEditAndTypedRefusal()throws {
        let registry=try ComponentRegistry.builtIn(),original=GraphNode(id:"stack",type:"stack",name:"Paired",config:[:])
        let a=try TypedTensorContract(sample:[4,8],dtype:.int64),b=try TypedTensorContract(sample:[4,8],dtype:.float32)
        XCTAssertEqual(try TensorOperationAuthoring.switching(original,configurationVersion:2,to:"stack",configuration:["dim":-1],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:["a":a.layout,"b":a.layout],registry:registry),original)
        let next=try TensorOperationAuthoring.switching(original,configurationVersion:2,to:"stack",configuration:["dim":1],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:["a":a.layout,"b":a.layout],registry:registry,inputContracts:["a":a,"b":a])
        XCTAssertEqual(next.id,original.id);XCTAssertEqual(next.name,original.name);XCTAssertEqual(next.config["dim"],1)
        XCTAssertThrowsError(try TensorOperationAuthoring.switching(original,configurationVersion:2,to:"stack",configuration:["dim":1],connectedInputs:["a","b"],connectedOutputs:["out"],inputs:["a":a.layout,"b":b.layout],registry:registry,inputContracts:["a":a,"b":b]))
    }
    func testExistingWorkflowCompilesAfterAxisEditWithoutRewiring()throws {
        let registry=try ComponentRegistry.builtIn()
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let document=try PreparedWorkflowExample.component("stack").document(device:"cpu")
        let node=try XCTUnwrap(document.nodes.first {if case .modelPrimitive("stack",_,_)=$0.payload {return true};return false})
        let changed=try WorkflowEditing.apply(.setTensorOperation(address:node.address,type:"stack",configuration:["dim":-2]),to:document,modelRegistry:registry,dataRegistry:data).document
        XCTAssertEqual(changed.edges,document.edges)
        XCTAssertEqual(changed.nodes.map(\.address),document.nodes.map(\.address))
        let boundary=try XCTUnwrap(document.nodes.first {WorkflowModelBoundary.isVisibleDefinition($0,in:document)})
        let snapshot=try WorkflowCompiler.modelSnapshot(changed,definition:boundary.address,modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(snapshot.graph.nodes.first {$0.type=="stack"}?.config["dim"],-2)
        XCTAssertThrowsError(try WorkflowEditing.apply(.setTensorOperation(address:node.address,type:"stack",configuration:["dim":0]),to:document,modelRegistry:registry,dataRegistry:data))
    }

}
