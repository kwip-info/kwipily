import XCTest
@testable import KwipMLCore

final class ValidationTensorContractTests:XCTestCase {
    func testLegacyFlowInfersFloatContractsAndPreservesShapes() throws {
        let registry=try ComponentRegistry.builtIn()
        let result=GraphValidator.validate(ExampleFlows.parallelClassifier,registry:registry)
        XCTAssertTrue(result.isValid)
        XCTAssertEqual(result.tensorContracts.count,result.tensorLayouts.count)
        XCTAssertFalse(result.tensorContracts.isEmpty)
        XCTAssertTrue(result.tensorContracts.values.allSatisfy {$0.dtype == .float32})
        for (id,contract) in result.tensorContracts {XCTAssertEqual(result.tensorLayouts[id],contract.layout)}
    }
    func testPortsReadSourceTypeAndRemoveStaleShapeTogether() throws {
        let graph=GraphDocument(name:"Types",nodes:[
            GraphNode(id:"i",type:"input",name:"User label",x:0,y:0,config:["features":2]),
            GraphNode(id:"o",type:"output",name:"Result",x:100,y:0)
        ],edges:[GraphEdge(id:"e",source:"i",target:"o",targetPort:"in")])
        let registry=try ComponentRegistry.builtIn()
        var result=ValidationResult()
        result.tensorContracts["i"]=try TypedTensorContract(sample:[2],dtype:.int64)
        for (id,port,direction) in [("i","out",PortDirection.output),("o","in",.input)] {
            XCTAssertEqual(result.tensorPortDescriptor(nodeID:id,port:port,direction:direction,graph:graph,registry:registry).declared,["[N, 2]","int64"])
        }
        result.tensorContracts.removeValue(forKey:"i")
        XCTAssertNil(result.tensorLayouts["i"])
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"o",port:"in",direction:.input,graph:graph,registry:registry).declared,["Shape unresolved"])
    }
}
