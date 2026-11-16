import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TrainedModelInputContractTests:XCTestCase {
    func testExportInputUsesCapturedIntegerBoundaryAndExactColumnOrder()throws {
        let registry=try ComponentRegistry.builtIn()
        let nodes=[GraphNode(id:"source",type:"index_input",config:["shape":.array([1,2])]),
                   GraphNode(id:"lookup",type:"embedding",config:Dictionary(uniqueKeysWithValues:EmbeddingComponent.definition.fields.map {($0.key,$0.defaultValue)})),
                   GraphNode(id:"flat",type:"flatten",config:["start_dim":1,"end_dim":-1]),
                   GraphNode(id:"head",type:"linear",config:["in_features":16,"out_features":2,"bias":true]),
                   GraphNode(id:"output",type:"output")]
        let graph=GraphDocument(nodes:nodes,edges:zip(nodes,nodes.dropFirst()).map {GraphEdge(source:$0.id,target:$1.id)})
        let value=try TrainedModelInputContract(graph:graph,registry:registry,columns:["second"," first "],target:"label")
        XCTAssertEqual(value.input["dtype"],.string("int64"))
        XCTAssertEqual(value.input["shape"],.array([.string("N"),1,2]))
        XCTAssertEqual(value.input["featureColumns"],.array([.string("second"),.string(" first ")]))
        XCTAssertEqual(value.pythonShape,"[N, 1, 2]")
        XCTAssertEqual(value.reviewSummary,"Prepared inputs · int64 · shape [N, 1, 2]")
        XCTAssertTrue(value.valueGuidance.contains("integer"))
        XCTAssertThrowsError(try TrainedModelInputContract(graph:graph,registry:registry,columns:["a"],target:"label"))
        XCTAssertThrowsError(try TrainedModelInputContract(graph:graph,registry:registry,columns:["a","b"],target:"a"))
    }
    func testLegacyFloatExportContractRemainsCompatible()throws {
        let value=try TrainedModelInputContract(graph:ExampleFlows.parallelClassifier,registry:.builtIn(),columns:["x","y"],target:"label")
        XCTAssertEqual(value.input,["dtype":.string("float32"),"shape":.array([.string("N"),2]),"featureColumns":.array([.string("x"),.string("y")]),"preprocessing":.string("alreadyPrepared")])
    }
}
