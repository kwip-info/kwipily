import XCTest
@testable import KwipMLCore

final class TensorRoutingPortIntegrationTests:XCTestCase {
    private let registry=try! ComponentRegistry.builtIn()
    func testAllFourRealPreparedModelShapesResolveAtNamedPorts()throws {
        for id in ["split","chunk","gather","einsum"] {
            let graph=try ExampleFlows.tensorRoutingClassifier(componentID:id),result=GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(result.isValid,"\(id): \(result.issues.map(\.message))")
            XCTAssertEqual(result.outputTensorContracts[.init("output","out")]?.layout.sample.dimensions,[2])
            if id == "split" {XCTAssertEqual(result.outputTensorContracts[.init("routing","part2")]?.layout.sample.dimensions,[2,2]);XCTAssertNil(result.outputTensorContracts[.init("routing","out")])}
            if id == "chunk" {XCTAssertEqual(result.outputTensorContracts[.init("routing","part4")]?.layout.sample.dimensions,[2,1]);XCTAssertNil(result.outputTensorContracts[.init("routing","part5")])}
            if id == "gather" {XCTAssertEqual(result.outputTensorContracts[.init("positions","indices")]?.dtype,.int64);XCTAssertEqual(result.outputTensorContracts[.init("routing","out")]?.dtype,.float32)}
        }
    }
    func testUnaryArrangementRequiresOnlyItsActualSource()throws {
        let graph=GraphDocument(name:"Diagonal",nodes:[
            .init(id:"input",type:"input",name:"Source",x:0,y:0,config:["features":2]),
            .init(id:"learn",type:"linear",name:"Learn",x:100,y:0,config:["in_features":2,"out_features":9,"bias":true]),
            .init(id:"shape",type:"reshape",name:"Square sample",x:200,y:0,config:["shape":.array([3,3])]),
            .init(id:"routing",type:"einsum",name:"Diagonal",x:300,y:0,config:["pattern":.string("diagonal")]),
            .init(id:"head",type:"linear",name:"Class scores",x:400,y:0,config:["in_features":3,"out_features":2,"bias":true]),
            .init(id:"output",type:"output",name:"Output",x:500,y:0)],edges:[
                .init(id:"e0",source:"input",sourcePort:"out",target:"learn",targetPort:"in"),
                .init(id:"e1",source:"learn",sourcePort:"out",target:"shape",targetPort:"in"),
                .init(id:"e2",source:"shape",sourcePort:"out",target:"routing",targetPort:"a"),
                .init(id:"e3",source:"routing",sourcePort:"out",target:"head",targetPort:"in"),
                .init(id:"e4",source:"head",sourcePort:"out",target:"output",targetPort:"in")])
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.isValid,result.issues.map(\.message).joined(separator:"; "))
        XCTAssertEqual(result.outputTensorContracts[.init("routing","out")]?.layout.sample.dimensions,[3])
    }
    func testUnavailableChunkPartRefusesAndSavedConnectionIdentityIsRetained()throws {
        var graph=try ExampleFlows.tensorRoutingClassifier(componentID:"chunk")
        let edge=try XCTUnwrap(graph.edges.firstIndex(where:{$0.source == "routing"}));let savedID=graph.edges[edge].id
        graph.edges[edge].sourcePort="part5"
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertFalse(result.isValid);XCTAssertEqual(graph.edges[edge].id,savedID);XCTAssertEqual(graph.edges[edge].sourcePort,"part5")
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"routing",port:"part5",direction:.output,graph:graph,registry:registry).kind,.unknown)
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"routing",port:"part4",direction:.output,graph:graph,registry:registry).declared,["[N, 2, 1]","float32"])
    }
    func testConnectionsUseActualPartsAndPreserveResolvedPartShape()throws {
        var graph=try ExampleFlows.tensorRoutingClassifier(componentID:"chunk")
        let removed=try XCTUnwrap(graph.edges.first(where:{$0.source == "routing"}))
        graph.edges.removeAll {$0.id == removed.id}
        let state=GraphEditorState(document:graph)
        XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:.init("routing","part5"),target:.init(removed.target,removed.targetPort)),to:state,registry:registry,makeID:{"missing-part"})) {error in
            XCTAssertEqual((error as? GraphCommandError)?.code,.missingPort)
        }
        let connected=try GraphCommandEngine.apply(.connect(source:.init("routing","part4"),target:.init(removed.target,removed.targetPort)),to:state,registry:registry,makeID:{"actual-part"})
        XCTAssertEqual(connected.after.document.edges.last?.sourcePort,"part4")
        XCTAssertTrue(GraphValidator.validate(connected.after.document,registry:registry).isValid)
    }
    func testGatherRefusesFloatPositionsWithoutChangingGraph()throws {
        var graph=try ExampleFlows.tensorRoutingClassifier(componentID:"gather")
        let removed=try XCTUnwrap(graph.edges.first(where:{$0.target == "routing" && $0.targetPort == "indices"}))
        graph.edges.removeAll {$0.id == removed.id}
        let source=try XCTUnwrap(graph.edges.first(where:{$0.target == "routing" && $0.targetPort == "source"}))
        let state=GraphEditorState(document:graph)
        XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:.init(source.source,source.sourcePort),target:.init("routing","indices")),to:state,registry:registry,makeID:{"wrong-positions"})) {error in
            XCTAssertEqual((error as? GraphCommandError)?.code,.incompatibleShape)
        }
        XCTAssertEqual(state.document,graph)
    }

}
