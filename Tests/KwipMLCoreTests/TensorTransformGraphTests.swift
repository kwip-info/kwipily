import XCTest
@testable import KwipMLCore

final class TensorTransformGraphTests:XCTestCase {
    func testBundledExampleTrainsThroughAllThreeTransforms()throws {
        let graph=ExampleFlows.tensorLayoutClassifier
        let result=GraphValidator.validate(graph,registry:try .builtIn())
        XCTAssertTrue(result.isValid,"\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["swap"]?.displayDimensions,["N","4","3"])
        XCTAssertEqual(result.widths["flat"],12)
        XCTAssertEqual(result.widths["output"],2)
    }
    func testBundledTransformsMatchLessonContractsAndExecuteShapePropagation()throws {
        let registry=try ComponentRegistry.builtIn()
        for type in ["reshape","transpose","flatten"] {
            XCTAssertEqual(try XCTUnwrap(registry.component(type:type)),TensorLearningCanvas.registry.component(type:type))
        }
        let (graph,_)=try fixture()
        XCTAssertTrue(GraphValidator.validate(graph,registry:registry).isValid)
    }
    func fixture()throws->(GraphDocument,ComponentRegistry) {
        let registry=try ComponentRegistry.builtIn()
        let nodes=[GraphNode(id:"input",type:"input",config:["features":2]),GraphNode(id:"project",type:"linear",config:["in_features":2,"out_features":12]),GraphNode(id:"shape",type:"reshape",config:["shape":.array([3.0,4.0])]),GraphNode(id:"swap",type:"transpose",config:["dim0":1,"dim1":-1]),GraphNode(id:"flat",type:"flatten",config:["start_dim":1,"end_dim":-1]),GraphNode(id:"head",type:"linear",config:["in_features":12,"out_features":2]),GraphNode(id:"output",type:"output")]
        let edges=zip(nodes,nodes.dropFirst()).map {GraphEdge(source:$0.id,target:$1.id)}
        return (GraphDocument(nodes:nodes,edges:edges),registry)
    }
    func testConnectionsRespectFullShapesAndLastAxisLinear()throws {
        var (graph,registry)=try fixture()
        graph.edges.removeAll {$0.target == "head" || $0.target == "output"}
        let state=GraphEditorState(document:graph)
        XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:GraphPortAddress("swap","out"),target:GraphPortAddress("head","in")),to:state,registry:registry))
        XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:GraphPortAddress("swap","out"),target:GraphPortAddress("output","in")),to:state,registry:registry))
        graph.nodes[5].config["in_features"]=3
        XCTAssertNoThrow(try GraphCommandEngine.apply(.connect(source:GraphPortAddress("swap","out"),target:GraphPortAddress("head","in")),to:GraphEditorState(document:graph),registry:registry))
        graph.nodes.append(GraphNode(id:"other",type:"reshape",config:["shape":.array([2,2,3])]))
        graph.nodes.append(GraphNode(id:"sum",type:"add"))
        graph.edges += [GraphEdge(source:"project",target:"other"),GraphEdge(source:"other",target:"sum",targetPort:"a")]
        XCTAssertThrowsError(try GraphCommandEngine.apply(.connect(source:GraphPortAddress("swap","out"),target:GraphPortAddress("sum","b")),to:GraphEditorState(document:graph),registry:registry))
    }
    func testPortPresentationUsesConnectedInputAndFullOutputShape()throws {
        let (graph,registry)=try fixture()
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"shape",port:"in",direction:.input,graph:graph,registry:registry).declared,["[N, 12]","float32"])
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"shape",port:"out",direction:.output,graph:graph,registry:registry).declared,["[N, 3, 4]","float32"])
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"shape",port:"wrong",direction:.output,graph:graph,registry:registry).kind,.unknown)
        XCTAssertTrue(result.tensorPortDescriptor(nodeID:"shape",port:"out",direction:.output,graph:graph,registry:registry).measured.isEmpty)
    }
    func testGraphPropagatesRanksAndCanonicalSettings()throws {
        let (graph,registry)=try fixture()
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.isValid,"\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["swap"]?.displayDimensions,["N","4","3"])
        XCTAssertEqual(result.tensorLayouts["flat"]?.displayDimensions,["N","12"])
        XCTAssertNil(result.widths["swap"])
        XCTAssertEqual(result.widths["output"],2)
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:graph.nodes[2],schemaVersion:2,registry:registry)["shape"],.array([3,4]))
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:graph.nodes[2],schemaVersion:1,registry:registry))
    }
    func testInvalidReshapeStopsDependentShapePropagation()throws {
        var (graph,registry)=try fixture()
        graph.nodes[2].config["shape"] = .array([5,-1])
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.issues.contains {$0.nodeID == "shape" && $0.message.contains("infer")})
        XCTAssertNil(result.tensorLayouts["output"])
        graph.nodes[2].config["shape"] = .array([-1,-1])
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:graph.nodes[2],schemaVersion:2,registry:registry))
    }
}
