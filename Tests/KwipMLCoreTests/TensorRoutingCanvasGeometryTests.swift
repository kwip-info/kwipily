import XCTest
@testable import KwipMLCore

final class TensorRoutingCanvasGeometryTests: XCTestCase {
    private let parts = (0..<8).map { "part\($0)" }
    private var registry: ComponentRegistry {
        .init(components: [.init(id: "linear", inputs: ["in"], outputs: ["out"]),
            .init(id: "concat", inputs: ["a", "b"], outputs: ["out"]),
            .init(id: "split", inputs: ["in"], outputs: parts),
            .init(id: "chunk", inputs: ["in"], outputs: parts),
            .init(id: "einsum", inputs: ["a", "b"], outputs: ["out"])])
    }
    private func validation(_ shapes: [String: [Int]]) throws -> ValidationResult {
        var result = ValidationResult()
        for (node, shape) in shapes {
            result.outputTensorContracts[.init(node, "out")] = .init(layout: try .init(sample: shape), dtype: .float32)
        }
        return result
    }
    func testConfiguredPartsArePerNodeAndChangeWithActualShape() throws {
        let graph = GraphDocument(nodes: [.init(id:"source",type:"linear"),.init(id:"other",type:"linear"),
            .init(id:"three",type:"split",config:["split_size":2]),
            .init(id:"one",type:"split",config:["split_size":9]),
            .init(id:"five",type:"chunk",config:["chunks":6])], edges:[
                .init(id:"a",source:"source",target:"three"), .init(id:"b",source:"source",target:"one"),
                .init(id:"c",source:"other",target:"five")])
        let snapshot = CanvasGeometrySnapshot(registry:registry,presentation:.init(),graph:graph,
            validation:try validation(["source":[2,5],"other":[2,13]]))
        XCTAssertEqual(snapshot.definition(nodeID:"three")?.outputs,["part0","part1","part2"])
        XCTAssertEqual(snapshot.definition(nodeID:"one")?.outputs,["part0"])
        XCTAssertEqual(snapshot.definition(nodeID:"five")?.outputs,Array(parts.prefix(5)))
        XCTAssertNotEqual(snapshot.size(nodeID:"one"),snapshot.size(nodeID:"five"))
        let changed = CanvasGeometrySnapshot(registry:registry,presentation:.init(),graph:graph,
            validation:try validation(["source":[2,2],"other":[2,2]]))
        XCTAssertEqual(changed.definition(nodeID:"three")?.outputs,["part0"])
        XCTAssertEqual(snapshot.definition(nodeID:"three")?.outputs,["part0","part1","part2"])
    }
    func testUnaryEinsumAndUnresolvedPartVocabulary() {
        let graph = GraphDocument(nodes:[.init(id:"diagonal",type:"einsum",config:["pattern":.string("diagonal")]),
            .init(id:"product",type:"einsum"),.init(id:"pending",type:"split")])
        let snapshot = CanvasGeometrySnapshot(registry:registry,presentation:.init(),graph:graph,validation:.init())
        XCTAssertEqual(snapshot.definition(nodeID:"diagonal")?.inputs,["a"])
        XCTAssertEqual(snapshot.definition(nodeID:"product")?.inputs,["a","b"])
        XCTAssertEqual(snapshot.definition(nodeID:"pending")?.outputs,parts)
    }
    func testRoutingCeilingsDoNotMoveOldGraphsInAnyPresentation() {
        let prior = ComponentRegistry(components:registry.components.filter { !["split","chunk","einsum"].contains($0.id) })
        let graph = GraphDocument(nodes:[.init(id:"existing",type:"linear",x:170,y:-45)])
        for direction in [CanvasFlowDirection.leftToRight,.topToBottom] {
            for detail in [CanvasNodeDetail.small,.medium,.large] {
                var settings = CanvasPresentationSettings(); settings.flowDirection=direction; settings.nodeDetail=detail
                let before = CanvasGeometrySnapshot(registry:prior,presentation:settings)
                let after = CanvasGeometrySnapshot(registry:registry,presentation:settings,graph:graph,validation:.init())
                XCTAssertEqual(after.projection,before.projection)
                XCTAssertEqual(after.projection.project(CGPoint(x:170,y:-45)),before.projection.project(CGPoint(x:170,y:-45)))
                XCTAssertEqual(after.size(nodeID:"existing"),before.size(type:"linear"))
            }
        }
    }
    func testUnavailableSavedEndpointsHaveDistinctRepairAnchorsWithoutRewritingEdges() throws {
        let graph = GraphDocument(nodes:[.init(id:"source",type:"linear"),.init(id:"split",type:"split",config:["split_size":9]),
            .init(id:"target",type:"linear")],edges:[.init(id:"in",source:"source",target:"split"),
                .init(id:"old",source:"split",sourcePort:"part7",target:"target")])
        let before = graph
        let snapshot = CanvasGeometrySnapshot(registry:registry,presentation:.init(),graph:graph,
            validation:try validation(["source":[2,5]]))
        XCTAssertEqual(snapshot.definition(nodeID:"split")?.outputs,["part0"])
        XCTAssertNotEqual(snapshot.portAnchor(nodeID:"split",output:true,port:"part7"),snapshot.portAnchor(nodeID:"split",output:true,port:"part0"))
        XCTAssertEqual(graph,before)
        XCTAssertEqual(graph.edges.last?.sourcePort,"part7")
    }
}
