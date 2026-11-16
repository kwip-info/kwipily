import XCTest
import CoreGraphics
@testable import KwipMLCore

final class TensorRoutingAutoPlacementTests: XCTestCase {
    func testRibbonInsertionAvoidsActualRenderedRoutingGeometry() throws {
        let registry = try ComponentRegistry.builtIn()
        for direction in [CanvasFlowDirection.leftToRight, .topToBottom] {
            for detail in [CanvasNodeDetail.small, .medium, .large] {
                let settings = CanvasPresentationSettings(flowDirection: direction, nodeDetail: detail)
                for offset in [0.0, 41.0, 97.0] {
                    var graph = try ExampleFlows.tensorRoutingClassifier(componentID: "chunk")
                    for index in graph.nodes.indices {
                        graph.nodes[index].x = 60 + Double(index % 4) * 180 + offset
                        graph.nodes[index].y = 420 + Double(index / 4) * 120 + Double(index % 3) * 17
                    }
                    let snapshot = CanvasGeometrySnapshot(registry: registry, presentation: settings, graph: graph, validation: GraphValidator.validate(graph, registry: registry))
                    let occupied = graph.nodes.map { CGRect(origin: snapshot.projection.project(CGPoint(x: $0.x, y: $0.y)), size: snapshot.size(nodeID: $0.id)).insetBy(dx: -12, dy: -12) }
                    let newSize = CanvasNodeGeometry.size(detail: detail, direction: direction, inputCount: 1, outputCount: 1)
                    let expected = try XCTUnwrap((0..<128).map { CGPoint(x: 60 + Double($0 % 4) * 180, y: 420 + Double($0 / 4) * 120) }.first { point in
                        let bounds = CGRect(origin: snapshot.projection.project(point), size: newSize)
                        return !occupied.contains { $0.intersects(bounds) }
                    })
                    XCTAssertEqual(CanvasAutoPlacement.origin(type: "linear", graph: graph, registry: registry, catalog: registry, presentation: settings, start: CGPoint(x: 60, y: 420)), expected)
                }
            }
        }
    }
}
