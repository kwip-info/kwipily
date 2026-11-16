import XCTest
@testable import KwipMLCore

final class CanvasPresentationTests: XCTestCase {
    func testDisplaySpacingPreservesPairwiseSeparationAndAuthoredCoordinates() {
        let origins = [CGPoint(x:40,y:40),CGPoint(x:230,y:40),CGPoint(x:40,y:164),CGPoint(x:420,y:288)]
        for detail in CanvasNodeDetail.allCases {
            for direction in CanvasFlowDirection.allCases {
                let settings = CanvasPresentationSettings(flowDirection:direction,nodeDetail:detail)
                let projection = CanvasCoordinateProjection(settings:settings)
                let size = CanvasNodeGeometry.size(detail:detail,direction:direction,inputCount:2,outputCount:2)
                let rects = origins.map { CGRect(origin:projection.project($0),size:size) }
                for index in origins.indices {
                    let restored = projection.inverse(projection.project(origins[index]))
                    XCTAssertEqual(restored.x,origins[index].x,accuracy:0.000001)
                    XCTAssertEqual(restored.y,origins[index].y,accuracy:0.000001)
                    for other in origins.indices where other > index { XCTAssertFalse(rects[index].intersects(rects[other])) }
                }
                let delta = CGPoint(x:37,y:-19)
                let restoredDelta = projection.inverse(projection.project(delta))
                XCTAssertEqual(restoredDelta.x,delta.x,accuracy:0.000001)
                XCTAssertEqual(restoredDelta.y,delta.y,accuracy:0.000001)
            }
        }
    }
    func testSpacingUsesRegistryPortCountsAndNeverCompressesSmallPositions() {
        let small = CanvasCoordinateProjection(settings:.init(nodeDetail:.small))
        XCTAssertGreaterThanOrEqual(small.scaleX,1); XCTAssertGreaterThanOrEqual(small.scaleY,1)
        let branching = CanvasCoordinateProjection(settings:.init(flowDirection:.topToBottom),maximumInputCount:8)
        XCTAssertGreaterThanOrEqual(branching.scaleX,1)
        let extent = CanvasNodeGeometry.size(detail:.medium,direction:.topToBottom,inputCount:8,outputCount:2)
        XCTAssertGreaterThanOrEqual(branching.project(CGPoint(x:150,y:0)).x,extent.width)
    }
    func testDirectionProjectionIsExactlyReversible() {
        let point = CGPoint(x: 173.5, y: 492.25)
        for direction in CanvasFlowDirection.allCases {
            XCTAssertEqual(CanvasNodeGeometry.inverse(point: CanvasNodeGeometry.project(point: point, direction: direction), direction: direction), point)
        }
        XCTAssertEqual(CanvasNodeGeometry.project(point: point, direction: .topToBottom), CGPoint(x: point.y, y: point.x))
    }
    func testAllSizesKeepBranchPortsDistinctAndInsideBounds() {
        for detail in CanvasNodeDetail.allCases {
            for direction in CanvasFlowDirection.allCases {
                let size = CanvasNodeGeometry.size(detail: detail, direction: direction, inputCount: 8, outputCount: 2)
                let ports = (0..<8).map { CanvasNodeGeometry.portAnchor(size: size, direction: direction, output: false, index: $0, count: 8) }
                XCTAssertEqual(Set(ports.map { "\($0.x),\($0.y)" }).count, 8)
                for port in ports { XCTAssertGreaterThanOrEqual(port.x, 0); XCTAssertLessThanOrEqual(port.x, size.width); XCTAssertGreaterThanOrEqual(port.y, 0); XCTAssertLessThanOrEqual(port.y, size.height) }
                for pair in zip(ports, ports.dropFirst()) { XCTAssertGreaterThanOrEqual(hypot(pair.0.x-pair.1.x,pair.0.y-pair.1.y), 28) }
            }
        }
    }
    func testWireTangentsFollowRowIconsAndInverseDragFollowsArrangement() {
        let controls = CanvasNodeGeometry.controlPoints(start: CGPoint(x: 40,y: 20), end: CGPoint(x: 130,y: 270), direction: .topToBottom)
        XCTAssertGreaterThan(controls.0.x, 40); XCTAssertEqual(controls.0.y, 20)
        XCTAssertLessThan(controls.1.x, 130); XCTAssertEqual(controls.1.y, 270)
        XCTAssertEqual(CanvasNodeGeometry.inverse(point: CGPoint(x: 30,y: -10), direction: .topToBottom), CGPoint(x: -10,y: 30))
    }
}
