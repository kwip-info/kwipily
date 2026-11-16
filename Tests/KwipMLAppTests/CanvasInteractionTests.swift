import XCTest
import CoreGraphics
@testable import KwipMLApp

final class CanvasInteractionTests: XCTestCase {
    func testPortTargetDoesNotGuessAtLowZoomOrOverlappingBranches() {
        let anchors = [CGPoint(x:100,y:40),CGPoint(x:100,y:68)]
        XCTAssertEqual(CanvasInteractionGeometry.portTarget(point:anchors[0],anchors:anchors,zoom:1),0)
        XCTAssertNil(CanvasInteractionGeometry.portTarget(point:CGPoint(x:100,y:54),anchors:anchors,zoom:0.25))
        XCTAssertNil(CanvasInteractionGeometry.portTarget(point:anchors[0],anchors:anchors,zoom:0))
    }
    func testMacDeleteCharactersRouteWithoutTreatingOrdinaryTextAsDeletion() {
        for character in ["\u{8}", "\u{7f}", "\u{f728}"] { XCTAssertTrue(CanvasKeyRouting.isDelete(character)) }
        for character in ["", "d", "Delete", "x", "\n"] { XCTAssertFalse(CanvasKeyRouting.isDelete(character)) }
    }

    func testGroupTranslationClampsWholeSelectionWithoutChangingSpacing() {
        let value = CanvasInteractionGeometry.translation(CGSize(width: -100, height: 900), origins: [CGPoint(x: -999_980, y: 30), CGPoint(x: 500, y: 999_700)])
        XCTAssertEqual(value.width, -20)
        XCTAssertEqual(value.height, 300)
    }
    func testPlacementAndGroupMovementBeyondOriginalCanvasExtent() {
        let origin = CGPoint(x:8500,y:2200)
        XCTAssertEqual(CanvasInteractionGeometry.boundedOrigin(origin),origin)
        let delta = CanvasInteractionGeometry.translation(CGSize(width:500,height:900),origins:[origin,CGPoint(x:9500,y:3100)])
        XCTAssertEqual(delta,CGSize(width:500,height:900))
        XCTAssertEqual(CanvasInteractionGeometry.boundedOrigin(CGPoint(x:1_200_000,y:-8)),CGPoint(x:1_000_000,y:-8))
    }
    func testGroupTranslationLeavesInvalidImportedDraftInPlace() {
        XCTAssertEqual(CanvasInteractionGeometry.translation(CGSize(width: 20, height: 20), origins: [CGPoint(x: -.infinity, y: 0)]), .zero)
    }
    func testWireHitToleranceUsesScreenPointsAtEveryZoom() {
        for zoom in [0.1, 0.25, 0.65, 1.0, 1.5] {
            XCTAssertTrue(CanvasInteractionGeometry.hitsWire(point: CGPoint(x: 600, y: 100 + 7 / zoom), start: CGPoint(x: 0, y: 100), end: CGPoint(x: 1200, y: 100), zoom: zoom))
            XCTAssertFalse(CanvasInteractionGeometry.hitsWire(point: CGPoint(x: 600, y: 100 + 12 / zoom), start: CGPoint(x: 0, y: 100), end: CGPoint(x: 1200, y: 100), zoom: zoom))
        }
    }
    func testWireHitRejectsEndpointAndInvalidScale() {
        XCTAssertFalse(CanvasInteractionGeometry.hitsWire(point: .zero, start: .zero, end: CGPoint(x: 300,y: 0), zoom: 1))
        XCTAssertFalse(CanvasInteractionGeometry.hitsWire(point: CGPoint(x: 150,y: 0), start: .zero, end: CGPoint(x: 300,y: 0), zoom: 0))
    }
}
