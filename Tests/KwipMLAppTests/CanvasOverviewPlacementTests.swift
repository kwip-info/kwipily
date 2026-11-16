import XCTest
import CoreGraphics
@testable import KwipMLApp

final class CanvasOverviewPlacementTests: XCTestCase {
    func testReportedSmallValidationOverlapSuppressesMap() {
        let viewport=CGSize(width:1253,height:818)
        let validation=CGRect(x:1067,y:656,width:38,height:44)
        XCTAssertFalse(CanvasOverviewPlacement.shouldShow(viewport:viewport,toolbarInset:60,graphNodes:[validation],zoom:1,scrollOrigin:.zero))
        XCTAssertTrue(CanvasOverviewPlacement.shouldShow(viewport:viewport,toolbarInset:60,graphNodes:[CGRect(x:900,y:656,width:38,height:44)],zoom:1,scrollOrigin:.zero))
    }
    func testProjectedNodeBoundsIncludeZoomScrollAndReservedInset() {
        let frame=CanvasOverviewPlacement.screenFrame(graph:CGRect(x:100,y:200,width:76,height:76),zoom:0.5,scrollOrigin:CGPoint(x:20,y:30),toolbarInset:60)
        XCTAssertEqual(frame,CGRect(x:30,y:130,width:38,height:38))
        for zoom in [0.1,0.5,1.5] {
            let viewport=CGSize(width:600,height:500)
            let graph=CGRect(x:450/zoom,y:360/zoom,width:38/zoom,height:44/zoom)
            XCTAssertFalse(CanvasOverviewPlacement.shouldShow(viewport:viewport,toolbarInset:102,graphNodes:[graph],zoom:zoom,scrollOrigin:CGPoint(x:0,y:50)))
        }
    }
    func testProtectivePaddingAndNarrowViewport() {
        XCTAssertFalse(CanvasOverviewPlacement.shouldShow(viewport:CGSize(width:150,height:500),toolbarInset:60,graphNodes:[],zoom:1,scrollOrigin:.zero))
        XCTAssertFalse(CanvasOverviewPlacement.shouldShow(viewport:CGSize(width:600,height:160),toolbarInset:102,graphNodes:[],zoom:1,scrollOrigin:.zero))
        // Map starts x446; the node ends x442, separated by only four screen points.
        XCTAssertFalse(CanvasOverviewPlacement.shouldShow(viewport:CGSize(width:600,height:500),toolbarInset:60,graphNodes:[CGRect(x:420,y:360,width:22,height:30)],zoom:1,scrollOrigin:.zero))
        XCTAssertTrue(CanvasOverviewPlacement.shouldShow(viewport:CGSize(width:600,height:500),toolbarInset:60,graphNodes:[CGRect(x:900,y:900,width:76,height:76)],zoom:1,scrollOrigin:.zero))
    }
}
