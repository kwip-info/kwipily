import XCTest
import CoreGraphics
import AppKit
@testable import KwipMLApp

final class CanvasViewportInsetTests: XCTestCase {
    func testRevealKeepsVisibleNodesStillAndPansOffscreenNodesWithoutChangingScale() {
        let viewport=CGSize(width:600,height:400),origin=CGPoint(x:100,y:200)
        XCTAssertEqual(CanvasViewportGeometry.revealOrigin(bounds:CGRect(x:140,y:240,width:100,height:80),zoom:1,origin:origin,viewport:viewport,reservedTop:60),origin)
        for zoom in [0.1,0.5,1.0,1.5] {
            let bounds=CGRect(x:2000,y:1400,width:150,height:84)
            let next=CanvasViewportGeometry.revealOrigin(bounds:bounds,zoom:zoom,origin:.zero,viewport:viewport,reservedTop:60)
            XCTAssertGreaterThanOrEqual(bounds.minX*zoom-next.x,12)
            XCTAssertLessThanOrEqual(bounds.maxX*zoom-next.x,viewport.width-12)
            XCTAssertGreaterThanOrEqual(bounds.minY*zoom+60-next.y,72)
            XCTAssertLessThanOrEqual(bounds.maxY*zoom+60-next.y,viewport.height-12)
        }
        let back=CanvasViewportGeometry.revealOrigin(bounds:CGRect(x:30,y:40,width:100,height:80),zoom:1,origin:CGPoint(x:1000,y:1000),viewport:viewport,reservedTop:60)
        XCTAssertEqual(back,CGPoint(x:18,y:28))
    }
    func testSeventeenLargeCardsFitInShortResultsSplitInBothOrientations() {
        for vertical in [false,true] {
            let nodes=(0..<17).map { index in
                vertical ? CGRect(x:80,y:CGFloat(index)*300,width:240,height:260)
                    : CGRect(x:CGFloat(index)*300,y:80,width:260,height:240)
            }
            for inset:CGFloat in [60,102] {
                let viewport=CGSize(width:1058,height:310)
                let fit=CanvasViewportGeometry.fitAllViewport(nodes:nodes,viewport:viewport,reservedTop:inset)
                XCTAssertTrue(fit.fullyFits)
                XCTAssertGreaterThanOrEqual(fit.zoom,0.01)
                if vertical {XCTAssertLessThan(fit.zoom,0.1)}
                for node in nodes {
                    let screen=node.applying(CGAffineTransform(scaleX:fit.zoom,y:fit.zoom)).offsetBy(dx:-fit.origin.x,dy:inset-fit.origin.y)
                    XCTAssertGreaterThanOrEqual(screen.minX,12-0.000001)
                    XCTAssertGreaterThanOrEqual(screen.minY,inset+12-0.000001)
                    XCTAssertLessThanOrEqual(screen.maxX,viewport.width-12+0.000001)
                    XCTAssertLessThanOrEqual(screen.maxY,viewport.height-12+0.000001)
                }
            }
        }
    }
    func testFitReportsRemainingClippingAtMinimumScaleAndDegenerateViewport() {
        let extreme=[CGRect(x:0,y:0,width:200,height:100_000)]
        let constrained=CanvasViewportGeometry.fitAllViewport(nodes:extreme,viewport:CGSize(width:1058,height:310),reservedTop:102)
        XCTAssertEqual(constrained.zoom,0.01)
        XCTAssertFalse(constrained.fullyFits)
        let enlarged=CanvasViewportGeometry.fitAllViewport(nodes:extreme,viewport:CGSize(width:1058,height:1200),reservedTop:102)
        XCTAssertTrue(enlarged.fullyFits)
        let tiny=CanvasViewportGeometry.fitAllViewport(nodes:[CGRect(x:0,y:0,width:10,height:10)],viewport:CGSize(width:20,height:20),reservedTop:60)
        XCTAssertFalse(tiny.fullyFits)
        XCTAssertTrue(CanvasViewportGeometry.fitAllViewport(nodes:[],viewport:.zero,reservedTop:60).fullyFits)
    }
    func testFitAllIgnoresEmptyOriginSpaceAndKeepsEveryCardVisible() {
        for viewport in [CGSize(width:900,height:500),CGSize(width:400,height:250)] {
        let nodes=[CGRect(x:40,y:30,width:150,height:84),CGRect(x:500,y:220,width:150,height:120)]
        let shifted=nodes.map {$0.offsetBy(dx:2000,dy:1500)}
        let first=CanvasViewportGeometry.fitAllViewport(nodes:nodes,viewport:viewport,reservedTop:60)
        let second=CanvasViewportGeometry.fitAllViewport(nodes:shifted,viewport:viewport,reservedTop:60)
        XCTAssertEqual(first.zoom,second.zoom,accuracy:0.000001)
        XCTAssertLessThanOrEqual(first.zoom,1)
        for node in shifted {
            let screen=node.applying(CGAffineTransform(scaleX:second.zoom,y:second.zoom)).offsetBy(dx:-second.origin.x,dy:60-second.origin.y)
            XCTAssertGreaterThanOrEqual(screen.minX,12)
            XCTAssertGreaterThanOrEqual(screen.minY,72)
            XCTAssertLessThanOrEqual(screen.maxX,viewport.width-12)
            XCTAssertLessThanOrEqual(screen.maxY,viewport.height-12)
        }
        let empty=CanvasViewportGeometry.fitAllViewport(nodes:[],viewport:viewport,reservedTop:60)
        XCTAssertEqual(empty.zoom,1);XCTAssertEqual(empty.origin,.zero)
        }
    }
    func testEmptyCanvasStartsAtReadableScaleEvenInShortResultsSplit() {
        for viewport in [CGSize(width:600,height:180), CGSize(width:1200,height:700)] {
            XCTAssertEqual(CanvasViewportGeometry.fitAllZoom(nodes:[],viewport:viewport,reservedTop:102),1)
        }
    }
    func testPopulatedCanvasStillFitsNodesAndToolbar() {
        let nodes=[CGRect(x:50,y:30,width:180,height:100),CGRect(x:1800,y:1400,width:180,height:100)]
        let fit=CanvasViewportGeometry.fitAllViewport(nodes:nodes,viewport:CGSize(width:600,height:500),reservedTop:102)
        XCTAssertLessThan(fit.zoom,1)
        for node in nodes {
            XCTAssertGreaterThanOrEqual(node.minX*fit.zoom-fit.origin.x,0)
            XCTAssertGreaterThanOrEqual(node.minY*fit.zoom-fit.origin.y,0)
            XCTAssertLessThanOrEqual(node.maxX*fit.zoom-fit.origin.x,588)
            XCTAssertLessThanOrEqual(node.maxY*fit.zoom-fit.origin.y+102,488)
        }
    }
    func testZoomedOutDocumentCoversWholeVisibleCanvasAndPreservesDropCoordinates() {
        let viewport=CGSize(width:1274,height:399)
        let content=CGSize(width:1600,height:900)
        let extent=CanvasViewportGeometry.documentSize(content:content,zoom:0.4,reservedTop:60,viewport:viewport)
        XCTAssertEqual(extent.width,1274);XCTAssertEqual(extent.height,420)
        let drop=CGPoint(x:870,y:186)
        XCTAssertTrue(CGRect(origin:.zero,size:extent).contains(drop))
        XCTAssertEqual(CanvasViewportGeometry.graphLocation(documentPoint:drop,zoom:0.4,reservedTop:60),CGPoint(x:2175,y:315))
        let graph=CanvasViewportGeometry.graphExtent(content:content,zoom:0.4,reservedTop:60,viewport:viewport)
        XCTAssertEqual(graph.width*0.4,extent.width,accuracy:0.00001)
        XCTAssertEqual(graph.height*0.4+60,extent.height,accuracy:0.00001)
        let expanded=CanvasViewportGeometry.documentSize(content:content,zoom:0.1,reservedTop:60,viewport:viewport)
        XCTAssertEqual(expanded,viewport)
    }
    func testLargeFlowFitCanUseLessThanTwentyFivePercent() {
        let bounds=CGRect(x:0,y:0,width:3000,height:2400)
        let zoom=CanvasViewportGeometry.fitZoom(bounds:bounds,viewport:CGSize(width:600,height:500),reservedTop:60,maximum:1)
        XCTAssertLessThan(zoom,0.25)
        XCTAssertGreaterThanOrEqual(zoom,0.1)
        XCTAssertLessThanOrEqual(bounds.width*zoom,576)
        XCTAssertLessThanOrEqual(bounds.height*zoom+60,476)
    }
    @MainActor func testMeasuredToolbarRetargetsPendingNativeRestoreWithoutLosingOrigin() {
        let scroll=NSScrollView(frame:NSRect(x:0,y:0,width:600,height:400))
        let document=NSView(frame:NSRect(x:0,y:0,width:1600,height:1002))
        scroll.documentView=document
        let controller=CanvasScrollController()
        let requested=CGPoint(x:120,y:150)
        controller.requestRestore(to:requested,document:CGSize(width:1600,height:960))
        controller.attach(scroll)
        XCTAssertTrue(controller.hasPendingRestore)
        controller.retargetPendingRestore(document:CGSize(width:1600,height:1002))
        controller.restoreWhenReady()
        XCTAssertFalse(controller.hasPendingRestore)
        XCTAssertEqual(controller.origin?.x,requested.x)
        XCTAssertEqual(controller.origin?.y,requested.y)
    }
    func testFitReservesMeasuredStackedToolbarAndKeepsOriginVisible() {
        let inset = CanvasViewportGeometry.topInset(toolbarHeight: 78)
        XCTAssertEqual(inset,102)
        XCTAssertEqual(CanvasViewportGeometry.topInset(toolbarHeight:28),60)
        let zoom = CanvasViewportGeometry.fitZoom(bounds:CGRect(x:0,y:0,width:600,height:1000),viewport:CGSize(width:582,height:500),reservedTop:inset,maximum:1)
        XCTAssertGreaterThanOrEqual(inset,78+20)
        XCTAssertLessThanOrEqual(inset+1000*zoom,500-24)
    }
    func testMinimapExcludesOverlayAfterScrollingAtEveryScale() {
        for zoom in [0.1,0.25,1,1.5] {
            let viewport=CanvasViewportGeometry.visibleGraphRect(scrollOrigin:CGPoint(x:60,y:240),viewport:CGSize(width:600,height:500),zoom:zoom,reservedTop:102)
            XCTAssertEqual(viewport.minY,240/zoom,accuracy:0.00001)
            XCTAssertEqual(viewport.height,398/zoom,accuracy:0.00001)
            XCTAssertEqual(viewport.minX,60/zoom,accuracy:0.00001)
        }
    }
    func testSelectionAndDropUseSameInsetAtEveryScale() {
        let bounds = CGRect(x:230,y:400,width:150,height:84)
        for zoom in [0.1,0.25,1,1.5] {
            let inset:CGFloat=102
            let origin=CanvasViewportGeometry.selectionOrigin(bounds:bounds,zoom:zoom,reservedTop:inset)
            XCTAssertGreaterThanOrEqual(inset+bounds.minY*zoom-origin.y,inset)
            let drop=CGPoint(x:bounds.minX*zoom,y:inset+bounds.minY*zoom)
            let graph=CanvasViewportGeometry.graphLocation(documentPoint:drop,zoom:zoom,reservedTop:inset)
            XCTAssertEqual(graph.x,bounds.minX,accuracy:0.00001)
            XCTAssertEqual(graph.y,bounds.minY,accuracy:0.00001)
            let extent=CanvasViewportGeometry.documentSize(content:CGSize(width:1600,height:900),zoom:zoom,reservedTop:inset)
            XCTAssertEqual(extent.height,900*zoom+inset)
        }
    }
}
