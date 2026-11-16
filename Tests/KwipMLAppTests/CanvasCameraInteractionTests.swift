import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

final class CanvasCameraInteractionTests: XCTestCase {
    func testSensitivityScalesWheelResponseAndPreservesPointerAnchor() {
        let origin=CGPoint(x:-80,y:120),pointer=CGPoint(x:300,y:240)
        for precise in [false,true] {
            let slow=CanvasCameraGeometry.zoom(origin:origin,zoom:0.5,pointer:pointer,delta:1000,precise:precise,reservedTop:60,sensitivity:0.5)
            let fast=CanvasCameraGeometry.zoom(origin:origin,zoom:0.5,pointer:pointer,delta:1000,precise:precise,reservedTop:60,sensitivity:2)
            XCTAssertLessThan(slow.zoom,fast.zoom)
            XCTAssertLessThanOrEqual(fast.zoom/0.5,1.272)
            XCTAssertEqual((origin.x+pointer.x)/0.5,(fast.origin.x+pointer.x)/fast.zoom,accuracy:0.00001)
            XCTAssertEqual((origin.y+pointer.y-60)/0.5,(slow.origin.y+pointer.y-60)/slow.zoom,accuracy:0.00001)
        }
    }
    func testWheelAxisPolicySeparatesNavigationZoomAndMomentum() {
        XCTAssertEqual(CanvasCameraGeometry.wheelAction(deltaX:30,deltaY:2,precise:true,momentum:false),.pan(30))
        XCTAssertEqual(CanvasCameraGeometry.wheelAction(deltaX:-2,deltaY:0,precise:false,momentum:false),.pan(-24))
        XCTAssertEqual(CanvasCameraGeometry.wheelAction(deltaX:1,deltaY:8,precise:true,momentum:false),.zoom(8))
        XCTAssertEqual(CanvasCameraGeometry.wheelAction(deltaX:0,deltaY:8,precise:true,momentum:true),.none)
        XCTAssertEqual(CanvasCameraGeometry.wheelAction(deltaX:20,deltaY:1,precise:true,momentum:true),.pan(20))
        XCTAssertEqual(CanvasCameraGeometry.wheelAction(deltaX:.nan,deltaY:8,precise:true,momentum:false),.none)
    }
    func testLargeWheelPacketCannotJumpFromOverviewToMaximumZoom() {
        let origin=CGPoint(x:-80,y:120),pointer=CGPoint(x:300,y:240)
        for precise in [false,true] {
            let up=CanvasCameraGeometry.zoom(origin:origin,zoom:0.37,pointer:pointer,delta:1000,precise:precise,reservedTop:60)
            let down=CanvasCameraGeometry.zoom(origin:origin,zoom:0.37,pointer:pointer,delta:-1000,precise:precise,reservedTop:60)
            XCTAssertLessThanOrEqual(up.zoom/0.37,1.13)
            XCTAssertGreaterThanOrEqual(down.zoom/0.37,0.88)
            XCTAssertEqual((origin.x+pointer.x)/0.37,(up.origin.x+pointer.x)/up.zoom,accuracy:0.00001)
            XCTAssertEqual((origin.y+pointer.y-60)/0.37,(up.origin.y+pointer.y-60)/up.zoom,accuracy:0.00001)
        }
    }
    func testWheelZoomKeepsPointUnderPointerAtEveryScale() {
        for zoom in [0.1, 0.65, 1.5] {
            let origin = CGPoint(x: -300, y: 140), pointer = CGPoint(x: 180, y: 240)
            let next = CanvasCameraGeometry.zoom(origin: origin, zoom: zoom, pointer: pointer, delta: 2, precise: false, reservedTop: 60)
            XCTAssertEqual((origin.x + pointer.x) / zoom, (next.origin.x + pointer.x) / next.zoom, accuracy: 0.00001)
            XCTAssertEqual((origin.y + pointer.y - 60) / zoom, (next.origin.y + pointer.y - 60) / next.zoom, accuracy: 0.00001)
            XCTAssertTrue(CanvasZoomLimits.supported.contains(next.zoom))
        }
    }
    func testGridDensityStaysBoundedAndOrientationAnchorStaysOnScreen() {
        for zoom in [0.1,0.3,0.65,1.5] {
            XCTAssertGreaterThanOrEqual(CanvasCameraGeometry.gridStep(zoom:zoom)*zoom,12)
            let old=CGPoint(x:100,y:300),new=CGPoint(x:300,y:100),origin=CGPoint(x:-100,y:120)
            let target=CanvasCameraGeometry.reanchored(origin:origin,oldAnchor:old,newAnchor:new,zoom:zoom)
            XCTAssertEqual(old.x*zoom-origin.x,new.x*zoom-target.x,accuracy:0.00001)
            XCTAssertEqual(old.y*zoom-origin.y,new.y*zoom-target.y,accuracy:0.00001)
        }
    }
    func testMarqueeWorksInEveryDragDirectionAndExtendsCapturedSelection() {
        let bounds = ["a": CGRect(x: 10,y: 10,width: 40,height: 40), "b": CGRect(x: 200,y: 200,width: 40,height: 40)]
        for (a,b) in [(CGPoint.zero, CGPoint(x:60,y:60)), (CGPoint(x:60,y:60), CGPoint.zero)] {
            XCTAssertEqual(CanvasCameraGeometry.selection(start:a,end:b,bounds:bounds,initial:["b"],extending:false), ["a"])
            XCTAssertEqual(CanvasCameraGeometry.selection(start:a,end:b,bounds:bounds,initial:["b"],extending:true), ["a","b"])
        }
    }
    func testSignedCameraRoundTripsWithoutMovingGraphOrLosingValidation() throws {
        let saved = CanvasViewState(zoom:0.65,scrollX:-900,scrollY:-600)
        let restored = CanvasViewportProjection.restored(saved,nodes:[],edges:[])
        XCTAssertEqual(restored,saved)
        var arrangement = WorkspaceViewState(); arrangement.modelCanvas=saved
        try arrangement.validate()
        XCTAssertEqual(try JSONDecoder().decode(CanvasViewState.self,from:JSONEncoder().encode(saved)), saved)
    }
    @MainActor func testClipAllowsNegativeAndBeyondDocumentCameraWithoutResizingDocument() {
        let clip = CanvasInfiniteClipView(frame:CGRect(x:0,y:0,width:300,height:200))
        let document = NSView(frame:CGRect(x:0,y:0,width:400,height:300)); clip.documentView=document
        XCTAssertEqual(clip.constrainBoundsRect(CGRect(x:-700,y:-500,width:300,height:200)).origin, CGPoint(x:-700,y:-500))
        XCTAssertEqual(clip.constrainBoundsRect(CGRect(x:9000,y:8000,width:300,height:200)).origin, CGPoint(x:9000,y:8000))
        XCTAssertEqual(document.frame.size,CGSize(width:400,height:300))
    }
}
