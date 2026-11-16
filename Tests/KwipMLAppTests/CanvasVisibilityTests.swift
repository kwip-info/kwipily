import XCTest
import CoreGraphics
@testable import KwipMLApp

final class CanvasVisibilityTests: XCTestCase {
    func testOnlyDistantArtworkIsOmittedAndActiveNodesAlwaysPaint() {
        let viewport=CGSize(width:800,height:600),camera=CGPoint(x:200,y:-100)
        let distant=CGRect(x:5000,y:5000,width:256,height:150)
        XCTAssertFalse(CanvasCameraGeometry.shouldPaintNode(worldBounds:distant,cameraOrigin:camera,zoom:0.65,viewport:viewport,reservedTop:60,active:false))
        XCTAssertTrue(CanvasCameraGeometry.shouldPaintNode(worldBounds:distant,cameraOrigin:camera,zoom:0.65,viewport:viewport,reservedTop:60,active:true))
        XCTAssertTrue(CanvasCameraGeometry.shouldPaintNode(worldBounds:distant,cameraOrigin:camera,zoom:0.65,viewport:.zero,reservedTop:60,active:false))
    }
    func testIntersectionUsesFullScaledBoundsAndScreenSpaceOverscan() {
        for zoom in [0.1,0.65,1.5] {
            let camera=CGPoint(x:-400,y:220),viewport=CGSize(width:800,height:600)
            // Origin is far above the viewport, but a tall dynamic-port node reaches it.
            let tall=CGRect(x:(camera.x+200)/zoom,y:(camera.y-800)/zoom,width:256,height:1000/zoom)
            XCTAssertTrue(CanvasCameraGeometry.shouldPaintNode(worldBounds:tall,cameraOrigin:camera,zoom:zoom,viewport:viewport,reservedTop:60,active:false))
            let grazing=CGRect(x:(camera.x+810)/zoom,y:camera.y/zoom,width:256,height:150)
            XCTAssertTrue(CanvasCameraGeometry.shouldPaintNode(worldBounds:grazing,cameraOrigin:camera,zoom:zoom,viewport:viewport,reservedTop:60,active:false))
            let far=CGRect(x:(camera.x+2000)/zoom,y:camera.y/zoom,width:256,height:150)
            XCTAssertFalse(CanvasCameraGeometry.shouldPaintNode(worldBounds:far,cameraOrigin:camera,zoom:zoom,viewport:viewport,reservedTop:60,active:false))
        }
    }
}
