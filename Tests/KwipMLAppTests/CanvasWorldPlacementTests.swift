import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasWorldPlacementTests:XCTestCase {
    func testNegativeNodesAreInsideSceneWithoutMutatingWorldCoordinates() {
        let original=CGRect(x:-340,y:-220,width:88,height:104)
        let origin=CanvasWorldGeometry.sceneOrigin(nodes:[original,CGRect(x:140,y:300,width:100,height:100)])
        let scene=CanvasWorldGeometry.scenePoint(world:original.origin,origin:origin)
        XCTAssertGreaterThanOrEqual(scene.x,12);XCTAssertGreaterThanOrEqual(scene.y,12)
        XCTAssertEqual(CanvasWorldGeometry.worldPoint(scene:scene,origin:origin),original.origin)
        for zoom in [0.1,0.7,1.5] {
            let camera=CGPoint(x:-500,y:-300)
            XCTAssertEqual(scene.x*zoom+origin.x*zoom-camera.x,original.minX*zoom-camera.x,accuracy:0.000001)
        }
    }
    func testDropAndMovementKeepNegativeWorldLocationAndGroupSpacing() {
        let at=CGPoint(x:-400,y:-280)
        XCTAssertEqual(CanvasInteractionGeometry.boundedOrigin(at),at)
        let moved=CanvasInteractionGeometry.translation(CGSize(width:-200,height:-100),origins:[at,CGPoint(x:60,y:90)])
        XCTAssertEqual(moved,CGSize(width:-200,height:-100))
    }
    func testFitIncludesNegativeWorldBounds() {
        let nodes=[CGRect(x:-500,y:-400,width:88,height:104),CGRect(x:20,y:30,width:150,height:84)]
        let fitted=CanvasViewportGeometry.fitAllViewport(nodes:nodes,viewport:CGSize(width:1200,height:800),reservedTop:60)
        XCTAssertLessThan(fitted.origin.x,0);XCTAssertLessThan(fitted.origin.y,0)
        XCTAssertEqual(nodes[0].minX*fitted.zoom-fitted.origin.x,12,accuracy:0.001)
    }
}
