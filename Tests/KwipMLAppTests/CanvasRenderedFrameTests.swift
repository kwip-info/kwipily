import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

/// Compare camera math with the actual hosted AppKit surface, including titlebar,
/// inset workspace, negative graph coordinates and saved viewport restoration.
@MainActor final class CanvasRenderedFrameTests: XCTestCase {
    func testDerivedFrameMatchesNativeSurfaceAcrossPanZoomAndResize() async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:root,storageOverride:storage,detectCapabilities:false)
        store.graph.nodes=[.init(id:"left",type:"linear",name:"Left test node",x:-180,y:-100),.init(id:"right",type:"linear",name:"Right test node",x:420,y:320)]
        store.graph.edges=[.init(id:"wire",source:"left",sourcePort:"out",target:"right",targetPort:"in")]
        store.canvasViewState = .init(zoom:0.65,scrollX:-60,scrollY:90)
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:800,height:600),styleMask:[.titled],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false
        let host=NSHostingView(rootView:GraphCanvasView(store:store,publishesHistoryFocus:false).padding(.leading,37).padding(.top,23))
        window.contentView=host;window.orderFront(nil)
        defer {window.orderOut(nil);window.close()}
        func settle() async throws {
            host.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(200));host.layoutSubtreeIfNeeded()
            try await Task.sleep(for:.milliseconds(30));host.layoutSubtreeIfNeeded()
        }
        func find(_ view:NSView)->CanvasScrollProbe? {
            if let probe=view as? CanvasScrollProbe {return probe}
            for child in view.subviews {if let probe=find(child) {return probe}}
            return nil
        }
        try await settle()
        let probe=try XCTUnwrap(find(host)),controller=try XCTUnwrap(probe.controller)
        // A live wheel packet must publish scale and its pointer-preserving origin
        // together; waiting for document layout briefly renders the wrong anchor.
        let wheelPointer=CGPoint(x:300,y:240)
        let wheelBefore=store.canvasViewState
        let wheelExpected=CanvasCameraGeometry.zoom(origin:try XCTUnwrap(controller.origin),zoom:wheelBefore.zoom,pointer:wheelPointer,delta:2,precise:false,reservedTop:controller.reservedTop)
        XCTAssertTrue(controller.input.routeWheel(point:wheelPointer,inside:true,deltaX:0,deltaY:2,precise:false,momentum:false))
        XCTAssertEqual(store.canvasViewState.zoom,wheelExpected.zoom,accuracy:0.000001)
        XCTAssertEqual(try XCTUnwrap(controller.origin).x,wheelExpected.origin.x,accuracy:0.000001)
        XCTAssertEqual(try XCTUnwrap(controller.origin).y,wheelExpected.origin.y,accuracy:0.000001)
        XCTAssertFalse(controller.hasPendingRestore)
        try await settle()
        for (zoom,origin,size) in [(0.65,CGPoint(x:-260,y:180),CGSize(width:800,height:600)),
                                   (1.2,CGPoint(x:300,y:-220),CGSize(width:960,height:680)),
                                   (0.3,CGPoint(x:-500,y:-400),CGSize(width:740,height:540))] {
            window.setContentSize(size)
            store.canvasViewState = .init(zoom:zoom,scrollX:origin.x,scrollY:origin.y)
            store.canvasViewRevision=UUID()
            try await settle()
            let clip=try XCTUnwrap(controller.scrollView?.contentView)
            let viewportInWindow=clip.convert(clip.bounds,to:nil)
            let viewport=CGRect(x:viewportInWindow.minX,y:window.frame.height-viewportInWindow.maxY,width:viewportInWindow.width,height:viewportInWindow.height)
            let geometry=CanvasGeometrySnapshot(registry:store.registry,presentation:store.canvasViewState.presentation,graph:store.graph,validation:store.validation)
            let scene=CanvasWorldGeometry.sceneOrigin(nodes:store.graph.nodes.map {CGRect(origin:geometry.projection.project(CGPoint(x:$0.x,y:$0.y)),size:geometry.size(nodeID:$0.id))})
            let native=probe.convert(probe.bounds,to:nil)
            let actual=CGPoint(x:native.minX,y:window.frame.height-native.maxY)
            let expected=CanvasCameraGeometry.contentFrame(viewport:viewport,sceneOrigin:scene,cameraOrigin:origin,zoom:zoom,size:probe.bounds.size).origin
            XCTAssertEqual(controller.origin,origin)
            XCTAssertEqual(actual.x,expected.x,accuracy:1)
            XCTAssertEqual(actual.y,expected.y,accuracy:1)
            // The pure transform must agree with the actual native frame.
            let node=store.graph.nodes[0]
            let graphPoint=CanvasWorldGeometry.scenePoint(world:geometry.projection.project(CGPoint(x:node.x,y:node.y)),origin:scene)
            let screen=CGPoint(x:actual.x-viewport.minX+graphPoint.x*zoom,y:actual.y-viewport.minY+controller.reservedTop+graphPoint.y*zoom)
            let roundTrip=CanvasCameraGeometry.renderedGraphPoint(viewportPoint:screen,viewportFrame:viewport,contentFrame:CGRect(origin:expected,size:probe.bounds.size),zoom:zoom,reservedTop:controller.reservedTop)
            XCTAssertEqual(roundTrip.x,graphPoint.x,accuracy:1)
            XCTAssertEqual(roundTrip.y,graphPoint.y,accuracy:1)
            func checkInstalledHits() async throws {
                let current=try XCTUnwrap(controller.origin)
                let geom=CanvasGeometrySnapshot(registry:store.registry,presentation:store.canvasViewState.presentation,graph:store.graph,validation:store.validation)
                func point(_ node:GraphNode,_ offset:CGPoint)->CGPoint {
                    let world=geom.projection.project(CGPoint(x:node.x,y:node.y))
                    return CGPoint(x:(world.x+offset.x)*zoom-current.x,y:(world.y+offset.y)*zoom-current.y+controller.reservedTop)
                }
                let first=store.graph.nodes[0],last=store.graph.nodes[1],size=geom.size(nodeID:first.id)
                let center=point(first,CGPoint(x:size.width/2,y:size.height/2))
                XCTAssertFalse(controller.input.canMarquee?(center) ?? true)
                let output=point(first,geom.portAnchor(nodeID:first.id,output:true,port:"out"))
                let input=point(last,geom.portAnchor(nodeID:last.id,output:false,port:"in"))
                let middle=CGPoint(x:(output.x+input.x)/2,y:(output.y+input.y)/2)
                XCTAssertTrue(controller.input.onConnectionPress?(middle) ?? false)
                XCTAssertEqual(store.selectedEdgeID,"wire")
                store.clearSelection()
                let margin=30/zoom
                let start=point(first,CGPoint(x:-margin,y:-margin)),end=point(first,CGPoint(x:size.width+margin,y:size.height+margin))
                XCTAssertTrue(controller.input.canMarquee?(start) ?? false)
                controller.input.onMarquee?(start,end,false,true)
                XCTAssertEqual(store.selectedNodeIDs,[first.id])
                store.clearSelection()
            }
            try await checkInstalledHits()
            // Ordinary camera motion does not reinstall the closures via a view revision.
            controller.navigate(to:CGPoint(x:origin.x+73,y:origin.y-41))
            try await settle();try await checkInstalledHits()
            store.graph.nodes[0].x = store.graph.nodes[0].x < 0 ? 40 : -180
            try await settle();try await checkInstalledHits()
        }
    }
}
