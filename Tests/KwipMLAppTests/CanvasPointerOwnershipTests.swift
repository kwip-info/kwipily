import XCTest
import AppKit
@testable import KwipMLApp

@MainActor final class CanvasPointerOwnershipTests: XCTestCase {
    func testSecondaryPanOwnsDragOverNodesThroughReleaseOutsideViewport() {
        for right in [true,false] {
            let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
            let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
            scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
            let controller=CanvasScrollController();controller.attach(scroll)
            let monitor=controller.input;monitor.controller=controller
            var point=CGPoint(x:100,y:140),connections=0,marquees=0
            monitor.viewportPoint={_,_ in point}
            monitor.hover.set("node:a",hovered:true)
            monitor.onConnectionPress={_ in connections += 1;return true}
            monitor.canMarquee={_ in true};monitor.onMarquee={_,_,_,_ in marquees += 1}
            controller.navigate(to:CGPoint(x:-20,y:80))
            func event(_ type:NSEvent.EventType,_ location:CGPoint,modifiers:NSEvent.ModifierFlags=[])->NSEvent {
                NSEvent.mouseEvent(with:type,location:location,modifierFlags:modifiers,timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1)!
            }
            XCTAssertNil(monitor.routedEvent(event(right ? .rightMouseDown:.leftMouseDown,CGPoint(x:100,y:150),modifiers:right ? []:.control)))
            // Control may be released and the pointer may leave the canvas before mouse-up.
            point=CGPoint(x:700,y:450)
            XCTAssertNil(monitor.routedEvent(event(right ? .rightMouseDragged:.leftMouseDragged,CGPoint(x:140,y:180))))
            XCTAssertEqual(controller.origin,CGPoint(x:-60,y:110))
            let unrelated=event(right ? .leftMouseUp:.rightMouseUp,CGPoint(x:140,y:180))
            XCTAssertTrue(monitor.routedEvent(unrelated) === unrelated)
            XCTAssertNil(monitor.routedEvent(event(right ? .rightMouseDragged:.leftMouseDragged,CGPoint(x:150,y:190))))
            XCTAssertEqual(controller.origin,CGPoint(x:-70,y:120))
            XCTAssertNil(monitor.routedEvent(event(right ? .rightMouseUp:.leftMouseUp,CGPoint(x:150,y:190))))
            let released=event(right ? .rightMouseDragged:.leftMouseDragged,CGPoint(x:200,y:200))
            XCTAssertTrue(monitor.routedEvent(released) === released)
            XCTAssertEqual(connections,0);XCTAssertEqual(marquees,0)
            monitor.stop()
        }
    }
    func testEscapeCancelsSecondaryPanAndSpaceDragStillWorks() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let controller=CanvasScrollController();controller.attach(scroll)
        let monitor=controller.input;monitor.controller=controller
        monitor.viewportPoint={_,_ in CGPoint(x:100,y:140)}
        func event(_ type:NSEvent.EventType,_ x:CGFloat=100)->NSEvent {
            NSEvent.mouseEvent(with:type,location:CGPoint(x:x,y:150),modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1)!
        }
        controller.navigate(to:CGPoint(x:0,y:80))
        XCTAssertNil(monitor.routedEvent(event(.rightMouseDown)))
        let escape=NSEvent.keyEvent(with:.keyDown,location:.zero,modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,characters:"\u{1b}",charactersIgnoringModifiers:"\u{1b}",isARepeat:false,keyCode:53)!
        XCTAssertNil(monitor.routedEvent(escape))
        let drag=event(.rightMouseDragged,140)
        XCTAssertTrue(monitor.routedEvent(drag) === drag)
        XCTAssertEqual(controller.origin,CGPoint(x:0,y:80))
        monitor.keyboardPanAllowed={true}
        let space=NSEvent.keyEvent(with:.keyDown,location:.zero,modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,characters:" ",charactersIgnoringModifiers:" ",isARepeat:false,keyCode:49)!
        XCTAssertNil(monitor.routedEvent(space))
        XCTAssertNil(monitor.routedEvent(event(.leftMouseDown)))
        XCTAssertNil(monitor.routedEvent(event(.leftMouseDragged,140)))
        XCTAssertEqual(controller.origin,CGPoint(x:-40,y:80))
        XCTAssertNil(monitor.routedEvent(event(.leftMouseUp,140)))
        monitor.stop()
    }
    func testSecondaryPanLeavesRibbonAndCanvasChromeNative() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let controller=CanvasScrollController();controller.attach(scroll)
        let monitor=controller.input;monitor.controller=controller
        var point=CGPoint(x:100,y:140);monitor.viewportPoint={_,_ in point}
        for token in ["toolbar","overview",""] {
            monitor.hover.clear()
            if token.isEmpty {point=CGPoint(x:100,y:20)} else {monitor.hover.set(token,hovered:true)}
            for right in [true,false] {
                let press=NSEvent.mouseEvent(with:right ? .rightMouseDown:.leftMouseDown,location:CGPoint(x:100,y:150),modifierFlags:right ? []:.control,timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1)!
                XCTAssertTrue(monitor.routedEvent(press) === press)
            }
        }
        monitor.stop()
    }
    func testWheelOwnershipLeavesRibbonNativePansHorizontallyAndStopsZoomMomentum() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let controller=CanvasScrollController();controller.attach(scroll)
        let monitor=controller.input;monitor.controller=controller
        controller.navigate(to:CGPoint(x:-20,y:80))
        var zooms=0;monitor.onZoom={_,_,_ in zooms += 1}
        let point=CGPoint(x:100,y:140)
        XCTAssertFalse(monitor.routeWheel(point:point,inside:false,deltaX:30,deltaY:0,precise:true,momentum:false))
        XCTAssertEqual(controller.origin,CGPoint(x:-20,y:80))
        XCTAssertTrue(monitor.routeWheel(point:point,inside:true,deltaX:30,deltaY:1,precise:true,momentum:false))
        XCTAssertEqual(controller.origin,CGPoint(x:-50,y:80));XCTAssertEqual(zooms,0)
        XCTAssertTrue(monitor.routeWheel(point:point,inside:true,deltaX:0,deltaY:4,precise:true,momentum:false))
        XCTAssertEqual(zooms,1)
        XCTAssertTrue(monitor.routeWheel(point:point,inside:true,deltaX:0,deltaY:4,precise:true,momentum:true))
        XCTAssertEqual(zooms,1)
        XCTAssertFalse(monitor.routeWheel(point:point,inside:false,deltaX:0,deltaY:4,precise:true,momentum:true))
        monitor.stop()
    }
    func testOwnedMarqueeConsumesPressInsteadOfAlsoDeliveringItToSwiftUI() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let controller=CanvasScrollController();controller.attach(scroll)
        let monitor=controller.input;monitor.controller=controller;monitor.canMarquee={_ in true}
        var starts=0;monitor.onMarquee={_,_,_,_ in starts += 1}
        let event=NSEvent.mouseEvent(with:.leftMouseDown,location:CGPoint(x:100,y:150),modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1)!
        XCTAssertNil(monitor.routedEvent(event))
        XCTAssertEqual(starts,1)
        monitor.stop()
    }
    func testHoverRegionsProtectNodePortsAndWiresBeforeMousePress() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let controller=CanvasScrollController();controller.attach(scroll)
        let monitor=controller.input;monitor.controller=controller;monitor.canMarquee={_ in true}
        var starts=0;monitor.onMarquee={_,_,_,_ in starts += 1}
        let event=NSEvent.mouseEvent(with:.leftMouseDown,location:CGPoint(x:100,y:150),modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1)!
        for token in ["node:a","input:a:x","output:a:y","wire:a-b"] {
            monitor.hover.set(token,hovered:true)
            XCTAssertTrue(monitor.routedEvent(event) === event)
            XCTAssertEqual(starts,0)
            monitor.hover.set(token,hovered:false)
        }
        monitor.hover.set("node:a",hovered:true);monitor.hover.set("input:a:x",hovered:true)
        monitor.hover.set("node:a",hovered:false)
        XCTAssertTrue(monitor.hover.isOccupied)
        monitor.stop()
        XCTAssertFalse(monitor.hover.isOccupied)
    }
    func testRenderedGraphHitUsesActualFrameAndTopLeadingWindowCoordinates() {
        let bounds=CGRect(x:20,y:30,width:900,height:600)
        let flipped=CanvasCameraGeometry.topLeading(CGPoint(x:170,y:230),bounds:bounds,flipped:true)
        let unflipped=CanvasCameraGeometry.topLeading(CGPoint(x:170,y:430),bounds:bounds,flipped:false)
        XCTAssertEqual(flipped,CGPoint(x:150,y:200));XCTAssertEqual(unflipped,flipped)
        let viewport=CGRect(x:120,y:80,width:800,height:400)
        for zoom in [0.1,0.65,1.5] {
            let content=CGRect(x:-180,y:130,width:1200,height:800)
            let graph=CanvasCameraGeometry.renderedGraphPoint(viewportPoint:CGPoint(x:150,y:200),viewportFrame:viewport,contentFrame:content,zoom:zoom,reservedTop:40)
            XCTAssertEqual(graph.x*zoom,450,accuracy:0.0001)
            XCTAssertEqual(graph.y*zoom,110,accuracy:0.0001)
        }
    }

    func testTitledWindowViewportConversionIncludesNativeTitlebarOffset() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.titled],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let titlebar=window.frame.height-scroll.bounds.height
        let viewport=CGRect(x:0,y:titlebar,width:600,height:400)
        let point=CanvasCameraGeometry.viewportPoint(windowPoint:CGPoint(x:100,y:300),window:window,viewportFrame:viewport,clip:scroll.contentView)
        XCTAssertEqual(point,CGPoint(x:100,y:100))
    }

    func testConnectionOwnsOnlyItsPressDragAndReleaseBeforeMarquee() {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let scroll=NSScrollView(frame:CGRect(x:0,y:0,width:600,height:400))
        scroll.documentView=NSView(frame:CGRect(x:0,y:0,width:900,height:600));window.contentView=scroll
        let controller=CanvasScrollController();controller.attach(scroll)
        let monitor=controller.input;monitor.controller=controller;monitor.canMarquee={_ in true}
        var connectionPresses=0,marquees=0
        monitor.onConnectionPress={_ in connectionPresses += 1;return true}
        monitor.onMarquee={_,_,_,_ in marquees += 1}
        func event(_ type:NSEvent.EventType)->NSEvent {
            NSEvent.mouseEvent(with:type,location:CGPoint(x:100,y:150),modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1)!
        }
        XCTAssertNil(monitor.routedEvent(event(.leftMouseDown)))
        XCTAssertEqual(connectionPresses,1);XCTAssertEqual(marquees,0)
        XCTAssertNil(monitor.routedEvent(event(.leftMouseDragged)))
        XCTAssertNil(monitor.routedEvent(event(.leftMouseUp)))
        let unrelated=event(.leftMouseUp)
        XCTAssertTrue(monitor.routedEvent(unrelated) === unrelated)
        monitor.stop()
    }

}
