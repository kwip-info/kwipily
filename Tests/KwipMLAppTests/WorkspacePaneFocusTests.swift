import XCTest
import AppKit
import Combine
@testable import KwipMLApp

@MainActor final class WorkspacePaneFocusTests:XCTestCase {
    func testRoutesOnlyOwnVisiblePaneAndNeverChangesResponder()throws {
        let window=NSWindow(contentRect:.init(x:0,y:0,width:800,height:600),styleMask:[.titled],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false
        let root=NSView(frame:.init(x:0,y:0,width:800,height:600));window.contentView=root
        defer {window.close()}
        let focus=WorkspacePaneFocus(),left=NSView(frame:.init(x:0,y:0,width:200,height:600)),right=NSView(frame:.init(x:600,y:0,width:200,height:600))
        root.addSubview(left);root.addSubview(right)
        let a=UUID(),b=UUID();focus.register(a,view:left,pane:.inspector);focus.register(b,view:right,pane:.lesson)
        defer {focus.unregister(a);focus.unregister(b)}
        let responder=window.firstResponder
        let click=try XCTUnwrap(NSEvent.mouseEvent(with:.leftMouseDown,location:.init(x:700,y:300),modifierFlags:[],timestamp:0,windowNumber:window.windowNumber,context:nil,eventNumber:1,clickCount:1,pressure:1))
        focus.observe(click)
        XCTAssertEqual(focus.active,.lesson)
        XCTAssertTrue(window.firstResponder===responder,"Focus decoration must not steal text/button input")
        right.isHidden=true
        XCTAssertNil(focus.pane(at:.init(x:700,y:300)))
        focus.unregister(a)
        XCTAssertNil(focus.pane(at:.init(x:100,y:300)))
    }
    func testRepeatedFocusDoesNotPublishOnEveryPointerPacket() {
        let focus=WorkspacePaneFocus();var updates=0
        let observation=focus.objectWillChange.sink {updates+=1}
        for _ in 0..<100 {focus.activate(.results)}
        XCTAssertEqual(updates,1)
        focus.activate(.canvas);XCTAssertEqual(updates,2)
        withExtendedLifetime(observation) {}
    }
}
