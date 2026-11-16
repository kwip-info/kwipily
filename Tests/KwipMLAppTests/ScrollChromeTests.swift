import XCTest
import AppKit
@testable import KwipMLApp

final class ScrollChromeTests:XCTestCase {
    @MainActor func testDataTableKeepsNativeScrollbarsAndPinnedHeader() {
        let root = NSView()
        let scroll = CapturedTableScrollView()
        root.addSubview(scroll)
        let controller = NativeScrollChromeController(root: root)
        controller.refresh()
        XCTAssertTrue(scroll.hasHorizontalScroller)
        XCTAssertTrue(scroll.hasVerticalScroller)
        XCTAssertNotNil((scroll.documentView as? NSTableView)?.headerView)
        XCTAssertEqual((scroll.documentView as? NSTableView)?.columnAutoresizingStyle, .noColumnAutoresizing)
    }
    @MainActor func testHostedScrollHierarchyIsFoundAndLaterNativeResetIsSuppressed() async throws {
        let root=NSView(frame:NSRect(x:0,y:0,width:300,height:200))
        let container=NSView(frame:root.bounds);root.addSubview(container)
        let scroll=NSScrollView(frame:root.bounds);container.addSubview(scroll)
        scroll.hasVerticalScroller=true;scroll.hasHorizontalScroller=true;scroll.scrollerStyle = .legacy
        let controller=NativeScrollChromeController(root:root)
        controller.refresh()
        XCTAssertFalse(scroll.hasVerticalScroller);XCTAssertFalse(scroll.hasHorizontalScroller)
        XCTAssertEqual(scroll.scrollerStyle,.overlay)
        // SwiftUI/native layout can restore its own preference after initial mounting.
        scroll.hasVerticalScroller=true;scroll.hasHorizontalScroller=true;scroll.scrollerStyle = .legacy
        try await Task.sleep(nanoseconds:30_000_000)
        XCTAssertFalse(scroll.hasVerticalScroller);XCTAssertFalse(scroll.hasHorizontalScroller)
        XCTAssertEqual(scroll.scrollerStyle,.overlay)
        withExtendedLifetime(controller) {}
    }
    @MainActor func testNewlyMountedScrollIsHandledOnRefreshWithoutChangingOtherWindows() {
        let root=NSView(),outside=NSScrollView()
        outside.hasVerticalScroller=true
        let controller=NativeScrollChromeController(root:root)
        let mounted=NSScrollView();mounted.hasVerticalScroller=true;root.addSubview(mounted)
        controller.refresh()
        XCTAssertFalse(mounted.hasVerticalScroller)
        XCTAssertTrue(outside.hasVerticalScroller)
    }
}
