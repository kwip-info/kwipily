import XCTest
import AppKit
import SwiftUI
@testable import KwipMLApp

@MainActor final class CanvasHostingClipTests: XCTestCase {
    func testHostingCameraAttachmentKeepsCoordinateOrientation() throws {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let host=NSHostingView(rootView:ScrollView([.horizontal,.vertical]) {Color.red.frame(width:900,height:800)})
        window.contentView=host;host.layoutSubtreeIfNeeded()
        func find(_ view:NSView)->NSScrollView? {
            if let scroll=view as? NSScrollView {return scroll}
            return view.subviews.lazy.compactMap(find).first
        }
        let scroll=try XCTUnwrap(find(host))
        let original=scroll.contentView
        let document=try XCTUnwrap(scroll.documentView)
        let originalFlip=original.isFlipped
        let docFlip=document.isFlipped
        let originalPoint=document.convert(CGPoint(x:100,y:100),from:host)
        let controller=CanvasScrollController();controller.attach(scroll)
        let replacementPoint=document.convert(CGPoint(x:100,y:100),from:host)
        print("Hosting clip original=\(type(of:original)) flip=\(originalFlip) document=\(type(of:document)) flip=\(docFlip) replacement=\(type(of:scroll.contentView)) flip=\(scroll.contentView.isFlipped) originalPoint=\(originalPoint) replacementPoint=\(replacementPoint)")
        XCTAssertEqual(scroll.contentView.isFlipped,originalFlip)
        XCTAssertEqual(replacementPoint,originalPoint)
    }
    func testOriginalHostingClipClampsBoundsAndKeepsHostOwnership() throws {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let host=NSHostingView(rootView:ScrollView([.horizontal,.vertical]) {Color.red.frame(width:900,height:800)})
        window.contentView=host;host.layoutSubtreeIfNeeded()
        func find(_ view:NSView)->NSScrollView? {
            if let scroll=view as? NSScrollView {return scroll}
            return view.subviews.lazy.compactMap(find).first
        }
        let scroll=try XCTUnwrap(find(host));let clip=scroll.contentView
        let document=try XCTUnwrap(scroll.documentView)
        let size=document.frame.size
        for (origin,atMinimum) in [(CGPoint(x:-400,y:-300),true),(CGPoint(x:4000,y:3000),false)] {
            clip.setBoundsOrigin(origin)
            scroll.reflectScrolledClipView(clip);host.layoutSubtreeIfNeeded()
            XCTAssertTrue(scroll.contentView === clip)
            // Legacy scrollbars occupy viewport space; overlay scrollbars do not.
            // Layout must settle before measuring the actual scrollable extent.
            let expected = atMinimum ? CGPoint.zero : CGPoint(
                x:max(0,size.width-clip.bounds.width),y:max(0,size.height-clip.bounds.height))
            XCTAssertEqual(clip.bounds.origin,expected)
            XCTAssertEqual(document.frame.size,size)
            XCTAssertEqual(document.convert(CGPoint(x:50,y:60),from:clip),CGPoint(x:50,y:60))
        }
    }

    func testLogicalCameraKeepsNativeClipFixedWhilePanningBeyondDocument() throws {
        let window=NSWindow(contentRect:CGRect(x:0,y:0,width:600,height:400),styleMask:[.borderless],backing:.buffered,defer:false)
        let host=NSHostingView(rootView:ScrollView([.horizontal,.vertical]) {Color.red.frame(width:900,height:800)})
        window.contentView=host;host.layoutSubtreeIfNeeded()
        func find(_ view:NSView)->NSScrollView? {
            if let scroll=view as? NSScrollView {return scroll}
            return view.subviews.lazy.compactMap(find).first
        }
        let scroll=try XCTUnwrap(find(host)),clip=scroll.contentView,document=try XCTUnwrap(scroll.documentView)
        let size=document.frame.size
        let controller=CanvasScrollController();controller.attach(scroll)
        XCTAssertTrue(scroll.contentView === clip)
        for point in [CGPoint(x:-700,y:-500),CGPoint(x:9000,y:8000)] {
            controller.navigate(to:point)
            XCTAssertEqual(controller.origin,point)
            XCTAssertEqual(scroll.contentView.bounds.origin,.zero)
            XCTAssertEqual(document.frame.size,size)
        }
    }

}
