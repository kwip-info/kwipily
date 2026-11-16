import XCTest
import AppKit
import SwiftUI
import KwipMLCore
@testable import KwipMLApp

@MainActor final class HistoryExpansionLayoutTests:XCTestCase {
    func testExpandedHistoryDisclosuresKeepTablesInsideCompactWindow()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let document=WorkflowDocument(projectID:UUID().uuidString)
        _ = try workspace.projectLibrary.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        try owner.configureCanvasVersions(root:root)
        let host=NSHostingView(rootView:CanvasVersionHistoryView(owner:owner,workspace:workspace,captureExpanded:true,informationExpanded:true))
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:640,height:480),styleMask:[.borderless],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false;window.contentView=host
        defer {window.contentView=nil;window.close()}
        func tables(_ view:NSView)->[CapturedTableContainerView] {
            (view as? CapturedTableContainerView).map {[$0]} ?? view.subviews.flatMap(tables)
        }
        func scrolls(_ view:NSView)->[NSScrollView] {
            ((view as? NSScrollView).map {[$0]} ?? []) + view.subviews.flatMap(scrolls)
        }
        for size in [NSSize(width:640,height:480),NSSize(width:480,height:360),NSSize(width:1000,height:700)] {
            window.setContentSize(size);host.frame=NSRect(origin:.zero,size:size)
            host.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(100));host.layoutSubtreeIfNeeded()
            let native=try XCTUnwrap(tables(host).first)
            let rect=native.convert(native.bounds,to:host)
            XCTAssertGreaterThanOrEqual(rect.minY,-1,"Expanding disclosures must not move tables over the toolbar")
            XCTAssertLessThanOrEqual(rect.maxY,host.bounds.maxY+1,"Table must fit its allocated History viewport")
            XCTAssertLessThanOrEqual(host.frame.height,size.height+1,"Disclosure content must scroll rather than inflate the window")
            if size.width==480 {
                let footer=try XCTUnwrap(scrolls(host).first { !($0 is CapturedTableScrollView) && !$0.isHiddenOrHasHiddenAncestor && ($0.documentView?.bounds.height ?? 0)>$0.contentView.bounds.height+1 })
                let document=try XCTUnwrap(footer.documentView)
                document.scroll(NSPoint(x:0,y:max(0,document.bounds.height-footer.contentView.bounds.height)))
                XCTAssertGreaterThan(footer.contentView.bounds.origin.y,0,"All expanded information must remain reachable by scrolling")
            }
        }
    }
}
