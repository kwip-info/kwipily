import XCTest
import AppKit
import SwiftUI
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTemplateLayoutTests:XCTestCase {
    func testTemplateBannerDoesNotInflateWorkbenchBeyondWindow()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let document=try WorkflowTemplates.instantiate(id:"rental-planning")
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let session=try XCTUnwrap(workspace.workflowSession)
        let host=NSHostingView(rootView:WorkflowWorkbenchView(owner:session.owner,canvas:session.canvas,sources:session.sourceProvider,outputs:session,executionHost:session,workspace:workspace))
        let window=NSWindow(contentRect:NSRect(x:0,y:0,width:1200,height:900),styleMask:[.borderless],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false;window.contentView=host
        defer {window.contentView=nil;window.close()}
        for size in [NSSize(width:1200,height:900),NSSize(width:1000,height:760)] {
            window.setContentSize(size);host.frame=NSRect(origin:.zero,size:size)
            host.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(150));host.layoutSubtreeIfNeeded()
            XCTAssertLessThanOrEqual(host.fittingSize.height,size.height+1,"Setup must scroll internally instead of pushing the toolbar above the window")
            XCTAssertLessThanOrEqual(host.frame.height,size.height+1)
        }
    }

    func testSetupPopupReusesOwnerAndResizesWithoutChangingTheWorkflow()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let document=try WorkflowTemplates.instantiate(id:"rental-planning")
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let owner=try XCTUnwrap(workspace.workflowSession?.owner)
        let before=try WorkflowCodec.encode(owner.document)
        let navigation=ApplicationNavigationWindows(presentsWindows:false)
        navigation.showTemplateSetup(workspace:workspace)
        let window=try XCTUnwrap(navigation.templateSetupWindow(sessionID:owner.sessionID))
        defer {window.close()}
        try await Task.sleep(for:.milliseconds(150))
        XCTAssertLessThanOrEqual(window.contentLayoutRect.height,900,"Opening setup must not expand the native window to the scroll content height")
        XCTAssertEqual(window.tabbingMode,.disallowed)
        XCTAssertTrue(window.styleMask.contains(.resizable))
        navigation.showTemplateSetup(workspace:workspace)
        XCTAssertTrue(navigation.templateSetupWindow(sessionID:owner.sessionID) === window)
        let host=try XCTUnwrap(window.contentView)
        for size in [NSSize(width:940,height:730),NSSize(width:800,height:640)] {
            window.setContentSize(size);host.frame=NSRect(origin:.zero,size:size)
            host.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(150));host.layoutSubtreeIfNeeded()
            XCTAssertLessThanOrEqual(host.fittingSize.height,size.height+1)
        }
        XCTAssertEqual(try WorkflowCodec.encode(owner.document),before)
        XCTAssertEqual(workspace.activeProjectID,document.projectID)
        navigation.closeTemplateSetup(sessionID:owner.sessionID)
        XCTAssertNil(navigation.templateSetupWindow(sessionID:owner.sessionID))
    }
}
