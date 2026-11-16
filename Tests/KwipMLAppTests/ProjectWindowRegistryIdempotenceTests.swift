import XCTest
import KwipMLCore
import AppKit
import Combine
@testable import KwipMLApp

@MainActor final class ProjectWindowRegistryIdempotenceTests:XCTestCase {
    private var root:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}

    func testRepeatedKeyLauncherRegistrationAndSelectionDoNotPublish()throws {
        _ = NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let factory={WorkspaceStore(rootOverride:self.root,storageOverride:directory,detectCapabilities:false)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:factory)
        let session=ProjectWindowSession(workspace:factory())
        session.workspace.appPreferences.restoreCanvasSessionAtLaunch=false
        let window=KeyLauncherTestWindow(contentRect:.zero,styleMask:[.titled],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false
        defer {window.close()}
        registry.registerLauncher(window,session:session)
        XCTAssertTrue(registry.activeSession === session)
        var publications=0
        let observation=registry.objectWillChange.sink {publications += 1}
        defer {observation.cancel()}
        for _ in 0..<20 {
            registry.registerLauncher(window,session:session)
            registry.select(window:window)
        }
        XCTAssertEqual(publications,0,"A launcher render must not publish the same command owner and trigger another render.")
        session.workspace.statusMessage="A real active-session change"
        XCTAssertEqual(publications,1,"The retained subscription still forwards actual active-session changes.")
        registry.select(window:nil)
        XCTAssertNil(registry.activeSession)
        XCTAssertEqual(publications,2)
        registry.select(window:nil)
        session.workspace.statusMessage="Inactive session"
        XCTAssertEqual(publications,2,"Repeated no-owner selection is silent and detached sessions stop forwarding.")
    }

    func testOwnerSwitchRoutesOnlyTheNewSessionAndAttachmentTracksWindow()throws {
        _ = NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let factory={WorkspaceStore(rootOverride:self.root,storageOverride:directory,detectCapabilities:false)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:factory)
        let first=ProjectWindowSession(workspace:factory()),second=ProjectWindowSession(workspace:factory())
        first.workspace.appPreferences.restoreCanvasSessionAtLaunch=false
        second.workspace.appPreferences.restoreCanvasSessionAtLaunch=false
        let a=NSWindow(contentRect:.zero,styleMask:[.titled],backing:.buffered,defer:false)
        let b=NSWindow(contentRect:.zero,styleMask:[.titled],backing:.buffered,defer:false)
        a.isReleasedWhenClosed=false;b.isReleasedWhenClosed=false
        defer {a.close();b.close()}
        let attachment=ProjectLauncherAttachment.Attachment()
        attachment.session=first;attachment.registry=registry
        a.contentView?.addSubview(attachment)
        attachment.register()
        XCTAssertTrue(registry.session(for:a) === first)
        registry.select(window:a)
        var publications=0
        let observation=registry.objectWillChange.sink {publications += 1}
        defer {observation.cancel()}
        attachment.register();attachment.register()
        XCTAssertEqual(publications,0)
        attachment.removeFromSuperview()
        attachment.session=second
        b.contentView?.addSubview(attachment)
        attachment.register()
        XCTAssertTrue(registry.session(for:b) === second)
        registry.select(window:b)
        XCTAssertTrue(registry.activeSession === second)
        let afterSwitch=publications
        first.workspace.statusMessage="Old owner"
        XCTAssertEqual(publications,afterSwitch)
        second.workspace.statusMessage="Current owner"
        XCTAssertEqual(publications,afterSwitch+1)
        registry.select(window:b)
        XCTAssertEqual(publications,afterSwitch+1)
    }
}

@MainActor private final class KeyLauncherTestWindow:NSWindow {
    override var isKeyWindow:Bool {true}
}

extension ProjectWindowRegistryIdempotenceTests {
    func testSavingInCanvasRefreshesWelcomeAndMissingRecentReportsFailure()throws {
        _=NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let factory={WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:factory)
        let launcher=ProjectWindowSession(workspace:factory())
        launcher.workspace.appPreferences.restoreCanvasSessionAtLaunch=false
        let window=NSWindow(contentRect:.zero,styleMask:[.titled],backing:.buffered,defer:false)
        window.isReleasedWhenClosed=false
        defer {window.close()}
        registry.registerLauncher(window,session:launcher)
        let controller=try registry.newFlow(present:false)
        let owner=try XCTUnwrap(controller.session.workspace.workflowSession?.owner)
        XCTAssertTrue(owner.rename("Saved new title",lease:owner.lease()))
        XCTAssertTrue(controller.session.workspace.saveManagedProject())
        XCTAssertEqual(launcher.workspace.recentProjects.first?.title,"Saved new title")
        let missing=ProjectRecord(id:UUID().uuidString,name:"Missing",path:"/missing.kwipml",updatedAt:Date())
        launcher.workspace.openManagedProject(missing)
        XCTAssertNotNil(registry.navigationError)
        XCTAssertEqual(registry.projectCount,1,"Failed opening must not register a blank owner")
        XCTAssertTrue(registry.close(controller))
    }
}
