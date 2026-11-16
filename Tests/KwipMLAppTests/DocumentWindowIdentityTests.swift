import XCTest
import AppKit
@testable import KwipMLApp

final class DocumentWindowIdentityTests: XCTestCase {
    func testProjectLessonAndExternalLocationsAreDistinctWithoutGuessingPaths() {
        let url = URL(fileURLWithPath: "/tmp/user-selected/main.kwipflow")
        let project = DocumentWindowIdentity.project(url: url, isLesson: false, isExternal: false, isManaged: true)
        XCTAssertEqual(project.kind, .project)
        XCTAssertEqual(project.location, .managed)
        let lesson = DocumentWindowIdentity.project(url: url, isLesson: true, isExternal: false, isManaged: true)
        XCTAssertEqual(lesson.kind, .lesson)
        XCTAssertNotEqual(project.symbol, lesson.symbol)
        let external = DocumentWindowIdentity.project(url: url, isLesson: false, isExternal: true, isManaged: true)
        XCTAssertEqual(external.location, .external)
        XCTAssertEqual(external.representedURL, url)
        XCTAssertEqual(DocumentWindowIdentity.project(url: nil, isLesson: false, isExternal: false, isManaged: false).location, .unsaved)
    }
    func testSubflowLocationUsesAuthorityRatherThanFileName() {
        let url = URL(fileURLWithPath: "/tmp/models/block.kwipflow")
        let target = CanvasTabSession.Target.subflow(root: .managed(UUID()), path: "models/block.kwipflow", documentID: UUID())
        let managed = DocumentWindowIdentity.block(url: url, restorationTarget: target, hasExternalAccess: false)
        XCTAssertEqual(managed.kind, .block)
        XCTAssertEqual(managed.location, .managed)
        XCTAssertEqual(DocumentWindowIdentity.block(url: url, restorationTarget: nil, hasExternalAccess: false).location, .localFile)
        XCTAssertEqual(DocumentWindowIdentity.block(url: url, restorationTarget: target, hasExternalAccess: true).location, .external)
    }
    @MainActor func testApplyingIdentityPreservesTitleAndExactDocumentProxyURL() {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: .init(x: 0, y: 0, width: 400, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.title = "My model"; window.tab.title = "My model"
        let url = URL(fileURLWithPath: "/tmp/my project/../my project/main.kwipflow")
        let identity = DocumentWindowIdentity.project(url: url, isLesson: false, isExternal: true, isManaged: false)
        identity.apply(to: window)
        XCTAssertEqual(window.representedURL, url)
        XCTAssertEqual(window.title, "My model")
        XCTAssertEqual(window.tab.title, "My model")
        XCTAssertTrue(window.tab.toolTip.contains("External folder"))
        XCTAssertNotNil(window.tab.accessoryView)
        XCTAssertNotNil(window.standardWindowButton(.documentIconButton)?.image)
        window.close()
    }
}
