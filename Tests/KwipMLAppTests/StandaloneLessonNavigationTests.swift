import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class StandaloneLessonNavigationTests: XCTestCase {
    func testStandaloneProjectsActionSavesWithoutReplacingLessonAndRefusesWhileBusy() throws {
        _ = NSApplication.shared
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: {
            WorkspaceStore(rootOverride: root, storageOverride: dir, detectCapabilities: false)
        })
        let origin = try registry.newFlow(present: false)
        var opened: String?
        registry.reopenLibrary = { opened = $0 }
        origin.session.affine.start()
        let lesson = try XCTUnwrap(registry.openControllers.first { $0.session.affine.isActive })
        XCTAssertEqual(lesson.session.affine.leaveActionTitle, "Projects")
        let lessonID = lesson.session.workspace.activeProjectID
        lesson.session.workspace.isRunning = true
        lesson.session.affine.leave(returnToPreviousProject: true)
        XCTAssertNil(opened)
        lesson.session.workspace.isRunning = false
        lesson.session.affine.leave(returnToPreviousProject: true)
        XCTAssertEqual(opened, "Projects")
        XCTAssertEqual(lesson.session.workspace.activeProjectID, lessonID)
        XCTAssertTrue(lesson.session.affine.isActive)
        XCTAssertNotEqual(origin.session.workspace.activeProjectID, lessonID)
        for controller in registry.openControllers { XCTAssertTrue(registry.close(controller)) }
    }
}
