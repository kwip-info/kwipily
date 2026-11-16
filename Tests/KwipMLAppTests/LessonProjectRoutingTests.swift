import XCTest
@testable import KwipMLApp

final class LessonProjectRoutingTests: XCTestCase {
    @MainActor func testLauncherCodeAndDiagramRoutesPreserveExplicitLessonContext() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start()
        let id = try XCTUnwrap(workspace.activeProjectID)
        let record = try XCTUnwrap(workspace.projects.first { $0.id == id })
        XCTAssertTrue(workspace.isManagedLesson(record))
        for section in ["Code", "Build"] {
            workspace.beginProject(.init(name: "Untitled flow"))
            workspace.openManagedProject(record, section: section)
            XCTAssertTrue(workspace.isAffineLessonActive)
            XCTAssertTrue(lesson.isActive)
            XCTAssertEqual(workspace.activeSection, "Learn")
            workspace.run()
            XCTAssertFalse(workspace.isRunning)
        }
    }
}
