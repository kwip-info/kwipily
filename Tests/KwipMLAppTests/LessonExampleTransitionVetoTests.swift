import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LessonExampleTransitionVetoTests:XCTestCase {
    @MainActor private func protectedLesson(_ directory:URL)throws->(WorkspaceStore,NormalizationLearningStore) {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
        let lesson=NormalizationLearningStore(workspace:workspace);lesson.start()
        let id=try XCTUnwrap(workspace.activeProjectID)
        let marker=directory.appendingPathComponent("learning/projects/\(id)/normalization.json")
        try FileManager.default.removeItem(at:marker)
        try FileManager.default.createDirectory(at:marker,withIntermediateDirectories:false)
        return (workspace,lesson)
    }
    @MainActor func testOpenExamplePreservesFailedSaveDiagnosticAndLesson() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:directory)}
        let (workspace,lesson)=try protectedLesson(directory)
        let id=workspace.activeProjectID,graph=workspace.graph,draft=lesson.draft
        XCTAssertFalse(workspace.loadExample())
        XCTAssertEqual(workspace.activeProjectID,id);XCTAssertEqual(workspace.graph,graph);XCTAssertEqual(lesson.draft,draft)
        XCTAssertTrue(workspace.isNormalizationLessonActive)
        XCTAssertTrue(workspace.statusMessage.contains("save failed"),workspace.statusMessage)
        XCTAssertFalse(workspace.statusMessage.contains("classifier loaded"))
    }
    @MainActor func testStartGuidedCourseDoesNotApplyTutorialStateAfterSaveVeto() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:directory)}
        let (workspace,lesson)=try protectedLesson(directory)
        let id=workspace.activeProjectID,graph=workspace.graph,draft=lesson.draft
        workspace.showPreview=true
        XCTAssertNil(workspace.tutorialStep)
        workspace.startCourse()
        XCTAssertEqual(workspace.activeProjectID,id);XCTAssertEqual(workspace.graph,graph);XCTAssertEqual(lesson.draft,draft)
        XCTAssertNil(workspace.tutorialStep);XCTAssertTrue(workspace.showPreview)
        XCTAssertTrue(workspace.isNormalizationLessonActive)
        XCTAssertTrue(workspace.statusMessage.contains("save failed"),workspace.statusMessage)
    }
}
