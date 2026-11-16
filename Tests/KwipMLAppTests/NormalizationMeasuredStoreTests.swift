import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Retired DATA normalization is readable history, never a new assessment path.
final class NormalizationMeasuredStoreTests:XCTestCase {
    @MainActor func testHistoricalDraftReopensButDirectSubmissionCannotStartOrRewriteEvidence()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        let lesson=NormalizationLearningStore(workspace:workspace)
        lesson.start(mode:.assessment)
        lesson.setNotes("Retain my historical construction")
        XCTAssertTrue(lesson.save())
        let id=try XCTUnwrap(workspace.activeProjectID), graph=workspace.graph
        let marker=directory.appendingPathComponent("learning/projects/\(id)/normalization.json")
        let before=try Data(contentsOf:marker)
        XCTAssertFalse(lesson.canSubmit)
        lesson.submit()
        XCTAssertTrue(lesson.statusMessage.contains("read-only"))
        XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertTrue(lesson.attempts.isEmpty)
        XCTAssertNil(lesson.latestAttempt)
        XCTAssertEqual(try Data(contentsOf:marker),before)
        XCTAssertTrue(workspace.beginProject(.init(name:"Another project")))
        XCTAssertTrue(workspace.beginProject(graph,id:id))
        XCTAssertEqual(lesson.draft?.notes,"Retain my historical construction")
        XCTAssertNotNil(lesson.flowEditor)
        XCTAssertFalse(lesson.canSubmit)
        lesson.submit()
        XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertEqual(try Data(contentsOf:marker),before)
    }
}
