import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LearningResetSettingsTests: XCTestCase {
    func testPreviewIsScopedAndBusyOrChangedProgressCannotApply() throws {
        _ = NSApplication.shared
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: root, storageOverride: dir, detectCapabilities: false)
        let progress = LearningProgressStore(directory: dir.appendingPathComponent("learning"))
        let plan = LearningExamples.affineComposition(runtimeVersion: "test", seed: 1)
        let grade = LearningGrade(status: .incomplete, summary: "Practice", objectiveID: plan.objectiveID,
            contentVersion: plan.contentVersion, assessmentVersion: plan.assessmentVersion)
        let attempt = LearningAttempt(objectiveID: plan.objectiveID, contentVersion: plan.contentVersion,
            assessmentVersion: plan.assessmentVersion, mode: .guided, grade: grade)
        try progress.append(attempt)
        let sentinel = dir.appendingPathComponent("user-weights.bin")
        try Data("preserve".utf8).write(to: sentinel)
        var blocked = false
        let settings = LearningResetSettingsModel(workspace: workspace, blocked: { blocked })
        settings.refresh()
        XCTAssertEqual(settings.targets.count, 1)
        settings.preview(id: try XCTUnwrap(settings.targets.first?.id))
        XCTAssertEqual(settings.previewAttemptIDs, [attempt.id])
        blocked = true
        XCTAssertFalse(settings.apply())
        XCTAssertEqual(try progress.attempts().count, 1)
        blocked = false
        try progress.append(LearningAttempt(objectiveID: "other", contentVersion: plan.contentVersion,
            assessmentVersion: plan.assessmentVersion, mode: .guided, grade: grade))
        XCTAssertFalse(settings.apply(), "Stale preview must refuse, not reset a newer attempt set")
        settings.refresh(); settings.preview(id: "objective:" + plan.objectiveID)
        XCTAssertTrue(settings.apply())
        XCTAssertEqual(try progress.attempts().map(\.objectiveID), ["other"])
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("preserve".utf8))
        XCTAssertEqual(SettingsPage.matches("learning progress reset"), [.learning])
    }
}
