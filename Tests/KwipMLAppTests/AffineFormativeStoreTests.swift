import KwipMLCore
import XCTest

@testable import KwipMLApp

final class AffineFormativeStoreTests: XCTestCase {
  @MainActor func testRealFormativeEvidenceRestoreStalenessAndAssessmentPrivacy() async throws {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    guard FileManager.default.fileExists(atPath: repository.appendingPathComponent(".venv/bin/python").path)
    else { throw XCTSkip("Runtime unavailable") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
    let lesson = AffineLearningStore(workspace: workspace)
    lesson.start(mode: .guided)
    workspace.edit {
      $0 = GraphDocument(nodes: [
        GraphNode(id: "i", type: "input", config: ["features": 2]),
        GraphNode(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]),
        GraphNode(id: "o", type: "output")
      ], edges: [GraphEdge(source: "i", target: "m"), GraphEdge(source: "m", target: "o")])
    }
    workspace.selectedNodeID = "m"
    lesson.resetParameters(nodeID: "m")
    XCTAssertNil(lesson.formativeProjection)
    lesson.submit()
    try await finish(workspace)
    XCTAssertEqual(lesson.latestGrade?.status, .failed, lesson.statusMessage)
    let projection = try XCTUnwrap(lesson.formativeProjection)
    XCTAssertEqual(projection.fixtures.count, 4)
    XCTAssertTrue(projection.fixtures.allSatisfy { $0.measured.values == [0, 0, 0, 0] })
    XCTAssertTrue(projection.fixtures.contains { $0.expected.values.contains { $0 != 0 } })
    XCTAssertFalse(lesson.hasCurrentCompletion)
    lesson.setMode(.assessment)
    XCTAssertNil(lesson.formativeProjection, "Current assessment mode must hide all fixture values")
    lesson.setMode(.practice)
    XCTAssertNotNil(lesson.formativeProjection, "The recorded check is still guided")
    XCTAssertTrue(lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 1))
    XCTAssertNil(lesson.formativeProjection, "Do not attach old numerical tables to edited parameters")
    XCTAssertTrue(lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 0))
    XCTAssertNotNil(lesson.formativeProjection)
    lesson.save()
    let restored = AffineLearningStore(workspace: workspace)
    restored.reopenCurrentProject()
    XCTAssertEqual(restored.formativeProjection?.fixtures, projection.fixtures)
    restored.setMode(.assessment)
    restored.submit()
    try await finish(workspace)
    XCTAssertEqual(restored.latestAttempt?.mode, .assessment)
    XCTAssertNil(restored.formativeProjection)
    restored.setMode(.guided)
    XCTAssertNil(restored.formativeProjection, "Switching modes cannot reveal a recorded assessment")
    restored.save()
    let reopened = AffineLearningStore(workspace: workspace)
    reopened.reopenCurrentProject()
    XCTAssertNil(reopened.formativeProjection, "Privacy follows the persisted attempt mode")
  }

  @MainActor private func finish(_ workspace: WorkspaceStore) async throws {
    let deadline = Date().addingTimeInterval(25)
    while workspace.isAssessingLearning && Date() < deadline {
      try await Task.sleep(nanoseconds: 25_000_000)
    }
    XCTAssertFalse(workspace.isAssessingLearning)
  }
}
