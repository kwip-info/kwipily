import KwipMLCore
import XCTest

@testable import KwipMLApp

final class AffineRecoveryStoreTests: XCTestCase {
  @MainActor func testRecoveryTaskPersistsAndGradesPublicProbeConstructionWithoutCompositionCredit() async throws {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    guard FileManager.default.fileExists(atPath: repository.appendingPathComponent(".venv/bin/python").path)
    else { throw XCTSkip("Runtime unavailable") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
    let lesson = AffineLearningStore(workspace: workspace)
    lesson.start(identity: .recovery(taskSeed: 7), mode: .guided)
    let task = try XCTUnwrap(lesson.lessonTask)
    XCTAssertEqual(task.identity.assessmentID, "assessment.affine-from-probes")
    XCTAssertEqual(task.probes.count, 3)
    lesson.revealHint("probe-zero")
    lesson.setNotes("The zero input reveals the translation; subtract it from each basis output.")
    workspace.edit {
      $0 = GraphDocument(nodes: [
        GraphNode(id: "i", type: "input", config: ["features": 2]),
        GraphNode(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]),
        GraphNode(id: "o", type: "output")
      ], edges: [GraphEdge(source: "i", target: "m"), GraphEdge(source: "m", target: "o")])
    }
    workspace.selectedNodeID = "m"
    lesson.resetParameters(nodeID: "m")
    let zero = try XCTUnwrap(task.probes.first { $0.id == "zero" })
    for row in 0..<2 {
      XCTAssertTrue(lesson.setBias(nodeID: "m", index: row, value: zero.output[row]))
      for column in 0..<2 {
        let probe = try XCTUnwrap(task.probes.first { $0.id == "basis-\(column)" })
        XCTAssertTrue(lesson.setWeight(nodeID: "m", row: row, column: column,
                                      value: probe.output[row] - zero.output[row]))
      }
    }
    lesson.save()
    let restored = AffineLearningStore(workspace: workspace)
    restored.reopenCurrentProject()
    XCTAssertEqual(restored.lessonTask?.identity, task.identity)
    XCTAssertEqual(restored.lessonTask?.probes, task.probes)
    XCTAssertTrue(restored.draft?.hintIDs.contains("probe-zero") == true)
    restored.setMode(.assessment)
    XCTAssertEqual(restored.lessonTask?.probes, task.probes, "Mode changes must not replace the target")
    restored.submit()
    try await finish(workspace)
    XCTAssertEqual(restored.latestGrade?.status, .passed, restored.statusMessage)
    XCTAssertEqual(restored.latestAttempt?.objectiveID, "matrices.affine-from-probes")
    XCTAssertTrue(restored.hasCurrentCompletion)
    restored.setMode(.practice)
    XCTAssertNil(restored.formativeProjection, "Assessment attempt retains its original privacy")
    restored.submit()
    try await finish(workspace)
    XCTAssertEqual(restored.formativeProjection?.fixtures.count, 4, restored.statusMessage)
    let again = AffineLearningStore(workspace: workspace)
    again.reopenCurrentProject()
    XCTAssertTrue(again.latestResultIsCurrent)
    XCTAssertEqual(again.formativeProjection?.fixtures.count, 4)
    // Deliberately return control to the earlier session after checking reopen parity.
    XCTAssertTrue(workspace.learningProjects.attachAffine(restored))
    XCTAssertFalse(again.isActive)
    restored.start(identity: .composition, mode: .guided)
    XCTAssertEqual(restored.lessonTask?.identity, .composition)
    XCTAssertFalse(restored.hasCurrentCompletion, "Recovery cannot award composition completion")
    XCTAssertEqual(restored.statusMessage, "Build the requested function and enter any explicit parameters.")
  }

  @MainActor private func finish(_ workspace: WorkspaceStore) async throws {
    let deadline = Date().addingTimeInterval(25)
    while workspace.isAssessingLearning && Date() < deadline {
      try await Task.sleep(nanoseconds: 25_000_000)
    }
    XCTAssertFalse(workspace.isAssessingLearning)
  }
}
