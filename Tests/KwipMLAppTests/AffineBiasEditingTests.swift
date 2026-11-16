import KwipMLCore
import XCTest

@testable import KwipMLApp

final class AffineBiasEditingTests: XCTestCase {
  @MainActor func testBiasTogglePreservesDormantValuesAllowsWeightsAndInvalidatesOldEditors() throws
  {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let workspace = WorkspaceStore(
      rootOverride: root, storageOverride: root, detectCapabilities: false)
    let lesson = AffineLearningStore(workspace: workspace)
    lesson.start()
    workspace.edit {
      $0 = GraphDocument(
        nodes: [
          GraphNode(id: "i", type: "input", config: ["features": 2]),
          GraphNode(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]),
          GraphNode(id: "o", type: "output"),
        ], edges: [GraphEdge(source: "i", target: "m"), GraphEdge(source: "m", target: "o")])
    }
    workspace.selectNode("m")
    lesson.resetParameters(nodeID: "m")
    XCTAssertTrue(lesson.setBias(nodeID: "m", index: 0, value: 7))
    let oldRevision = lesson.parameterRevision
    XCTAssertTrue(workspace.setConfigValue("bias", value: .bool(false), nodeID: "m"))
    XCTAssertNotEqual(lesson.parameterRevision, oldRevision)
    XCTAssertTrue(lesson.parameterIssues.isEmpty, lesson.parameterIssues.joined())
    XCTAssertTrue(
      lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 3, revision: lesson.parameterRevision)
    )
    XCTAssertFalse(lesson.setBias(nodeID: "m", index: 0, value: 8))
    XCTAssertEqual(lesson.parameters(for: "m")?.bias.values.first, 7)
    XCTAssertTrue(workspace.setConfigValue("bias", value: .bool(true), nodeID: "m"))
    XCTAssertFalse(
      lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 99, revision: oldRevision))
    XCTAssertEqual(lesson.parameters(for: "m")?.weight.values.first, 3)
    XCTAssertEqual(lesson.parameters(for: "m")?.bias.values.first, 7)
    workspace.undo()
    XCTAssertEqual(workspace.selectedNode?.config["bias"], .bool(false))
    lesson.save()
    let reopened = AffineLearningStore(workspace: workspace)
    reopened.reopenCurrentProject()
    XCTAssertEqual(reopened.parameters(for: "m")?.bias.values.first, 7)
    XCTAssertEqual(reopened.parameters(for: "m")?.weight.values.first, 3)
    XCTAssertTrue(reopened.parameterIssues.isEmpty, reopened.parameterIssues.joined())
  }

  @MainActor func testMixedBiasAssessmentUsesActiveParametersAndVersionedReceipt() async throws {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    guard
      FileManager.default.fileExists(
        atPath: repository.appendingPathComponent(".venv/bin/python").path)
    else { throw XCTSkip("Runtime unavailable") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let workspace = WorkspaceStore(
      rootOverride: repository, storageOverride: root, detectCapabilities: false)
    let lesson = AffineLearningStore(workspace: workspace)
    lesson.start(mode: .assessment)
    workspace.edit {
      $0 = GraphDocument(
        nodes: [
          GraphNode(id: "i", type: "input", config: ["features": 2]),
          GraphNode(
            id: "identity", type: "linear",
            config: ["in_features": 2, "out_features": 2, "bias": false]),
          GraphNode(id: "affine", type: "linear", config: ["in_features": 2, "out_features": 2]),
          GraphNode(id: "o", type: "output"),
        ],
        edges: [
          GraphEdge(source: "i", target: "identity"),
          GraphEdge(source: "identity", target: "affine"), GraphEdge(source: "affine", target: "o"),
        ])
    }
    for (id, weights) in [("identity", [1.0, 0, 0, 1]), ("affine", [3.0, 5, -3.5, 5])] {
      workspace.selectNode(id)
      lesson.resetParameters(nodeID: id)
      for (index, value) in weights.enumerated() {
        XCTAssertTrue(lesson.setWeight(nodeID: id, row: index / 2, column: index % 2, value: value))
      }
    }
    XCTAssertTrue(lesson.setBias(nodeID: "affine", index: 0, value: -2.75))
    XCTAssertTrue(lesson.setBias(nodeID: "affine", index: 1, value: -3))
    XCTAssertTrue(lesson.canSubmit, lesson.parameterIssues.joined())
    lesson.submit()
    try await finish(workspace)
    XCTAssertEqual(lesson.latestGrade?.status, .passed, lesson.statusMessage)
    XCTAssertEqual(lesson.latestAttempt?.assessmentVersion, ContractVersion(1, 1, 0))
    XCTAssertTrue(lesson.hasCurrentCompletion)
    let restored = AffineLearningStore(workspace: workspace)
    restored.reopenCurrentProject()
    XCTAssertTrue(restored.resultSaved, restored.statusMessage)
    XCTAssertTrue(restored.latestResultIsCurrent)
    XCTAssertTrue(workspace.setConfigValue("bias", value: .bool(false), nodeID: "affine"))
    XCTAssertTrue(restored.canSubmit, restored.parameterIssues.joined())
    restored.submit()
    try await finish(workspace)
    XCTAssertEqual(restored.latestGrade?.status, .failed, restored.statusMessage)
    XCTAssertTrue(restored.resultSaved)
    XCTAssertEqual(restored.parameters(for: "affine")?.bias.values, [-2.75, -3])
  }
  @MainActor private func finish(_ workspace: WorkspaceStore) async throws {
    let deadline = Date().addingTimeInterval(25)
    while workspace.isAssessingLearning && Date() < deadline {
      try await Task.sleep(nanoseconds: 25_000_000)
    }
    XCTAssertFalse(workspace.isAssessingLearning)
  }
}
