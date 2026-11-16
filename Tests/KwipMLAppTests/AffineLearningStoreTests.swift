import KwipMLCore
import XCTest

@testable import KwipMLApp

final class AffineLearningStoreTests: XCTestCase {
  @MainActor func testManagedLessonRequiresExplicitParametersAndRejectsOldEditorRevision() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false)
    let lesson = AffineLearningStore(workspace: workspace)
    lesson.start()
    XCTAssertTrue(lesson.isActive)
    XCTAssertFalse(lesson.canSubmit)
    workspace.edit {
      $0 = GraphDocument(
        nodes: [
          GraphNode(id: "i", type: "input", config: ["features": 2]),
          GraphNode(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]),
          GraphNode(id: "o", type: "output"),
        ], edges: [GraphEdge(source: "i", target: "m"), GraphEdge(source: "m", target: "o")])
    }
    workspace.selectedNodeID = "m"
    lesson.resetParameters(nodeID: "m")
    let revision = lesson.parameterRevision
    lesson.resetParameters(nodeID: "m")
    XCTAssertFalse(lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 3, revision: revision))
    XCTAssertTrue(lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 3))
    lesson.save()
    let reopened = AffineLearningStore(workspace: workspace)
    reopened.reopenCurrentProject()
    XCTAssertEqual(reopened.parameters(for: "m")?.weight.values.first, 3)
    workspace.run()
    XCTAssertFalse(workspace.isRunning)
  }
  @MainActor func testActualAssessmentReceiptReopenFailureAndCancellation() async throws {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    guard
      FileManager.default.fileExists(
        atPath: repository.appendingPathComponent(".venv/bin/python").path)
    else { throw XCTSkip("Runtime unavailable") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(
      rootOverride: repository, storageOverride: directory, detectCapabilities: false)
    let lesson = AffineLearningStore(workspace: workspace)
    lesson.start(mode: .assessment)
    let id = try XCTUnwrap(workspace.activeProjectID)
    workspace.edit {
      $0 = GraphDocument(
        nodes: [
          GraphNode(id: "i", type: "input", config: ["features": 2]),
          GraphNode(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]),
          GraphNode(id: "o", type: "output"),
        ], edges: [GraphEdge(source: "i", target: "m"), GraphEdge(source: "m", target: "o")])
    }
    workspace.selectedNodeID = "m"
    lesson.resetParameters(nodeID: "m")
    for (i, value) in [3.0, 5, -3.5, 5].enumerated() {
      XCTAssertTrue(lesson.setWeight(nodeID: "m", row: i / 2, column: i % 2, value: value))
    }
    XCTAssertTrue(lesson.setBias(nodeID: "m", index: 0, value: -2.75))
    XCTAssertTrue(lesson.setBias(nodeID: "m", index: 1, value: -3))
    lesson.setNumericEditPending(id: "test", pending: true)
    XCTAssertFalse(lesson.canSubmit)
    lesson.setNumericEditPending(id: "test", pending: false)
    lesson.submit()
    try await finish(workspace)
    XCTAssertEqual(lesson.latestGrade?.status, .passed, lesson.statusMessage)
    XCTAssertTrue(lesson.resultSaved, lesson.statusMessage)
    XCTAssertTrue(lesson.hasCurrentCompletion)
    XCTAssertTrue(lesson.latestResultIsCurrent)
    let reopened = AffineLearningStore(workspace: workspace)
    reopened.reopenCurrentProject()
    XCTAssertEqual(reopened.latestGrade?.status, .passed, reopened.statusMessage)
    let latest = directory.appendingPathComponent("learning/projects/\(id)/latest.json")
    let attempt = try XCTUnwrap(reopened.latestAttempt)
    let immutable = directory.appendingPathComponent("learning/attempts/\(attempt.id).json")
    let originalReceipt = try Data(contentsOf: immutable)
    var tampered = try XCTUnwrap(
      JSONSerialization.jsonObject(with: originalReceipt) as? [String: Any])
    tampered["fixtureDigest"] = String(repeating: "0", count: 64)
    try JSONSerialization.data(withJSONObject: tampered).write(to: latest)
    let restoredImmutable = AffineLearningStore(workspace: workspace)
    restoredImmutable.reopenCurrentProject()
    XCTAssertTrue(
      restoredImmutable.resultSaved,
      "Untrusted convenience copy cannot replace the immutable receipt")
    var runtime = try XCTUnwrap(tampered["runtime"] as? [String: Any])
    runtime["version"] = "fabricated-version"
    tampered["runtime"] = runtime
    try JSONSerialization.data(withJSONObject: tampered).write(to: immutable)
    let rejected = AffineLearningStore(workspace: workspace)
    rejected.reopenCurrentProject()
    XCTAssertFalse(rejected.resultSaved)
    XCTAssertNil(rejected.latestGrade)
    try originalReceipt.write(to: immutable)
    try originalReceipt.write(to: latest)
    workspace.learningProjects.attachAffine(reopened)
    XCTAssertTrue(reopened.setBias(nodeID: "m", index: 0, value: 0))
    XCTAssertFalse(reopened.latestResultIsCurrent)
    reopened.submit()
    try await finish(workspace)
    XCTAssertEqual(reopened.latestGrade?.status, .failed, reopened.statusMessage)
    let attemptDirectory = directory.appendingPathComponent("learning/attempts")
    let progressURL = directory.appendingPathComponent("learning/progress.json")
    let receiptNamesBeforeCancel = try FileManager.default.contentsOfDirectory(atPath: attemptDirectory.path).sorted()
    let receiptsBeforeCancel = try Dictionary(uniqueKeysWithValues: receiptNamesBeforeCancel.map {
      ($0, try Data(contentsOf: attemptDirectory.appendingPathComponent($0)))
    })
    let progressBeforeCancel = try Data(contentsOf: progressURL)
    let latestBeforeCancel = try Data(contentsOf: latest)
    let attemptsBeforeCancel = reopened.attempts
    XCTAssertTrue(reopened.hasCurrentCompletion, "The earlier measured assessment pass survives a later numerical failure")
    reopened.submit()
    workspace.cancelRun()
    try await finish(workspace)
    XCTAssertFalse(workspace.isAssessingLearning)
    XCTAssertEqual(reopened.latestGrade?.status, .cancelled, reopened.statusMessage)
    XCTAssertFalse(reopened.resultSaved)
    XCTAssertNil(reopened.latestAttempt, "Cancellation must not create a measured receipt")
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: attemptDirectory.path).sorted(), receiptNamesBeforeCancel)
    for (name, bytes) in receiptsBeforeCancel {
      XCTAssertEqual(try Data(contentsOf: attemptDirectory.appendingPathComponent(name)), bytes)
    }
    XCTAssertEqual(try Data(contentsOf: progressURL), progressBeforeCancel)
    XCTAssertEqual(try Data(contentsOf: latest), latestBeforeCancel)
    XCTAssertEqual(reopened.attempts, attemptsBeforeCancel)
    XCTAssertTrue(reopened.hasCurrentCompletion, "Cancellation cannot erase the prior passing completion")
    let original = try Data(
      contentsOf: directory.appendingPathComponent("learning/projects/\(id)/draft.json"))
    workspace.beginProject(.init(name: "Untitled flow"))
    await Task.yield()
    XCTAssertEqual(
      try Data(contentsOf: directory.appendingPathComponent("learning/projects/\(id)/draft.json")),
      original)
  }

  @MainActor func testAlternativeTwoStageMapAndProgressWriteFailureAreDistinct() async throws {
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    guard
      FileManager.default.fileExists(
        atPath: repository.appendingPathComponent(".venv/bin/python").path)
    else { throw XCTSkip("Runtime unavailable") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(
      rootOverride: repository, storageOverride: directory, detectCapabilities: false)
    let active = AffineLearningStore(workspace: workspace)
    active.start(mode: .practice)
    workspace.edit {
      $0 = GraphDocument(
        nodes: [
          GraphNode(id: "i", type: "input", config: ["features": 2]),
          GraphNode(id: "a", type: "linear", config: ["in_features": 2, "out_features": 2]),
          GraphNode(id: "b", type: "linear", config: ["in_features": 2, "out_features": 2]),
          GraphNode(id: "o", type: "output"),
        ],
        edges: [
          GraphEdge(source: "i", target: "a"), GraphEdge(source: "a", target: "b"),
          GraphEdge(source: "b", target: "o"),
        ])
    }
    for (id, weights, bias) in [
      ("a", [2.0, -1, 0.5, 3], [1.0, -2]), ("b", [1.0, 2, -2, 1], [0.25, 1]),
    ] {
      workspace.selectedNodeID = id
      active.resetParameters(nodeID: id)
      for (i, value) in weights.enumerated() {
        XCTAssertTrue(active.setWeight(nodeID: id, row: i / 2, column: i % 2, value: value))
      }
      for (i, value) in bias.enumerated() {
        XCTAssertTrue(active.setBias(nodeID: id, index: i, value: value))
      }
    }
    active.submit()
    try await finish(workspace)
    XCTAssertEqual(active.latestGrade?.status, .passed, active.statusMessage)
    XCTAssertFalse(active.hasCurrentCompletion, "Practice is not assessment completion")
    let latest = directory.appendingPathComponent(
      "learning/projects/\(try XCTUnwrap(workspace.activeProjectID))/latest.json")
    try FileManager.default.removeItem(at: latest)
    try FileManager.default.createDirectory(at: latest, withIntermediateDirectories: true)
    active.setMode(.assessment)
    active.submit()
    try await finish(workspace)
    XCTAssertTrue(active.resultSaved, active.statusMessage)
    XCTAssertTrue(active.hasCurrentCompletion)
    XCTAssertTrue(active.statusMessage.contains("latest"))
    let restored = AffineLearningStore(workspace: workspace)
    restored.reopenCurrentProject()
    XCTAssertTrue(restored.resultSaved, restored.statusMessage)
    XCTAssertEqual(restored.latestGrade?.status, .passed)

    workspace.learningProjects.attachAffine(active)
    let progress = directory.appendingPathComponent("learning/progress.json")
    try Data("corrupt archive".utf8).write(to: progress)
    active.setMode(.assessment)
    active.submit()
    try await finish(workspace)
    XCTAssertEqual(active.latestGrade?.status, .passed, active.statusMessage)
    XCTAssertFalse(active.resultSaved)
    XCTAssertTrue(active.statusMessage.contains("Evaluated but not saved"))
    XCTAssertEqual(try String(contentsOf: progress), "corrupt archive")
  }

  @MainActor func testMissingReturnProjectCannotTurnLessonIntoOrdinaryTraining() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(rootOverride: directory, detectCapabilities: false)
    let active = AffineLearningStore(workspace: workspace)
    active.start()
    let draft = try XCTUnwrap(active.draft)
    let previous = try XCTUnwrap(draft.returnProjectID)
    try FileManager.default.removeItem(at: workspace.projectLibrary.documentURL(id: previous))
    active.leave(returnToPreviousProject: true)
    XCTAssertTrue(workspace.isAffineLessonActive)
    XCTAssertEqual(active.draft?.projectID, draft.projectID)
    XCTAssertTrue(active.statusMessage.contains("return project"))
  }

  @MainActor func testCorruptLessonSidecarRetainsProtectedIdentityAndAllowsExplicitNewProject()
    throws
  {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(rootOverride: directory, detectCapabilities: false)
    var lesson: AffineLearningStore? = AffineLearningStore(workspace: workspace)
    lesson?.start()
    let id = try XCTUnwrap(workspace.activeProjectID)
    let url = workspace.workspaceDirectory.appendingPathComponent(
      "learning/projects/\(id)/draft.json")
    lesson = nil
    try Data("corrupt lesson".utf8).write(to: url)
    let reopened = AffineLearningStore(workspace: workspace)
    reopened.reopenCurrentProject()
    XCTAssertTrue(reopened.isRestoreFailure)
    XCTAssertTrue(workspace.isAffineLessonActive)
    reopened.leave(returnToPreviousProject: true)
    XCTAssertTrue(workspace.isAffineLessonActive)
    workspace.beginProject(.init(name: "Untitled flow"))
    XCTAssertFalse(workspace.isAffineLessonActive)
    XCTAssertNotEqual(workspace.activeProjectID, id)
    XCTAssertEqual(try String(contentsOf: url), "corrupt lesson")
  }
  @MainActor func testLessonIdentityRestoresSynchronouslyBeforeRunCanStart() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let workspace = WorkspaceStore(rootOverride: directory, detectCapabilities: false)
    let active = AffineLearningStore(workspace: workspace)
    active.start()
    let id = try XCTUnwrap(workspace.activeProjectID)
    let graph = workspace.graph
    workspace.beginProject(.init(name: "Untitled flow"))
    XCTAssertFalse(workspace.isAffineLessonActive)
    XCTAssertTrue(workspace.beginProject(graph, id: id))
    XCTAssertTrue(workspace.isAffineLessonActive)
    XCTAssertEqual(active.draft?.projectID, id)
    workspace.run()
    XCTAssertTrue(workspace.statusMessage.contains("lesson assessment"))
  }
  @MainActor private func finish(_ workspace: WorkspaceStore) async throws {
    let deadline = Date().addingTimeInterval(25)
    while workspace.isAssessingLearning && Date() < deadline {
      try await Task.sleep(nanoseconds: 25_000_000)
    }
    XCTAssertFalse(workspace.isAssessingLearning)
  }

}
