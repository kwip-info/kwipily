import KwipMLCore
import XCTest

@testable import KwipMLApp

final class LegacyAffineMigrationTests: XCTestCase {
  @MainActor func testActualSchemaOneReceiptSurvivesEditableGraphMigration() throws {
    let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().appendingPathComponent("Fixtures/legacy-affine-receipt-v1.json")
    let bytes = try Data(contentsOf: fixture)
    let receipt = try JSONDecoder().decode(AffineLearningReceipt.self, from: bytes)
    XCTAssertEqual(receipt.submission.graph.schemaVersion, 1)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let workspace = WorkspaceStore(
      rootOverride: root, storageOverride: root, detectCapabilities: false)
    let learning = root.appendingPathComponent("learning")
    let project = learning.appendingPathComponent("projects/\(receipt.projectID)")
    let attempts = learning.appendingPathComponent("attempts")
    try FileManager.default.createDirectory(at: project, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: attempts, withIntermediateDirectories: true)
    var draft = AffineLearningDraft(
      projectID: receipt.projectID, graph: receipt.submission.graph, mode: .assessment)
    draft.parameters = try receipt.submission.parameters.mapValues { value in
      LinearParameterValues(weight: value.weight, bias: try XCTUnwrap(value.bias))
    }
    try AffineLearningCodec.encodeDraft(draft).write(
      to: project.appendingPathComponent("draft.json"))
    try bytes.write(to: attempts.appendingPathComponent(receipt.attempt.id + ".json"))
    try LearningProgressStore(directory: learning).append(receipt.attempt)
    let record = try workspace.projectLibrary.save(receipt.submission.graph, id: receipt.projectID)
    let lesson = AffineLearningStore(workspace: workspace)
    workspace.openManagedProject(record)
    XCTAssertEqual(workspace.graph.schemaVersion, 2)
    XCTAssertTrue(lesson.resultSaved, lesson.statusMessage)
    XCTAssertEqual(lesson.latestGrade?.status, .passed)
    XCTAssertTrue(lesson.latestResultIsCurrent)
    XCTAssertFalse(lesson.latestAssessmentIsCurrent)
    XCTAssertFalse(lesson.hasCurrentCompletion)
    XCTAssertEqual(
      try Data(contentsOf: attempts.appendingPathComponent(receipt.attempt.id + ".json")), bytes)
    let linear = try XCTUnwrap(workspace.graph.nodes.first { $0.type == "linear" })
    workspace.selectNode(linear.id)
    XCTAssertTrue(workspace.setConfigValue("bias", value: .bool(false), nodeID: linear.id))
    XCTAssertFalse(lesson.latestResultIsCurrent)
    XCTAssertTrue(lesson.parameterIssues.isEmpty, lesson.parameterIssues.joined())
    workspace.undo()
    XCTAssertTrue(lesson.latestResultIsCurrent)
    XCTAssertEqual(
      lesson.parameters(for: linear.id)?.bias.values,
      receipt.submission.parameters[linear.id]?.bias?.values)
  }
}
