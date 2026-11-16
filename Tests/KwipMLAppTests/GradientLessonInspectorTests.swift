import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientLessonInspectorTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func testRequestRawDraftColdRestorationAndDiscardPreservesUnrelatedPending() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        XCTAssertTrue(lesson.start(operation: GradientLearningTask.selectedInfluence.rawValue))
        let canvas = try XCTUnwrap(lesson.canvas), node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "gradient_policy" })
        canvas.selectNode(node.id, extending: false)
        let context = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
        XCTAssertEqual(Set(context.owners.map { $0.node + "." + $0.parameter }), ["first.weight", "first.bias", "second.weight", "second.bias"])
        XCTAssertTrue(context.setRawNumericDraft("accumulation_steps", text: "-", origin: context.numericOrigin("accumulation_steps")))
        XCTAssertFalse(context.canApply); XCTAssertFalse(try XCTUnwrap(lesson.draft).pendingFields.isEmpty)
        XCTAssertTrue(lesson.drafts.update { $0.pendingFields["unrelated"] = "-" })
        XCTAssertTrue(lesson.drafts.save())
        let id = try XCTUnwrap(workspace.activeProjectID), document = workspace.graph
        let cold = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let reopened = TensorLearningStore(workspace: cold)
        XCTAssertTrue(cold.beginProject(document, id: id))
        let restoredCanvas = try XCTUnwrap(reopened.canvas); restoredCanvas.selectNode(node.id, extending: false)
        let restored = try XCTUnwrap(restoredCanvas.gradientPolicyInspector(nodeID: node.id))
        XCTAssertEqual(restored.rawNumericDrafts["accumulation_steps"], "-")
        XCTAssertTrue(restored.discard()); XCTAssertEqual(try XCTUnwrap(reopened.draft).pendingFields, ["unrelated": "-"])
        XCTAssertTrue(context.discard())
        canvas.selectNode(try XCTUnwrap(canvas.graph.nodes.first { $0.type == "train" }).id, extending: false)
        XCTAssertFalse(context.setAccumulationSteps(2))
    }
    func testTrainabilityApplyUndoAndFixedContext() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        XCTAssertTrue(lesson.start(operation: GradientLearningTask.keepSettings.rawValue))
        let canvas = try XCTUnwrap(lesson.canvas), node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "trainability" })
        canvas.selectNode(node.id, extending: false)
        let context = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
        let owner = try XCTUnwrap(context.owners.first { $0.node == "first" && $0.parameter == "weight" })
        if context.rows.isEmpty { XCTAssertTrue(context.addOwner(owner.id)) }
        XCTAssertTrue(context.setMode("frozen", at: 0)); XCTAssertTrue(context.apply())
        let applied = try XCTUnwrap(canvas.graph.nodes.first { $0.id == node.id }).config
        canvas.undo(); XCTAssertNotEqual(canvas.graph.nodes.first { $0.id == node.id }?.config, applied)
        XCTAssertFalse(context.setMode("trainable", at: 0), "Undo retires old editor callbacks")
        let train = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "train" })
        XCTAssertNil(canvas.trainabilityInspector(nodeID: train.id))
        XCTAssertFalse(canvas.configure(nodeID: train.id, values: [:]))
    }
}
