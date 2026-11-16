import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientLessonInspectorPolicyValidationTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func withLesson(_ task: GradientLearningTask, _ body: (TensorLearningCanvasAdapter) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        XCTAssertTrue(lesson.start(operation: task.rawValue))
        try body(XCTUnwrap(lesson.canvas))
        withExtendedLifetime((workspace, lesson)) {}
    }
    func testFixedGradientPoliciesExplainRestrictionBeforeApply() throws {
        for task in [GradientLearningTask.keepSettings, .nextRun] {
            try withLesson(task) { canvas in
                let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "gradient_policy" })
                canvas.selectNode(node.id, extending: false)
                let context = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
                XCTAssertTrue(context.setAccumulationSteps(2))
                XCTAssertTrue(context.validationMessage?.contains("This lesson keeps Gradient policy fixed. Restore it and configure Trainability instead.") == true)
                XCTAssertFalse(context.canApply); XCTAssertFalse(context.apply())
                XCTAssertFalse(context.rejection?.contains("source changed") == true)
                XCTAssertEqual(canvas.graph.nodes.first { $0.id == node.id }?.config, node.config)
                XCTAssertFalse(canvas.canUndo)
                XCTAssertTrue(context.hasChanges, "Rejected draft remains available for correction or discard")
                XCTAssertTrue(context.discard()); XCTAssertNil(context.validationMessage)
            }
        }
    }
    func testFixedTrainabilityPoliciesExplainRestrictionBeforeApply() throws {
        for task in [GradientLearningTask.selectedInfluence, .combineSmallGroups] {
            try withLesson(task) { canvas in
                let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "trainability" })
                canvas.selectNode(node.id, extending: false)
                let context = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
                let owner = try XCTUnwrap(context.owners.first { $0.node == "first" && $0.parameter == "weight" })
                XCTAssertTrue(context.addOwner(owner.id)); XCTAssertTrue(context.setMode("frozen", at: 0))
                XCTAssertTrue(context.validationMessage?.contains("This lesson keeps Trainability fixed. Restore it and configure Gradient policy instead.") == true)
                XCTAssertFalse(context.canApply); XCTAssertFalse(context.apply())
                XCTAssertFalse(context.rejection?.contains("source changed") == true)
                XCTAssertEqual(canvas.graph.nodes.first { $0.id == node.id }?.config, node.config)
                XCTAssertFalse(canvas.canUndo); XCTAssertTrue(context.hasChanges)
                XCTAssertTrue(context.discard()); XCTAssertNil(context.validationMessage)
            }
        }
    }
    func testEditableGradientPoliciesStillApplyWithoutCompleteStarterConnections() throws {
        for task in [GradientLearningTask.selectedInfluence, .combineSmallGroups] {
            try withLesson(task) { canvas in
                let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "gradient_policy" })
                canvas.selectNode(node.id, extending: false)
                let context = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
                XCTAssertTrue(context.setAccumulationSteps(3)); XCTAssertNil(context.validationMessage)
                XCTAssertTrue(context.canApply); XCTAssertTrue(context.apply()); XCTAssertTrue(canvas.canUndo)
                XCTAssertEqual(canvas.graph.nodes.first { $0.id == node.id }?.config["accumulation_steps"], .integer(3))
            }
        }
    }
    func testEditableTrainabilityPoliciesStillApplyWithoutCompleteStarterConnections() throws {
        for task in [GradientLearningTask.keepSettings, .nextRun] {
            try withLesson(task) { canvas in
                let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "trainability" })
                canvas.selectNode(node.id, extending: false)
                let context = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
                let owner = try XCTUnwrap(context.owners.first { $0.node == "first" && $0.parameter == "weight" })
                if context.rows.isEmpty { XCTAssertTrue(context.addOwner(owner.id)) }
                guard case .object(let row) = context.rows[0] else { return XCTFail("Expected typed owner") }
                let current = row["mode"]?.stringValue
                XCTAssertTrue(context.setMode(current == "frozen" ? "trainable" : "frozen", at: 0))
                XCTAssertNil(context.validationMessage); XCTAssertTrue(context.canApply)
                XCTAssertTrue(context.apply()); XCTAssertTrue(canvas.canUndo)
            }
        }
    }
}
