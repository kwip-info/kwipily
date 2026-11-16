import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientLessonInspectorAvailabilityTests: XCTestCase {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func lesson(_ task: GradientLearningTask, root: URL) throws -> (WorkspaceStore, TensorLearningStore, TensorLearningCanvasAdapter) {
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        XCTAssertTrue(lesson.start(operation: task.rawValue))
        return (workspace, lesson, try XCTUnwrap(lesson.canvas))
    }

    func testGradientEditorSurvivesBusyRefreshAndUnlocksAfterCheck() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let (workspace, lesson, canvas) = try lesson(.selectedInfluence, root: root)
        let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "gradient_policy" })
        canvas.selectNode(node.id, extending: false)
        var busy = false
        canvas.assessmentIsRunning = { busy }
        let original = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
        XCTAssertTrue(original.setAccumulationSteps(2))
        let draft = original.draftConfiguration
        busy = true
        XCTAssertTrue(original.isBusy)
        XCTAssertFalse(original.setAccumulationSteps(3))
        XCTAssertFalse(original.apply()); XCTAssertFalse(original.discard())
        XCTAssertTrue(try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id)) === original,
                      "A transient check must retain the visible editor and its pending draft")
        XCTAssertEqual(original.draftConfiguration, draft)
        busy = false
        XCTAssertTrue(try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id)) === original)
        XCTAssertFalse(original.isBusy)
        XCTAssertTrue(original.setAccumulationSteps(3)); XCTAssertTrue(original.apply())
        XCTAssertEqual(canvas.graph.nodes.first { $0.id == node.id }?.config["accumulation_steps"], .integer(3))
        withExtendedLifetime((workspace, lesson)) {}
    }

    func testTrainabilityEditorSurvivesBusyRefreshAndUnlocksAfterCheck() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let (workspace, lesson, canvas) = try lesson(.keepSettings, root: root)
        let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "trainability" })
        canvas.selectNode(node.id, extending: false)
        var busy = false
        canvas.assessmentIsRunning = { busy }
        let original = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
        if original.rows.isEmpty { XCTAssertTrue(original.addOwner(try XCTUnwrap(original.owners.first).id)) }
        XCTAssertTrue(original.setMode("trainable", at: 0))
        let draft = original.draftConfiguration
        busy = true
        XCTAssertTrue(original.isBusy)
        XCTAssertFalse(original.setMode("frozen", at: 0))
        XCTAssertFalse(original.apply()); XCTAssertFalse(original.discard())
        XCTAssertTrue(try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id)) === original)
        XCTAssertEqual(original.draftConfiguration, draft)
        busy = false
        XCTAssertTrue(try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id)) === original)
        XCTAssertFalse(original.isBusy)
        XCTAssertTrue(original.setMode("frozen", at: 0)); XCTAssertTrue(original.apply())
        withExtendedLifetime((workspace, lesson)) {}
    }

    func testGradientStaleConfigurationSelectionAndRetiredSessionRemainBlocked() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let (workspace, lesson, canvas) = try lesson(.selectedInfluence, root: root)
        let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "gradient_policy" })
        canvas.selectNode(node.id, extending: false)
        let old = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
        var changed = node.config; changed["accumulation_steps"] = .integer(2)
        XCTAssertTrue(canvas.configure(nodeID: node.id, values: changed))
        XCTAssertFalse(old.setAccumulationSteps(3)); XCTAssertFalse(old.apply())
        let current = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
        canvas.selectNode(try XCTUnwrap(canvas.graph.nodes.first { $0.type == "train" }).id, extending: false)
        XCTAssertFalse(current.setAccumulationSteps(3))
        canvas.selectNode(node.id, extending: false)
        let final = try XCTUnwrap(canvas.gradientPolicyInspector(nodeID: node.id))
        lesson.drafts.discardLocalChanges()
        XCTAssertFalse(final.setAccumulationSteps(3)); XCTAssertFalse(final.apply())
        withExtendedLifetime(workspace) {}
    }

    func testTrainabilityStaleConfigurationSelectionAndRetiredSessionRemainBlocked() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let (workspace, lesson, canvas) = try lesson(.keepSettings, root: root)
        let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "trainability" })
        canvas.selectNode(node.id, extending: false)
        let old = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
        let owner = try XCTUnwrap(old.owners.first)
        let changed = try TrainabilityConfiguration(values: ["owners": .array([.object([
            "node": .string(owner.node), "parameter": .string(owner.parameter), "mode": .string("trainable")
        ])])]).values
        XCTAssertTrue(canvas.configure(nodeID: node.id, values: changed))
        XCTAssertFalse(old.setMode("frozen", at: 0)); XCTAssertFalse(old.apply())
        let current = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
        canvas.selectNode(try XCTUnwrap(canvas.graph.nodes.first { $0.type == "train" }).id, extending: false)
        XCTAssertFalse(current.setMode("frozen", at: 0))
        canvas.selectNode(node.id, extending: false)
        let final = try XCTUnwrap(canvas.trainabilityInspector(nodeID: node.id))
        lesson.drafts.discardLocalChanges()
        XCTAssertFalse(final.setMode("frozen", at: 0)); XCTAssertFalse(final.apply())
        withExtendedLifetime(workspace) {}
    }
}
