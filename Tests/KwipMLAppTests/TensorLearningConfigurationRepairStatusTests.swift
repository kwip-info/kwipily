import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorLearningConfigurationRepairStatusTests: XCTestCase {
    @MainActor func testRejectedSoftmaxBatchAxisThenValidSampleAxisReplacesErrorAndRetainsDraftText() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let session = TensorLearningDraftSession(repository: .init(directory: root))
        XCTAssertTrue(session.create(projectID: UUID().uuidString, operation: "softmax-first-sample-axis", mode: .practice))
        let canvas = TensorLearningCanvasAdapter(session: session)
        for type in ["input", "softmax", "output"] { canvas.addNode(type: type, x: 0, y: 0) }
        let input = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "input" })
        let softmax = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "softmax" })
        let output = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "output" })
        canvas.connect(source: input.id, sourcePort: "out", target: softmax.id, targetPort: "in")
        canvas.connect(source: softmax.id, sourcePort: "out", target: output.id, targetPort: "in")
        let before = canvas.graph
        XCTAssertTrue(canvas.setPending(nodeID: softmax.id, field: "dim", text: "unfinished"))
        XCTAssertFalse(canvas.configure(nodeID: softmax.id, values: ["dim": 0]))
        XCTAssertEqual(canvas.graph, before)
        XCTAssertEqual(canvas.pendingText(nodeID: softmax.id, field: "dim"), "unfinished")
        let rejection = canvas.statusMessage
        XCTAssertFalse(rejection.isEmpty)
        XCTAssertTrue(canvas.configure(nodeID: softmax.id, values: ["dim": 1]))
        XCTAssertEqual(canvas.graph.nodes.first { $0.id == softmax.id }?.config["dim"], 1)
        XCTAssertEqual(canvas.pendingText(nodeID: softmax.id, field: "dim"), "unfinished", "A direct valid edit does not silently erase a separate pending draft.")
        XCTAssertNotEqual(canvas.statusMessage, rejection)
        XCTAssertEqual(canvas.statusMessage, "Configuration applied.")
        XCTAssertFalse(canvas.configure(nodeID: softmax.id, values: ["dim": 0]))
        XCTAssertEqual(canvas.graph.nodes.first { $0.id == softmax.id }?.config["dim"], 1)
        XCTAssertNotEqual(canvas.statusMessage, "Configuration applied.")
    }
    @MainActor func testNativePendingRepairBackToAlreadyCommittedAxisAcknowledgesValidValue() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let session = TensorLearningDraftSession(repository: .init(directory: root))
        XCTAssertTrue(session.create(projectID: UUID().uuidString, operation: "softmax-first-sample-axis", mode: .practice))
        let canvas = TensorLearningCanvasAdapter(session: session)
        canvas.addNode(type: "softmax", x: 0, y: 0)
        let node = try XCTUnwrap(canvas.graph.nodes.first { $0.type == "softmax" })
        XCTAssertTrue(canvas.configure(nodeID: node.id, values: ["dim": 1]))
        let original = canvas.graph
        XCTAssertTrue(canvas.setPending(nodeID: node.id, field: "dim", text: "0"))
        XCTAssertFalse(canvas.applyPending(nodeID: node.id, field: "dim"))
        XCTAssertEqual(canvas.graph, original)
        XCTAssertEqual(canvas.pendingText(nodeID: node.id, field: "dim"), "0")
        let rejection = canvas.statusMessage
        XCTAssertTrue(canvas.setPending(nodeID: node.id, field: "dim", text: "1"))
        XCTAssertNil(canvas.pendingText(nodeID: node.id, field: "dim"), "Restoring the committed text removes the draft difference.")
        XCTAssertTrue(canvas.applyPending(nodeID: node.id, field: "dim"), "Return on the restored valid value must acknowledge the successful repair.")
        XCTAssertEqual(canvas.graph, original)
        XCTAssertNil(canvas.pendingText(nodeID: node.id, field: "dim"))
        XCTAssertNotEqual(canvas.statusMessage, rejection)
        XCTAssertEqual(canvas.statusMessage, "Configuration applied.")
        canvas.statusMessage = "Unrelated saved result is available."
        XCTAssertTrue(canvas.setPending(nodeID: node.id, field: "dim", text: "1"))
        XCTAssertEqual(canvas.statusMessage, "Unrelated saved result is available.", "Untouched no-draft text must not overwrite unrelated status.")
        XCTAssertFalse(canvas.applyPending(nodeID: "missing", field: "dim"))
        XCTAssertEqual(canvas.statusMessage, "Unrelated saved result is available.")
    }
    @MainActor func testInactiveOrMissingNodeConfigurationPreservesUnrelatedStatus() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let session = TensorLearningDraftSession(repository: .init(directory: root))
        XCTAssertTrue(session.create(projectID: UUID().uuidString, operation: "softmax-first-sample-axis", mode: .practice))
        let canvas = TensorLearningCanvasAdapter(session: session)
        canvas.statusMessage = "Unrelated saved result is available."
        XCTAssertFalse(canvas.configure(nodeID: "missing", values: ["dim": 1]))
        XCTAssertEqual(canvas.statusMessage, "Unrelated saved result is available.")
        XCTAssertTrue(session.create(projectID: UUID().uuidString, operation: "transpose", mode: .practice))
        XCTAssertFalse(canvas.configure(nodeID: "missing", values: ["dim": 1]))
        XCTAssertEqual(canvas.statusMessage, "Unrelated saved result is available.")
    }
}
