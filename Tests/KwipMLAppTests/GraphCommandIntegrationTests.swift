import XCTest
import KwipMLCore
@testable import KwipMLApp
final class GraphCommandIntegrationTests: XCTestCase {
    @MainActor func testDuplicateGroupAndUndoRedoRestoreSelectionAndTopology() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        store.loadExample()
        store.selectNode("branch-a"); store.selectNode("activation-a", extending: true)
        XCTAssertEqual(store.selectedNodeIDs, ["branch-a","activation-a"])
        let before = store.graph
        store.duplicateSelection()
        XCTAssertEqual(store.graph.nodes.count, before.nodes.count + 2)
        XCTAssertEqual(store.graph.edges.count, before.edges.count + 1)
        let copied = store.selectedNodeIDs
        XCTAssertTrue(copied.isDisjoint(with: ["branch-a","activation-a"]))
        store.undo()
        XCTAssertEqual(store.graph, before)
        XCTAssertEqual(store.selectedNodeIDs, ["branch-a","activation-a"])
        store.redo()
        XCTAssertEqual(store.selectedNodeIDs, copied)
        store.deleteSelection()
        XCTAssertEqual(store.graph, before)
    }
    @MainActor func testInvalidReconnectLeavesGraphAndHistoryUntouched() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        store.loadExample()
        let before = store.graph, edge = try XCTUnwrap(store.graph.edges.first)
        store.reconnect(edgeID: edge.id, source: "output", sourcePort: "out", target: edge.target, targetPort: edge.targetPort)
        XCTAssertEqual(store.graph, before)
        XCTAssertFalse(store.canUndo)
        XCTAssertFalse(store.statusMessage.isEmpty)
    }
}
