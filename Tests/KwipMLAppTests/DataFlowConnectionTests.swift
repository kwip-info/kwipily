import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowConnectionTests: XCTestCase {
    @MainActor private func store() -> DataFlowWorkspaceStore {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        return DataFlowWorkspaceStore(workspace: workspace)
    }
    @MainActor func testCurrentPartitionAndPreprocessOutputsRetainKnownRoles() {
        let value = store(); value.loadExample(columns: ["a", "b"])
        XCTAssertNil(value.connectionError(source: "split", sourcePort: "training", target: "train", targetPort: "table"))
        // View is a canvas sink; its retained execution output is not authorable.
        XCTAssertNotNil(value.connectionError(source: "train", sourcePort: "table", target: "fit", targetPort: "training"))
        value.project.graph.nodes.append(.init(id:"kept-training",operatorID:"kwip.data.columns",config:ColumnsConfigurationDraft.defaults))
        value.project.graph.edges.append(.init(from:.init(node:"fit",port:"training"),to:.init(node:"kept-training",port:"table")))
        XCTAssertNil(value.connectionError(source:"kept-training",sourcePort:"table",target:"fit",targetPort:"training"))
        XCTAssertEqual(value.knownPartition(nodeID:"kept-training",port:"table"),"training")
        XCTAssertNotNil(value.connectionError(source: "split", sourcePort: "validation", target: "fit", targetPort: "training"))
        XCTAssertNotNil(value.connectionError(source: "validation", sourcePort: "table", target: "fit", targetPort: "training"))
        XCTAssertNotNil(value.connectionError(source: "source", sourcePort: "table", target: "fit", targetPort: "training"))
        XCTAssertNotNil(value.connectionError(source: "fit", sourcePort: "state", target: "train", targetPort: "table"))
        XCTAssertNotNil(value.connectionError(source:"split",sourcePort:"test",target:"fit",targetPort:"training"))
        XCTAssertEqual(value.knownPartition(nodeID:"fit",port:"test"),"test")
        XCTAssertEqual(value.knownPartition(nodeID:"test",port:"table"),"test")
    }
    @MainActor func testExistingSplitWireCanBeReconnectedThroughSharedCommands() {
        let value = store(); value.loadExample(columns: ["a", "b"])
        let edge = value.project.graph.edges.first { $0.to.node == "train" && $0.to.port == "table" }!
        value.selectEdge(edge.id); value.deleteSelection()
        value.connect(source: "split", sourcePort: "validation", target: "fit", targetPort: "training")
        XCTAssertTrue(value.statusMessage.contains("requires"))
        value.connect(source: "split", sourcePort: "training", target: "train", targetPort: "table")
        XCTAssertFalse(value.statusMessage.contains("requires"))
        XCTAssertTrue(value.project.graph.edges.contains { $0.from.node == "split" && $0.to.node == "train" && $0.to.port == "table" })
    }
    @MainActor func testUnknownPartitionDefersAndCyclicDraftCannotRecurseIndefinitely() {
        let value = store(); value.loadExample(columns: ["a", "b"])
        value.project.graph.nodes.append(.init(id:"unresolved",operatorID:"kwip.data.columns",config:ColumnsConfigurationDraft.defaults))
        XCTAssertNil(value.knownPartition(nodeID:"unresolved",port:"table"))
        XCTAssertNil(value.connectionError(source:"unresolved",sourcePort:"table",target:"fit",targetPort:"training"))
        value.project.graph.edges.append(.init(from:.init(node:"unresolved",port:"table"),to:.init(node:"unresolved",port:"table")))
        XCTAssertNil(value.knownPartition(nodeID:"unresolved",port:"table"))
    }

}
