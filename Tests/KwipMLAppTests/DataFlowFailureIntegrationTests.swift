import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowFailureIntegrationTests: XCTestCase {
    @MainActor func testActualTypedFilterFailureSelectionStalenessAndRecovery() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        workspace.loadExample()
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadBundledSensorExample()
        try await wait(store)
        store.selectNode("quality-filter")
        store.rename("Invalid quality literal")
        let original = try XCTUnwrap(store.project.graph.nodes.first { $0.id == "quality-filter" }?.config)
        var invalid = original
        guard case .array(var conditions)=invalid["conditions"],case .object(var condition)=conditions.first else {return XCTFail("Missing current Filter condition")}
        condition["value"] = .integer(3);conditions[0] = .object(condition);invalid["conditions"] = .array(conditions)
        store.configure(invalid, nodeID: "quality-filter")
        store.runPreparation(); try await wait(store)
        let failure = try XCTUnwrap(store.preparationFailure, store.statusMessage)
        XCTAssertEqual(failure.detail.code, "nodeExecution")
        XCTAssertEqual(failure.detail.nodeID,"quality-filter")
        XCTAssertEqual(failure.detail.operatorID,"kwip.data.filterRows")
        XCTAssertTrue(failure.detail.cause.lowercased().contains("literal type"))
        XCTAssertEqual(failure.detail.nodeLabel, "Invalid quality literal")
        XCTAssertEqual(failure.detail.inputRowCount, 48)
        XCTAssertTrue(store.failureIsCurrent)
        XCTAssertNil(store.manifestURL)
        store.selectNode("source"); store.selectFailureNode()
        XCTAssertEqual(store.selectedNodeID, "quality-filter")
        store.configure(original, nodeID: "quality-filter")
        XCTAssertFalse(store.failureIsCurrent)
        store.runPreparation(); try await wait(store)
        XCTAssertTrue(store.resultsCurrent, store.statusMessage)
        XCTAssertNil(store.preparationFailure)
        let artifacts = try XCTUnwrap(store.manifestURL?.deletingLastPathComponent())
        XCTAssertFalse(FileManager.default.fileExists(atPath: artifacts.appendingPathComponent("node-labels.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: artifacts.deletingLastPathComponent().appendingPathComponent("input").path),
                       "Private source and diagnostic-label inputs are removed after the worker exits")
        XCTAssertEqual(store.project.editor["quality-filter"]?.name, "Invalid quality literal")
    }
    @MainActor func testEmptyMatchingBranchAttributesFailureToDownstreamPartition() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        workspace.loadExample();let store=DataFlowWorkspaceStore(workspace:workspace)
        store.loadBundledSensorExample();try await wait(store)
        store.selectNode("quality-filter")
        var config=try XCTUnwrap(store.project.graph.nodes.first {$0.id=="quality-filter"}?.config)
        guard case .array(var conditions)=config["conditions"],case .object(var condition)=conditions.first else {return XCTFail("Missing current Filter condition")}
        condition["value"] = .string("not-in-source");conditions[0] = .object(condition);config["conditions"] = .array(conditions)
        store.configure(config,nodeID:"quality-filter");store.runPreparation();try await wait(store)
        // Empty Filter matching output is valid. The later required Partition
        // population refuses it, and attribution must identify that actual block.
        let failure=try XCTUnwrap(store.preparationFailure,store.statusMessage)
        XCTAssertEqual(failure.detail.nodeID,"partition");XCTAssertEqual(failure.detail.operatorID,"kwip.data.partition")
        XCTAssertEqual(failure.detail.code,"nodeExecution");XCTAssertEqual(failure.detail.inputRowCount,0)
        XCTAssertTrue(store.failureIsCurrent);XCTAssertNil(store.manifestURL)
        store.selectFailureNode();XCTAssertEqual(store.selectedNodeID,"partition")
    }
    @MainActor private func wait(_ store: DataFlowWorkspaceStore) async throws {
        let deadline = Date().addingTimeInterval(20)
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(store.isBusy, "Preparation worker did not stop")
    }
}
