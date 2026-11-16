import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowWorkspaceTests: XCTestCase {
    @MainActor func testPartitionSummaryNamesEveryOutputDeterministically() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store = DataFlowWorkspaceStore(workspace: WorkspaceStore(rootOverride: root, detectCapabilities: false))
        store.loadExample(columns: ["x", "y"])
        store.previews["split"] = ["validation": ["kind": "table", "preview": ["rowCount": 16]], "training": ["kind": "table", "preview": ["rowCount": 48]], "test": ["kind":"table","preview":["rowCount":16]]]
        let node = try XCTUnwrap(store.graph.nodes.first { $0.id == "split" })
        XCTAssertEqual(store.canvasSummary(node), "test: 16 · training: 48 · validation: 16 rows")
    }
    @MainActor func testExampleUsesTrainingOnlyPreprocessAcrossThreePopulations() throws {
        let store = DataFlowWorkspaceStore(workspace: WorkspaceStore(rootOverride: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(), detectCapabilities: false))
        store.loadExample(columns: ["x", "y"])
        XCTAssertEqual(store.project.graph.nodes.count, 7)
        XCTAssertTrue(store.project.graph.edges.contains { $0.from.node == "split" && $0.from.port == "training" && $0.to.node == "fit" })
        XCTAssertEqual(store.project.graph.nodes.filter { $0.operatorID == "kwip.data.preprocess" }.count, 1)
        for role in ["training","validation","test"] {
            XCTAssertTrue(store.project.graph.edges.contains {$0.from.node == "split" && $0.from.port == role && $0.to.node == "fit" && $0.to.port == role})
        }
    }
    @MainActor func testPortableGraphExcludesBindingsAndIndependentCopyKeepsConfiguration() throws {
        let store = DataFlowWorkspaceStore(workspace: WorkspaceStore(rootOverride: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(), detectCapabilities: false))
        store.loadExample(columns: ["amount"])
        store.project.sourceBindings["source"] = "/private/example.csv"
        store.selectNode("cast", extending: false)
        store.duplicateSelection()
        let copies = store.project.graph.nodes.filter { $0.operatorID == "kwip.data.columns" }
        XCTAssertEqual(copies.count, 2)
        XCTAssertEqual(copies.first?.config, copies.last?.config)
        let portable = try DataFlowCodec.encodeGraph(store.project.graph)
        XCTAssertFalse(String(decoding: portable, as: UTF8.self).contains("/private"))
        store.undo()
        XCTAssertEqual(store.project.graph.nodes.count, 7)
    }
    @MainActor func testTypedConnectionRejectsValidationToTrainingFit() {
        let store = DataFlowWorkspaceStore(workspace: WorkspaceStore(rootOverride: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(), detectCapabilities: false))
        store.loadExample(columns: ["x"])
        let before = store.project
        guard let edge = store.graph.edges.first(where: { $0.target == "fit" }) else { return XCTFail("Missing fit edge") }
        store.reconnect(edgeID: edge.id, source: "split", sourcePort: "validation", target: "fit", targetPort: "training")
        XCTAssertEqual(store.project, before)
        XCTAssertTrue(store.statusMessage.contains("training"))
    }
    @MainActor func testRealPreparationPublishesOutputsAndReopensOnlyWithinCapturedProject() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath: repository.appendingPathComponent(".venv/bin/python").path) else { throw XCTSkip("Development Python unavailable") }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadExample(columns: [" x, coordinate ", "y"])
        let csv = directory.appendingPathComponent("measurements.csv")
        let rows = [#"" x, coordinate ",y"#] + (0..<40).map { "\($0),\($0 * 2 + 3)" }
        try rows.joined(separator: "\n").write(to: csv, atomically: true, encoding: .utf8)
        store.project.sourceBindings["source"] = csv.path
        store.runPreparation()
        XCTAssertTrue(store.isBusy, store.statusMessage)
        XCTAssertTrue(workspace.isBusy)
        let deadline = Date().addingTimeInterval(30)
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(store.isBusy)
        XCTAssertFalse(workspace.isPreparingData)
        XCTAssertNil(workspace.cancelDataPreparation)
        XCTAssertNotNil(store.manifestURL, store.statusMessage)
        XCTAssertEqual(store.previews.count, 7)
        let train = store.previews["train"]?["table"] as? [String: Any]
        let preview = train?["preview"] as? [String: Any]
        XCTAssertEqual(preview?["rowCount"] as? Int, 24)
        store.selectNode("cast"); store.rename("My conversions"); store.save()
        let reopened = DataFlowWorkspaceStore(workspace: workspace)
        XCTAssertEqual(reopened.project, store.project)
        let firstURL = reopened.saveURL
        workspace.activeProjectID = UUID().uuidString
        XCTAssertEqual(reopened.saveURL, firstURL)
        let other = DataFlowWorkspaceStore(workspace: workspace)
        XCTAssertNotEqual(other.saveURL, firstURL)
        XCTAssertTrue(other.project.sourceBindings.isEmpty)
        store.runPreparation()
        XCTAssertTrue(store.statusMessage.contains("Return to this preparation's project"))
        reopened.closeSourceInspection()
        let inspectionDeadline = Date().addingTimeInterval(10)
        while reopened.isBusy && Date() < inspectionDeadline { try await Task.sleep(nanoseconds: 25_000_000) }
        workspace.activeProjectID = store.projectID
        store.runPreparation()
        workspace.cancelRun()
        let cancelDeadline = Date().addingTimeInterval(10)
        while store.isBusy && Date() < cancelDeadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(workspace.isBusy)
        XCTAssertNotNil(store.manifestURL, "Cancellation retains the previous verified completion")
        XCTAssertTrue(store.resultsCurrent, "The retained result still matches these unchanged sources")
        XCTAssertTrue(store.statusMessage.contains("cancelled"))
    }

}
