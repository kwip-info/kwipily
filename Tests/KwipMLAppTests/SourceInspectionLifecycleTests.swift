import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SourceInspectionLifecycleTests: XCTestCase {
    @MainActor private func waitFor(_ predicate: () -> Bool) async throws {
        for _ in 0..<1000 {
            if predicate() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Inspection did not reach the expected state")
    }
    @MainActor private func fixture() throws -> (URL, URL, WorkspaceStore, DataFlowWorkspaceStore) {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("source.csv")
        try "a,b\n1,2\n".write(to: file, atomically: true, encoding: .utf8)
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let value = DataFlowWorkspaceStore(workspace: workspace)
        value.project.sourceBindings["source"] = file.path
        return (root, file, workspace, value)
    }
    @MainActor func testRefreshSerializesWorkersAndRejectsObsoleteEvents() async throws {
        let (root, _, workspace, value) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var exits: [(Int32, String, SourceInspectionResult?) -> Void] = [], hashes: [String] = []
        var cancelled = 0
        value.sourceInspectionExecute = { _, _, _, _, _, hash, _, exit in
            hashes.append(hash); exits.append(exit); return .init(cancel: { cancelled += 1 })
        }
        value.inspectSource(id: "source")
        XCTAssertTrue(workspace.isBusy)
        try await waitFor { exits.count == 1 }
        value.inspectSource(id: "source")
        XCTAssertEqual(exits.count, 1); XCTAssertEqual(cancelled, 1)
        exits[0](1, "obsolete error", nil)
        try await waitFor { exits.count == 2 }
        exits[0](1, "late obsolete error", nil)
        XCTAssertTrue(value.isInspectingSource); XCTAssertTrue(workspace.isBusy)
        XCTAssertFalse(value.statusMessage.contains("obsolete"))
        exits[1](0, "", .init(columns: ["a", "b"], rows: [["1", "2"]], rowCount: 1, sourceSHA256: hashes[1]))
        try await waitFor { !value.isInspectingSource }
        value.selectNode("cast")
        XCTAssertEqual(value.sourceColumns, ["a", "b"]); XCTAssertFalse(workspace.isBusy)
        value.inspectSource(id: "source")
        try await waitFor { exits.count == 3 }
        value.closeSourceInspection()
        XCTAssertTrue(workspace.isBusy)
        exits[2](0, "", .init(columns: ["late"], rows: [["1"]], rowCount: 1, sourceSHA256: hashes[2]))
        try await waitFor { !value.isInspectingSource }
        XCTAssertFalse(value.sourceColumns.contains("late")); XCTAssertFalse(workspace.isBusy)
    }
    @MainActor func testSourceMutationAndContextReplacementRejectHeadersAndPreserveNewReservation() async throws {
        let (root, file, workspace, value) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        var exit: ((Int32, String, SourceInspectionResult?) -> Void)?, hash = ""
        value.sourceInspectionExecute = { _, _, _, _, _, captured, _, finish in
            hash = captured; exit = finish; return .init(cancel: {})
        }
        value.inspectSource(id: "source")
        try await waitFor { exit != nil }
        try "x,y\n3,4\n".write(to: file, atomically: true, encoding: .utf8)
        exit?(0, "", .init(columns: ["a", "b"], rows: [["1", "2"]], rowCount: 1, sourceSHA256: hash))
        try await waitFor { !value.isInspectingSource }
        XCTAssertTrue(value.sourceHeaders.isEmpty)
        exit = nil; value.inspectSource(id: "source")
        try await waitFor { exit != nil }
        workspace.cancelDataPreparation = {}
        let reservation = workspace.dataPreparationReservationID
        workspace.activeProjectID = UUID().uuidString
        exit?(0, "", .init(columns: ["x", "y"], rows: [["3", "4"]], rowCount: 1, sourceSHA256: hash))
        try await waitFor { !value.isInspectingSource }
        XCTAssertTrue(value.sourceHeaders.isEmpty); XCTAssertTrue(workspace.isPreparingData)
        XCTAssertEqual(workspace.dataPreparationReservationID, reservation)
        workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
    }
    @MainActor func testImmediateCloseRetainsStoreUntilCaptureCancellationJoins() async throws {
        var context: (URL, URL, WorkspaceStore, DataFlowWorkspaceStore)? = try fixture()
        let root = try XCTUnwrap(context?.0), workspace = try XCTUnwrap(context?.2)
        defer { try? FileManager.default.removeItem(at: root) }
        var store: DataFlowWorkspaceStore? = context?.3
        weak var retained = store
        context = nil
        var launches = 0
        store?.sourceInspectionExecute = { _, _, _, _, _, _, _, _ in launches += 1; return .init(cancel: {}) }
        store?.inspectSource(id: "source")
        store?.closeSourceInspection(); store = nil
        XCTAssertTrue(workspace.isPreparingData)
        try await waitFor { !workspace.isPreparingData && retained == nil }
        XCTAssertEqual(launches, 0); XCTAssertNil(retained)
    }
    @MainActor func testActualSupervisedPreparationSourceInspectionPublishesVerifiedHeaders() async throws {
        let (root, _, workspace, value) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try XCTUnwrap(workspace.runtime)
        value.inspectSource(id: "source")
        try await waitFor { !value.isInspectingSource }
        value.selectNode("cast")
        XCTAssertEqual(value.sourceColumns, ["a", "b"], value.statusMessage)
        XCTAssertFalse(workspace.isPreparingData)
    }
    @MainActor func testIdentityAddressedConfigurationNeverEditsNewSelection() {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let value = DataFlowWorkspaceStore(workspace: WorkspaceStore(rootOverride: root, detectCapabilities: false))
        value.selectNode("cast"); let old = value.project.graph.nodes.first { $0.id == "cast" }!
        value.selectNode("split"); let split = value.project.graph.nodes.first { $0.id == "split" }!
        value.configure(old.config, nodeID: old.id)
        XCTAssertEqual(value.project.graph.nodes.first { $0.id == "split" }, split)
    }
    @MainActor func testHeadersFollowTheSelectedUpstreamSourceAndNeverUnion() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let a = root.appendingPathComponent("a.csv"), b = root.appendingPathComponent("b.csv")
        try "a\n1\n".write(to: a, atomically: true, encoding: .utf8); try "b\n2\n".write(to: b, atomically: true, encoding: .utf8)
        let value = DataFlowWorkspaceStore(workspace: WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false))
        let source = value.project.graph.nodes.first { $0.id == "source" }!
        value.project.graph.nodes.append(DataFlowNode(id: "other", operatorID: source.operatorID, config: source.config))
        value.project.sourceBindings = ["source": a.path, "other": b.path]
        value.sourceHeaders["source"] = SourceHeaderEvidence(path: a.path, config: source.config, fingerprint: try XCTUnwrap(value.sourceFingerprint(a.path)), columns: ["a"])
        value.sourceHeaders["other"] = SourceHeaderEvidence(path: b.path, config: source.config, fingerprint: try XCTUnwrap(value.sourceFingerprint(b.path)), columns: ["b"])
        value.selectNode("source"); XCTAssertEqual(value.sourceColumns, ["a"])
        value.selectNode("other"); XCTAssertEqual(value.sourceColumns, ["b"])
        value.selectNode("cast"); XCTAssertEqual(value.sourceColumns, ["a"])
        value.project.graph.edges.removeAll { $0.to.node == "cast" }
        value.project.graph.edges.append(DataFlowEdge(from: DataFlowEndpoint(node: "other", port: "table"), to: DataFlowEndpoint(node: "cast", port: "table")))
        XCTAssertEqual(value.sourceColumns, ["b"])
        value.clearSelection(); XCTAssertTrue(value.sourceColumns.isEmpty)
        value.selectNode("other"); value.bindSource(a.path, nodeID: "other")
        XCTAssertTrue(value.sourceColumns.isEmpty)
        value.undo(); XCTAssertEqual(value.project.sourceBindings["other"], b.path)
        value.loadExample(columns: ["new"]); XCTAssertTrue(value.sourceColumns.isEmpty)
    }

}
