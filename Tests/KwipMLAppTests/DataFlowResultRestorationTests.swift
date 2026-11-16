import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowResultRestorationTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @MainActor private func finish(_ store: DataFlowWorkspaceStore) async throws {
        let deadline = Date().addingTimeInterval(30)
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
        XCTAssertFalse(store.isBusy, store.statusMessage)
    }
    @MainActor func testReopensCompletedOutputsWithoutOriginalSourceAndPreservesThemAfterCancel() async throws {
        guard FileManager.default.isExecutableFile(atPath: repository.appendingPathComponent(".venv/bin/python").path) else { throw XCTSkip("Development Python unavailable") }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadExample(columns: ["x", "y"])
        let source = directory.appendingPathComponent("data.csv")
        try (["x,y"] + (0..<40).map { "\($0),\($0 * 2)" }).joined(separator: "\n").write(to: source, atomically: true, encoding: .utf8)
        store.bindSource(source.path, nodeID: "source")
        store.runPreparation(); try await finish(store)
        let manifest = try XCTUnwrap(store.manifestURL, store.statusMessage)
        let originalManifest = try Data(contentsOf: manifest)
        store.save()
        store.runPreparation(); store.cancelPreparation(); try await finish(store)
        XCTAssertEqual(store.manifestURL, manifest, "Cancellation must preserve the last verified result")
        XCTAssertEqual(store.previews.count, 7)
        XCTAssertEqual(Set(store.previews.keys),["source","cast","split","fit","train","validation","test"])
        try FileManager.default.removeItem(at: source)
        let reopened = DataFlowWorkspaceStore(workspace: workspace)
        try await finish(reopened)
        XCTAssertEqual(reopened.manifestURL, manifest, reopened.statusMessage)
        XCTAssertEqual(reopened.previews.count, 7, reopened.statusMessage)
        XCTAssertEqual(Set(reopened.previews.keys),Set(store.previews.keys))
        XCTAssertTrue(reopened.previewStatus(for:"test").contains("Historical"))
        XCTAssertFalse(reopened.isInspectingSource)
        XCTAssertFalse(reopened.resultsCurrent, "Restored historical artifacts do not certify current source files")
        XCTAssertEqual(try Data(contentsOf: manifest), originalManifest)
        XCTAssertTrue(reopened.restoredResult)
        XCTAssertTrue(reopened.previewStatus(for: "train").contains("Historical"))
        let remembered = try Data(contentsOf: store.completedResultURL)
        reopened.runPreparation(); try await finish(reopened)
        XCTAssertEqual(reopened.manifestURL, manifest, "Missing source failure retains historical output")
        XCTAssertEqual(try Data(contentsOf: store.completedResultURL), remembered)
        reopened.selectNode("split"); var changed = try XCTUnwrap(reopened.project.graph.nodes.first { $0.id == "split" }).config
        changed["seed"] = .integer(43); reopened.configure(changed)
        XCTAssertTrue(reopened.previewStatus(for: "train").contains("stale"))
        reopened.undo()
        XCTAssertTrue(reopened.previewStatus(for: "train").contains("sources unchecked"))

        // The retained process owns busy state until exit; cancelled/late events cannot publish.
        var exit: ((Int32, String, PreparationInspectionResult?) -> Void)?
        reopened.resultRestorationExecute = { _, _, _, _, _, _, done in exit = done; return .init(cancel: {}) }
        reopened.restoreCompletedOutputs()
        XCTAssertTrue(reopened.isRestoringResults); XCTAssertTrue(workspace.isBusy)
        await waitUntil { exit != nil }
        reopened.cancelResultRestoration()
        XCTAssertTrue(workspace.isBusy)
        exit?(0, "", nil)
        XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(reopened.previews.count, 7)
        XCTAssertEqual(reopened.manifestURL, manifest)

        exit = nil
        reopened.restoreCompletedOutputs()
        await waitUntil { exit != nil }
        workspace.activeProjectID = UUID().uuidString
        XCTAssertTrue(workspace.isBusy, "Project change does not relinquish a live worker's busy lease")
        let competing = DataFlowWorkspaceStore(workspace: workspace)
        competing.runPreparation()
        XCTAssertFalse(competing.isPreparing)
        XCTAssertTrue(workspace.isBusy)
        exit?(0, "", nil)
        XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(reopened.previews.count, 7)
        XCTAssertEqual(reopened.manifestURL, manifest)
        let other = DataFlowWorkspaceStore(workspace: workspace)
        XCTAssertNil(other.manifestURL); XCTAssertTrue(other.previews.isEmpty)
        workspace.activeProjectID = store.projectID

        // Artifact corruption is checked by the worker before any restored preview is accepted.
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: originalManifest) as? [String: Any])
        let artifacts = try XCTUnwrap(object["artifacts"] as? [[String: Any]])
        let table = try XCTUnwrap(artifacts.first { $0["kind"] as? String == "table" }?["path"] as? String)
        try Data("{}".utf8).write(to: manifest.deletingLastPathComponent().appendingPathComponent(table))
        let corrupted = DataFlowWorkspaceStore(workspace: workspace); try await finish(corrupted)
        XCTAssertNil(corrupted.manifestURL); XCTAssertTrue(corrupted.previews.isEmpty)
        XCTAssertTrue(corrupted.statusMessage.contains("not restored"))
        XCTAssertEqual(try Data(contentsOf: store.completedResultURL), remembered)
    }
    @MainActor private func waitUntil(_ ready: () -> Bool) async {
        let deadline = Date().addingTimeInterval(5)
        while !ready() && Date() < deadline { try? await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertTrue(ready())
    }
    @MainActor func testRestorationTerminalReservationAndReferenceGuards() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        let store = DataFlowWorkspaceStore(workspace: workspace)
        let output = directory.appendingPathComponent("dataflow/runs/\(UUID().uuidString)/artifacts")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let manifest = output.appendingPathComponent("manifest.json")
        try JSONSerialization.data(withJSONObject: ["graphDigest": String(repeating: "a", count: 64)]).write(to: manifest)
        try DataFlowCodec.encodeGraph(store.project.graph).write(to: output.appendingPathComponent("dataflow.json"))
        try store.persistCompletedResult(manifest: manifest, graph: store.project.graph, bindings: [:], digests: [:])
        let reference = try XCTUnwrap(store.completedResult)
        let report = DataFlowCacheReport(policy: "off", executedNodeIDs: store.project.graph.nodes.map(\.id), reusedNodeIDs: [], nodes: [])
        let result = PreparationInspectionResult(outputs: Dictionary(uniqueKeysWithValues: store.project.graph.nodes.map { (WorkflowNodeAddress(domain: .data, localID: $0.id), ["marker": JSONValue.string("saved")]) }), cacheReport: report)
        var launches = 0, cancellation = 0
        store.resultRestorationExecute = { _, _, _, _, original, pin, exit in
            launches += 1; XCTAssertEqual(original, manifest); XCTAssertEqual(pin, reference.manifestSHA256)
            exit(0, "", result); exit(1, "duplicate failure", nil)
            return .init(cancel: { cancellation += 1 })
        }
        store.restoreCompletedOutputs(); XCTAssertTrue(workspace.isBusy)
        await waitUntil { !store.isBusy }
        XCTAssertEqual(launches, 1); XCTAssertEqual(cancellation, 0)
        XCTAssertEqual(store.manifestURL, manifest); XCTAssertTrue(store.restoredResult)
        XCTAssertTrue(store.statusMessage.contains("Historical"))
        var exits: [(Int32, String, PreparationInspectionResult?) -> Void] = []
        store.resultRestorationExecute = { _, _, _, _, _, _, exit in
            launches += 1; exits.append(exit); return .init(cancel: { cancellation += 1 })
        }
        store.restoreCompletedOutputs(); await waitUntil { exits.count == 1 }
        workspace.cancelDataPreparation = {}; let foreign = workspace.dataPreparationReservationID
        exits[0](1, "failure", nil)
        XCTAssertTrue(workspace.isPreparingData); XCTAssertEqual(workspace.dataPreparationReservationID, foreign)
        XCTAssertEqual(store.manifestURL, manifest)
        workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
        store.restoreCompletedOutputs(); await waitUntil { exits.count == 2 }
        exits[0](1, "obsolete", nil); XCTAssertTrue(workspace.isBusy)
        store.closeSourceInspection(); XCTAssertTrue(workspace.isBusy)
        exits[1](0, "", result); XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(store.manifestURL, manifest); XCTAssertEqual(cancellation, 1)
        store.inspectionClosed = false
        var unsafe = reference; unsafe.manifestRelativePath = "dataflow/runs/../artifacts/manifest.json"
        try JSONEncoder().encode(unsafe).write(to: store.completedResultURL)
        store.restoreCompletedOutputs(); await waitUntil { !store.isBusy }
        XCTAssertEqual(launches, 3)
        try JSONEncoder().encode(reference).write(to: store.completedResultURL)
        try Data("{}".utf8).write(to: manifest)
        store.restoreCompletedOutputs(); await waitUntil { !store.isBusy }
        XCTAssertEqual(launches, 3)
        XCTAssertEqual(store.manifestURL, manifest)
    }
}
