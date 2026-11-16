import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCapturedCacheTests: XCTestCase {
    private let fixtureDigest = String(repeating: "a", count: 64)
    private struct Fixture {
        let root: URL
        let run: URL
        let reference: WorkflowArchiveCacheReference
        let manifest: URL
        let graph: URL
    }
    private func bytes(_ value: Any) throws -> Data { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]) }
    private func object(_ url: URL) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]) }
    private func fixture(policy: String = "materialize") throws -> Fixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let project = UUID().uuidString, directory = UUID().uuidString
        let run = root.appendingPathComponent("workflow-runs/\(project)/\(directory)")
        let artifacts = run.appendingPathComponent("preparation/artifacts")
        try FileManager.default.createDirectory(at: artifacts, withIntermediateDirectories: true)
        let graph = DataFlowDocument(nodes: [
            .init(id: "source", operatorID: "kwip.data.sourceCSV", config: ["separator": .string(",")]),
            .init(id: "cast", operatorID: "kwip.data.columns", config: CurrentExecutionWorkflowTestFixture.columns([("a","float64")]))],
            edges: [.init(from: .init(node: "source", port: "table"), to: .init(node: "cast", port: "table"))])
        let graphBytes = try WorkflowArchiveCodec.encode(graph)
        let graphURL = artifacts.appendingPathComponent("dataflow.json")
        try graphBytes.write(to: graphURL)
        let reuse = policy == "reuse"
        let nodes: [[String: Any]] = policy == "off" ? [] : [
            ["nodeID": "source", "cacheKey": fixtureDigest, "status": "executed", "reason": "sourceRevalidated"],
            ["nodeID": "cast", "cacheKey": fixtureDigest, "status": reuse ? "reused" : "executed", "reason": reuse ? "verifiedHit" : "notMaterialized"]]
        let cache: [String: Any] = ["schemaVersion": 1, "policy": policy, "implementationDigest": fixtureDigest,
            "sourceManifestSHA256": reuse ? fixtureDigest as Any : NSNull(), "nodes": nodes,
            "executedNodeIDs": reuse ? ["source"] : ["source", "cast"], "reusedNodeIDs": reuse ? ["cast"] : []]
        let inventory: [[String: Any]] = [
            ["path": "dataflow.json", "bytes": graphBytes.count, "sha256": WorkflowArchiveCodec.hash(graphBytes), "kind": "graph"],
            ["path": "unused-table.json", "bytes": 123, "sha256": fixtureDigest, "kind": "table", "nodeID": "cast", "port": "table"]]
        let manifestBytes = try bytes(["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed",
            "graphDigest": WorkflowArchiveCodec.hash(graphBytes), "registryDigest": fixtureDigest, "numpyVersion": "recorded-version",
            "cache": cache, "artifacts": inventory])
        let manifestURL = artifacts.appendingPathComponent("manifest.json")
        try manifestBytes.write(to: manifestURL)
        let pin = WorkflowArchiveCodec.hash(manifestBytes)
        let outcome = WorkflowArchiveOutcome(projectID: project, directoryID: directory, runID: UUID().uuidString,
            requestIdentity: fixtureDigest, status: .completed, terminalPhase: .training,
            startedAt: "2026-10-05T12:00:00.000Z", finishedAt: "2026-10-05T12:00:01.000Z",
            artifacts: [
                .init(path: "preparation/artifacts/manifest.json", bytes: manifestBytes.count, sha256: pin),
                .init(path: "preparation/artifacts/dataflow.json", bytes: graphBytes.count, sha256: WorkflowArchiveCodec.hash(graphBytes)),
                .init(path: "training/artifacts/manifest.json", bytes: 100, sha256: fixtureDigest)],
            capture: .init(path: "capture.json", bytes: 100, sha256: fixtureDigest), preparationManifestSHA256: pin,
            modelManifestSHA256: fixtureDigest, workerRunID: UUID().uuidString)
        try WorkflowArchiveCodec.encodeOutcome(outcome).write(to: run.appendingPathComponent("outcome.json"))
        return .init(root: root, run: run, reference: .init(projectID: project, directoryID: directory, preparationManifestSHA256: pin), manifest: manifestURL, graph: graphURL)
    }
    func testAcceptsMaterializeAndReuseWithoutReadingUnusedArtifactsOrCurrentRuntime() throws {
        for policy in ["materialize", "reuse"] {
            let f = try fixture(policy: policy); defer { try? FileManager.default.removeItem(at: f.root) }
            let captured = try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root)
            XCTAssertEqual(captured.reference, f.reference)
            XCTAssertEqual(captured.manifestURL.standardizedFileURL.path, f.manifest.standardizedFileURL.path)
            XCTAssertNoThrow(try captured.revalidate())
            XCTAssertFalse(FileManager.default.fileExists(atPath: f.run.appendingPathComponent("training/artifacts/manifest.json").path))
        }
    }
    func testCompletedPreparationRemainsSelectableAfterTrainingFailureOrCancellation() throws {
        for status in ["failed", "cancelled"] {
            let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
            let url = f.run.appendingPathComponent("outcome.json")
            var receipt = try object(url)
            receipt["status"] = status
            receipt.removeValue(forKey: "modelManifestSHA256")
            receipt.removeValue(forKey: "workerRunID")
            if status == "failed" { receipt["error"] = "Training failed after successful preparation." }
            let entries = try XCTUnwrap(receipt["artifacts"] as? [[String: Any]])
            receipt["artifacts"] = entries.filter { ($0["path"] as? String)?.hasPrefix("preparation/") == true }
            try bytes(receipt).write(to: url)
            XCTAssertNoThrow(try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root))
        }
    }
    func testRejectsOffMissingAndMalformedPins() throws {
        let off = try fixture(policy: "off"); defer { try? FileManager.default.removeItem(at: off.root) }
        XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: off.reference, workspaceDirectory: off.root))
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        for pin in [fixtureDigest, String(repeating: "A", count: 64), "short"] {
            let ref = WorkflowArchiveCacheReference(projectID: f.reference.projectID, directoryID: f.reference.directoryID, preparationManifestSHA256: pin)
            XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: ref, workspaceDirectory: f.root))
        }
        try FileManager.default.removeItem(at: f.manifest)
        XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root))
    }
    func testReceiptIdentityAndPinMustMatchManagedReference() throws {
        for key in ["projectID", "directoryID", "preparationManifestSHA256"] {
            let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
            let url = f.run.appendingPathComponent("outcome.json")
            var receipt = try object(url); receipt[key] = key == "preparationManifestSHA256" ? fixtureDigest : UUID().uuidString
            try bytes(receipt).write(to: url)
            XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root), key)
        }
    }
    func testGraphInventoryAndReportCannotBeRepinnedIntoContradictorySelection() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let original = try Data(contentsOf: f.graph)
        try Data("{}".utf8).write(to: f.graph)
        XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root))
        try original.write(to: f.graph)
        var manifest = try object(f.manifest)
        var cache = try XCTUnwrap(manifest["cache"] as? [String: Any])
        cache["reusedNodeIDs"] = ["source"]; manifest["cache"] = cache
        let changed = try bytes(manifest), pin = WorkflowArchiveCodec.hash(changed)
        try changed.write(to: f.manifest)
        let url = f.run.appendingPathComponent("outcome.json")
        var receipt = try object(url); receipt["preparationManifestSHA256"] = pin
        var entries = try XCTUnwrap(receipt["artifacts"] as? [[String: Any]])
        let index = try XCTUnwrap(entries.firstIndex { $0["path"] as? String == "preparation/artifacts/manifest.json" })
        entries[index]["sha256"] = pin; entries[index]["bytes"] = changed.count; receipt["artifacts"] = entries
        try bytes(receipt).write(to: url)
        let reference = WorkflowArchiveCacheReference(projectID: f.reference.projectID, directoryID: f.reference.directoryID, preparationManifestSHA256: pin)
        XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: reference, workspaceDirectory: f.root))
    }
    func testAnchoredReadsRejectDirectoryAndLeafSymlinks() throws {
        for relative in ["preparation", "preparation/artifacts/manifest.json", "preparation/artifacts/dataflow.json", "outcome.json"] {
            let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
            let original = f.run.appendingPathComponent(relative), moved = f.root.appendingPathComponent("moved")
            try FileManager.default.moveItem(at: original, to: moved)
            try FileManager.default.createSymbolicLink(at: original, withDestinationURL: moved)
            XCTAssertThrowsError(try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root), relative)
        }
    }
    func testRevalidationRejectsChangedManifestOrReceiptAfterCapture() throws {
        for relative in ["outcome.json", "preparation/artifacts/manifest.json", "preparation/artifacts/dataflow.json"] {
            let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
            let captured = try WorkflowCapturedCache.capture(reference: f.reference, workspaceDirectory: f.root)
            try Data("{}".utf8).write(to: f.run.appendingPathComponent(relative))
            XCTAssertThrowsError(try captured.revalidate(), relative)
        }
    }
}
