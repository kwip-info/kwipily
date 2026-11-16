import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationInspectionStreamTests: XCTestCase {
    private let digest = String(repeating: "a", count: 64)
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func fixture() throws -> (URL, String) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let graph = try DataFlowCodec.encodeGraph(.init(nodes: [.init(id: "source", operatorID: "kwip.data.sourceCSV")]))
        let registry = try Data(contentsOf: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let files = ["dataflow.json": graph, "dataflow-operators.json": registry]
        for (name, bytes) in files { try bytes.write(to: root.appendingPathComponent(name)) }
        let cache: [String: Any] = ["schemaVersion": 1, "policy": "off", "implementationDigest": digest,
            "sourceManifestSHA256": NSNull(), "nodes": [], "executedNodeIDs": ["source"], "reusedNodeIDs": []]
        let manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed",
            "graphDigest": digest, "registryDigest": digest, "numpyVersion": "2.5.3", "cache": cache,
            "artifacts": files.map { ["path": $0.key, "bytes": $0.value.count, "sha256": WorkflowArchiveCodec.hash($0.value)] }]
        let bytes = try JSONSerialization.data(withJSONObject: manifest)
        try bytes.write(to: root.appendingPathComponent("manifest.json"))
        return (root, WorkflowArchiveCodec.hash(bytes))
    }
    private func events(_ expectation: PreparationInspectionExpectation) throws -> [String] {
        let output: [String: Any] = ["kind": "table", "preview": ["partition": "unassigned", "dataVersion": digest,
            "rowCount": 1, "columns": [["name": "x", "dtype": "int64", "nullable": false]], "rows": [[1]], "diagnostics": [], "diagnosticCount": 0],
            "lineage": ["role": "unassigned", "sourceDigest": digest, "membershipDigest": digest, "dataVersion": digest,
                "fittedStateDigests": [], "partitionDigest": ""]]
        var cache = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(expectation.cacheReport)) as? [String: Any])
        if let pin = expectation.cacheReport.sourceManifestSHA256 { cache["sourceManifestSHA256"] = pin }
        else { cache["sourceManifestSHA256"] = NSNull() }
        let values: [[String: Any]] = [["event": "dataflowInspectionStarted"],
            ["event": "dataflowPreview", "nodeID": "source", "outputs": ["table": output],
             "execution": ["nodeID": "source", "status": "executed", "reason": "disabled", "cacheKey": NSNull()]],
            ["event": "dataflowInspectionCompleted", "manifestSHA256": expectation.manifestSHA256, "graphDigest": expectation.graphDigest,
             "cache": cache, "provenance": ["historical": true, "sourceRevalidated": false, "cacheReuseVerified": false,
                 "registryDigest": expectation.registryDigest, "numpyVersion": expectation.numpyVersion]]]
        return try values.map { String(decoding: try JSONSerialization.data(withJSONObject: $0), as: UTF8.self) }
    }
    func testPinnedMetadataReadBuildsReferenceFreeIdentityAndRejectsChangedGraphRegistryAndPin() throws {
        let (root, pin) = try fixture(), manifest = root.appendingPathComponent("manifest.json")
        let expectation = try PreparationInspectionExpectation.read(manifestURL: manifest, manifestSHA256: pin)
        XCTAssertEqual(expectation.nodes.map(\.address), [.init(domain: .data, localID: "source")])
        XCTAssertEqual(expectation.nodes.first?.operatorID, "kwip.data.sourceCSV")
        XCTAssertEqual(expectation.nodes.first?.portKinds, ["table": "table"])
        XCTAssertEqual(expectation.nodes.first?.name, "source")
        XCTAssertEqual(expectation.cacheReport.executedNodeIDs, ["source"])
        XCTAssertThrowsError(try PreparationInspectionExpectation.read(manifestURL: manifest, manifestSHA256: String(repeating: "0", count: 64)))
        for name in ["dataflow.json", "dataflow-operators.json"] {
            let url = root.appendingPathComponent(name), original = try Data(contentsOf: url)
            try Data("{}".utf8).write(to: url)
            XCTAssertThrowsError(try PreparationInspectionExpectation.read(manifestURL: manifest, manifestSHA256: pin), name)
            try original.write(to: url)
        }
        let registry = root.appendingPathComponent("dataflow-operators.json")
        let copy = root.appendingPathComponent("registry-copy.json")
        try FileManager.default.moveItem(at: registry, to: copy)
        try FileManager.default.createSymbolicLink(at: registry, withDestinationURL: copy)
        XCTAssertThrowsError(try PreparationInspectionExpectation.read(manifestURL: manifest, manifestSHA256: pin))
    }
    func testReferenceFreeStreamPublishesAllOutputsOnlyOnceAfterSuccessfulExit() throws {
        let (root, pin) = try fixture()
        let expectation = try PreparationInspectionExpectation.read(manifestURL: root.appendingPathComponent("manifest.json"), manifestSHA256: pin)
        let lines = try events(expectation)
        var stream = PreparationInspectionStream(expectation: expectation)
        for line in lines { try stream.receive(line) }
        let result = try stream.finish(exitCode: 0)
        XCTAssertEqual(result.outputs.keys.sorted { $0.localID < $1.localID }, [.init(domain: .data, localID: "source")])
        XCTAssertEqual(result.cacheReport, expectation.cacheReport)
        XCTAssertThrowsError(try stream.finish(exitCode: 0))
        XCTAssertThrowsError(try stream.receive(lines[0]))
        var failed = PreparationInspectionStream(expectation: expectation)
        for line in lines { try failed.receive(line) }
        XCTAssertThrowsError(try failed.finish(exitCode: 124))
        XCTAssertThrowsError(try failed.finish(exitCode: 0))
    }
    func testMissingDuplicateOrFalseHistoricalEvidenceCannotBeRepairedByLaterCompletion() throws {
        let (root, pin) = try fixture()
        let expectation = try PreparationInspectionExpectation.read(manifestURL: root.appendingPathComponent("manifest.json"), manifestSHA256: pin)
        let lines = try events(expectation)
        var incomplete = PreparationInspectionStream(expectation: expectation)
        try incomplete.receive(lines[0])
        XCTAssertThrowsError(try incomplete.receive(lines[2]))
        XCTAssertThrowsError(try incomplete.finish(exitCode: 0))
        var duplicate = PreparationInspectionStream(expectation: expectation)
        try duplicate.receive(lines[0]); try duplicate.receive(lines[1])
        XCTAssertThrowsError(try duplicate.receive(lines[1]))
        XCTAssertThrowsError(try duplicate.receive(lines[2]))
        for key in ["sourceRevalidated", "cacheReuseVerified"] {
            var value = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(lines[2].utf8)) as? [String: Any])
            var provenance = try XCTUnwrap(value["provenance"] as? [String: Any]); provenance[key] = true; value["provenance"] = provenance
            var invalid = PreparationInspectionStream(expectation: expectation)
            try invalid.receive(lines[0]); try invalid.receive(lines[1])
            XCTAssertThrowsError(try invalid.receive(String(decoding: JSONSerialization.data(withJSONObject: value), as: UTF8.self)))
            XCTAssertThrowsError(try invalid.receive(lines[2]))
        }
    }
}
