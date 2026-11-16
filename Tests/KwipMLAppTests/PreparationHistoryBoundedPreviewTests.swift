import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationHistoryBoundedPreviewTests: XCTestCase {
    private let pin = String(repeating: "a", count: 64)
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func fixture(change: (inout [String: JSONValue]) -> Void = { _ in }, sidecarChange: (inout [String: JSONValue]) -> Void = { _ in }) throws -> WorkflowDataPreparationHistory.Entry {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let graph = DataFlowDocument(nodes: [.init(id: "source", operatorID: "kwip.data.sourceCSV")])
        let graphBytes = try DataFlowCodec.encodeGraph(graph), graphDigest = WorkflowArchiveCodec.hash(graphBytes)
        var output: [String: JSONValue] = ["kind": .string("table"), "lineage": .object(["role": .string("unassigned"), "sourceDigest": .string(pin), "membershipDigest": .string(pin), "dataVersion": .string(pin), "partitionDigest": .string(""), "fittedStateDigests": .array([])]), "preview": .object(["partition": .string("unassigned"), "dataVersion": .string(pin), "rowCount": 500_001, "columns": .array([.object(["name": .string("x"), "dtype": .string("int64"), "nullable": false])]), "rows": .array([.array([1])]), "diagnostics": .array([]), "diagnosticCount": 0])]
        change(&output)
        var previewEnvelope: [String: JSONValue] = ["schemaVersion": .integer(1), "graphDigest": .string(graphDigest), "outputs": .object(["source": .object(["table": .object(output)])])]
        sidecarChange(&previewEnvelope)
        let sidecar = try WorkflowArchiveCodec.encode(previewEnvelope)
        // Deliberately opaque table bytes prove the new History path never JSON-decodes the full table.
        let contents = ["dataflow.json": graphBytes, "dataflow-operators.json": try Data(contentsOf: repo.appendingPathComponent("runner/dataflow-operators.json")), "node-000.json": Data("opaque full table".utf8), "previews.json": sidecar]
        let files = contents.map { WorkflowArchiveFile(path: $0.key, bytes: $0.value.count, sha256: WorkflowArchiveCodec.hash($0.value)) }
        for (name, bytes) in contents { try bytes.write(to: root.appendingPathComponent(name)) }
        let artifacts: [[String: Any]] = files.map { file in
            var item: [String: Any] = ["path": file.path, "bytes": file.bytes, "sha256": file.sha256, "kind": file.path == "previews.json" ? "previews" : "metadata"]
            if file.path == "node-000.json" { item["kind"] = "table"; item["nodeID"] = "source"; item["port"] = "table" }
            return item
        }
        let manifest = try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "graphDigest": graphDigest, "artifacts": artifacts], options: [.sortedKeys])
        try manifest.write(to: root.appendingPathComponent("manifest.json"))
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [.init(address: .init(domain: .data, localID: "source"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:]))])
        return .init(id: UUID().uuidString, completedAt: Date(), document: document, requestIdentity: pin, artifactDirectory: root, manifestSHA256: WorkflowArchiveCodec.hash(manifest), savedOutputs: [], artifacts: files)
    }
    func testLargeHistoryPreviewReadsOnlyBoundedSidecarAndMetadata() throws {
        let entry = try fixture()
        let result = try WorkflowDataPreparationHistory.readOutputs(entry)
        let value = try XCTUnwrap(result[.init(domain: .data, localID: "source")]?["table"])
        guard case .object(let output) = value, case .object(let preview) = output["preview"] else { return XCTFail("Missing preview") }
        XCTAssertEqual(preview["rowCount"], 500_001)
        XCTAssertEqual(preview["rows"], .array([.array([1])]))
    }
    func testSidecarTamperingAndInvalidPortShapesAreRejected() throws {
        let entry = try fixture()
        try Data("{}".utf8).write(to: entry.artifactDirectory.appendingPathComponent("previews.json"))
        XCTAssertThrowsError(try WorkflowDataPreparationHistory.readOutputs(entry))
        let invalid = try fixture(change: { output in output["kind"] = .string("state") })
        XCTAssertThrowsError(try WorkflowDataPreparationHistory.readOutputs(invalid))
        for mutation: (inout [String: JSONValue]) -> Void in [
            { $0["outputs"] = .object([:]) },
            { $0["outputs"] = .object(["source": .object(["foreignPort": .null])]) },
            { $0["graphDigest"] = .string(String(repeating: "b", count: 64)) }
        ] {
            let forged = try fixture(sidecarChange: mutation)
            XCTAssertThrowsError(try WorkflowDataPreparationHistory.readOutputs(forged))
        }
    }
    func testLegacyHistoryWithoutSidecarRetainsExistingTableReader() throws {
        let entry = try fixture()
        let table: [String: Any] = ["schemaVersion": 1, "kind": "table", "columns": [["name": "x", "dtype": "int64", "nullable": false]], "rows": [[7]], "lineage": ["role": "unassigned", "dataVersion": pin, "diagnostics": []]]
        let bytes = try JSONSerialization.data(withJSONObject: table)
        try bytes.write(to: entry.artifactDirectory.appendingPathComponent("node-000.json"))
        var files = entry.artifacts.filter { $0.path != "previews.json" && $0.path != "node-000.json" }
        files.append(.init(path: "node-000.json", bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes)))
        let manifestURL = entry.artifactDirectory.appendingPathComponent("manifest.json")
        var manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        var artifacts = try XCTUnwrap(manifest["artifacts"] as? [[String: Any]]).filter { $0["path"] as? String != "previews.json" }
        for index in artifacts.indices where artifacts[index]["path"] as? String == "node-000.json" {
            artifacts[index]["bytes"] = bytes.count; artifacts[index]["sha256"] = WorkflowArchiveCodec.hash(bytes)
        }
        manifest["artifacts"] = artifacts
        let raw = try JSONSerialization.data(withJSONObject: manifest); try raw.write(to: manifestURL)
        let legacy = WorkflowDataPreparationHistory.Entry(id: entry.id, completedAt: entry.completedAt, document: entry.document, requestIdentity: entry.requestIdentity, artifactDirectory: entry.artifactDirectory, manifestSHA256: WorkflowArchiveCodec.hash(raw), savedOutputs: [], artifacts: files)
        let result = try WorkflowDataPreparationHistory.readOutputs(legacy)
        guard case .object(let value) = result[.init(domain: .data, localID: "source")]?["table"], case .object(let preview) = value["preview"] else { return XCTFail("Missing legacy preview") }
        XCTAssertEqual(preview["rowCount"], 1)
        XCTAssertEqual(preview["rows"], .array([.array([7])]))
    }

}
