import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPreparationToolContractTests: XCTestCase {
    func testModelInputsMustBeCapturedRequiredNodesAndMatchPhysicalReceipt() throws {
        let root=try directory()
        XCTAssertNoThrow(try WorkflowPreparationToolContract.validate(arguments:arguments(root)+["--dataflow-required-nodes","[\"fit\"]","--dataflow-model-input-nodes","[\"fit\"]"],files:controls,directory:root))
        for raw in ["[\"other\"]","[\"fit\",\"fit\"]","[false]"] {
            XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments:arguments(root)+["--dataflow-required-nodes","[\"fit\"]","--dataflow-model-input-nodes",raw],files:controls,directory:root))
        }
        let receipt:[String:JSONValue]=["modelInputNodes":.array([.string("fit")]),"artifacts":.array([.object(["kind":.string("table"),"nodeID":.string("fit")])])]
        XCTAssertNoThrow(try WorkflowCompletedPreparationInventory.verifyModelInputs(data:WorkflowArchiveCodec.encode(receipt),expected:["fit"]))
        XCTAssertThrowsError(try WorkflowCompletedPreparationInventory.verifyModelInputs(data:WorkflowArchiveCodec.encode(receipt),expected:[]))
        var changed=receipt;changed["artifacts"] = .array([.object(["kind":.string("table"),"nodeID":.string("fit"),"tableStorageVersion":1])])
        XCTAssertThrowsError(try WorkflowCompletedPreparationInventory.verifyModelInputs(data:WorkflowArchiveCodec.encode(changed),expected:["fit"]))
    }
    private let digest = String(repeating: "a", count: 64)
    private func directory() throws -> URL {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(pointer) }
        return URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString).appendingPathComponent("workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
    }
    private func arguments(_ root: URL, policy: String = "off") -> [String] {
        ["--execute-dataflow", root.appendingPathComponent("input/graph.json").path,
         "--source-bindings", root.appendingPathComponent("input/bindings.json").path,
         "--dataflow-node-labels", root.appendingPathComponent("input/node-labels.json").path,
         "--output", root.appendingPathComponent("artifacts").path, "--dataflow-cache-policy", policy]
    }
    private func file(_ path: String, _ bytes: Int = 2) -> WorkflowArchiveFile { .init(path: path, bytes: bytes, sha256: digest) }
    private var controls: [WorkflowArchiveFile] { [file("input/graph.json"), file("input/bindings.json"), file("input/node-labels.json")] }
    func testMeasurementCollectionIsExplicitBooleanAndDoesNotPermitNewPaths() throws {
        let root=try directory()
        for raw in ["true","false"] {
            XCTAssertNoThrow(try WorkflowPreparationToolContract.validate(arguments:arguments(root)+["--dataflow-measurements",raw],files:controls,directory:root))
        }
        for raw in ["1","TRUE","/private/unreviewed",""] {
            XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments:arguments(root)+["--dataflow-measurements",raw],files:controls,directory:root))
        }
    }
    func testCacheAllowanceIsCanonicalBoundedAndNeverAllowedWhenOff() throws {
        let root = try directory()
        for raw in ["1", "200000000"] {
            XCTAssertNoThrow(try WorkflowPreparationToolContract.validate(arguments: arguments(root, policy: "materialize") + ["--dataflow-cache-limit-bytes", raw], files: controls, directory: root))
        }
        for raw in ["0", "-1", "200000001", "01", "1.0", "true"] {
            XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments: arguments(root, policy: "materialize") + ["--dataflow-cache-limit-bytes", raw], files: controls, directory: root))
        }
        XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments: arguments(root) + ["--dataflow-cache-limit-bytes", "1"], files: controls, directory: root))
    }
    func testPreserves128SourcesAndRejectsExcessBoundsAndMixedArguments() throws {
        let root = try directory(), args = arguments(root)
        let sources = (0..<128).map { file("input/sources/\(digest)/source\($0).csv", 10_000_000) }
        XCTAssertNoThrow(try WorkflowPreparationToolContract.validate(arguments: args, files: controls + sources, directory: root))
        XCTAssertEqual(try WorkflowPreparationToolContract.fileLimit(path: sources[0].path), 10_000_000)
        XCTAssertEqual(try WorkflowPreparationToolContract.fileLimit(path: "input/bindings.json"), 1_000_000)
        XCTAssertThrowsError(try WorkflowPreparationToolContract.fileLimit(path: "input/sources/\(digest)/" + String(repeating: "a", count: 256)))
        XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments: args, files: controls + sources + [file("input/sources/\(digest)/extra.csv")], directory: root))
        for extra in [["--epochs", "1"], ["--output", "/tmp/escape"], ["--retain-partial-dataflow"], ["--dataflow-reuse-manifest", "/tmp/manifest.json"]] {
            XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments: args + extra, files: controls, directory: root))
        }
        XCTAssertThrowsError(try WorkflowPreparationToolContract.fileLimit(path: "input/sources/\(digest)/../escape"))
        XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments: args, files: controls + [file("input/sources/\(String(repeating: "b", count: 64))/source.csv")], directory: root))
    }
    func testStagedGraphBindingsLabelsAndUsedSourceInventoryAreExact() throws {
        let root = try directory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) }
        let source = Data("x,y\n1,2\n".utf8), sourcePath = "input/sources/\(WorkflowArchiveCodec.hash(source))/data.csv"
        let graph = try DataFlowCodec.encodeGraph(.init(nodes: [.init(id: "source", operatorID: "kwip.data.sourceCSV")]))
        func inputs(bindings: [String: String], labels: [String: String]) throws -> [WorkflowArchiveFile] {
            let data = ["input/graph.json": graph, "input/bindings.json": try JSONEncoder().encode(bindings),
                        "input/node-labels.json": try JSONEncoder().encode(labels), sourcePath: source]
            for (path, bytes) in data {
                let url = root.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try bytes.write(to: url)
            }
            return data.map { .init(path: $0.key, bytes: $0.value.count, sha256: WorkflowArchiveCodec.hash($0.value)) }
        }
        let binding = ["source": root.appendingPathComponent(sourcePath).path]
        let valid = try inputs(bindings: binding, labels: ["source": "Source table"])
        XCTAssertNoThrow(try WorkflowPreparationToolContract.verifyInputs(arguments: arguments(root), files: valid, directory: root))
        for bindings in [[:], ["wrong": root.appendingPathComponent(sourcePath).path], ["source": "/tmp/data.csv"]] {
            let files = try inputs(bindings: bindings, labels: [:])
            XCTAssertThrowsError(try WorkflowPreparationToolContract.verifyInputs(arguments: arguments(root), files: files, directory: root))
        }
        for labels in [["wrong": "Missing node"], ["source": "   "], ["source": String(repeating: "x", count: 201)]] {
            let files = try inputs(bindings: binding, labels: labels)
            XCTAssertThrowsError(try WorkflowPreparationToolContract.verifyInputs(arguments: arguments(root), files: files, directory: root))
        }
    }
    func testEmptyGraphCannotBecomeAWorkerRequest() throws {
        let root = try directory()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("input"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) }
        let data = ["input/graph.json": try DataFlowCodec.encodeGraph(.init()), "input/bindings.json": Data("{}".utf8), "input/node-labels.json": Data("{}".utf8)]
        for (path, bytes) in data { try bytes.write(to: root.appendingPathComponent(path)) }
        let files = data.map { WorkflowArchiveFile(path: $0.key, bytes: $0.value.count, sha256: WorkflowArchiveCodec.hash($0.value)) }
        XCTAssertThrowsError(try WorkflowPreparationToolContract.verifyInputs(arguments: arguments(root), files: files, directory: root))
    }
    func testReuseInventoryRequiresCompletedManifestButAllowsDifferentPriorGraph() throws {
        let root = try directory()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("input/reuse"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) }
        let data = Data("{}".utf8), entry = WorkflowArchiveFile(path: "table.json", bytes: data.count, sha256: WorkflowArchiveCodec.hash(data))
        let manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "graphDigest": String(repeating: "b", count: 64),
            "artifacts": [["path": entry.path, "bytes": entry.bytes, "sha256": entry.sha256, "kind": "table"]]]
        let bytes = try JSONSerialization.data(withJSONObject: manifest)
        let files = [WorkflowArchiveFile(path: "input/reuse/manifest.json", bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes)),
                     WorkflowArchiveFile(path: "input/reuse/table.json", bytes: entry.bytes, sha256: entry.sha256)]
        let args = arguments(root, policy: "reuse") + ["--dataflow-reuse-manifest", root.appendingPathComponent("input/reuse/manifest.json").path,
            "--dataflow-reuse-manifest-sha256", WorkflowArchiveCodec.hash(bytes)]
        XCTAssertNoThrow(try WorkflowPreparationToolContract.validate(arguments: args, files: controls + files, directory: root))
        XCTAssertNoThrow(try WorkflowCompletedPreparationInventory.verify(data: bytes, files: files, prefix: "input/reuse/"))
        XCTAssertThrowsError(try WorkflowCompletedPreparationInventory.verify(data: bytes, files: files, prefix: "input/reuse/", expectedGraphDigest: digest))
        XCTAssertThrowsError(try WorkflowCompletedPreparationInventory.verify(data: bytes, files: Array(files.dropLast()), prefix: "input/reuse/"))
        XCTAssertThrowsError(try WorkflowPreparationToolContract.validate(arguments: arguments(root), files: controls + files, directory: root))
    }
}
