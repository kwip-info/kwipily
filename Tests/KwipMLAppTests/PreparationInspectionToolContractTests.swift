import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationInspectionToolContractTests: XCTestCase {
    private let digest = String(repeating: "a", count: 64)
    private func directory() throws -> URL {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(pointer) }
        return URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
            .appendingPathComponent("workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
    }
    private func arguments(_ root: URL, pin: String) -> [String] {
        ["--inspect-dataflow", root.appendingPathComponent("input/preparation/manifest.json").path,
         "--dataflow-manifest-sha256", pin]
    }
    private func file(_ name: String, bytes: Int = 2) -> WorkflowArchiveFile {
        .init(path: "input/preparation/" + name, bytes: bytes, sha256: digest)
    }
    func testExactOwnedArgumentsAndBoundedInventory() throws {
        let root = try directory(), args = arguments(root, pin: digest)
        let files = [file("manifest.json", bytes: 2_000_000)] + (0..<300).map { file("table\($0).json") }
        XCTAssertNoThrow(try WorkflowPreparationInspectionToolContract.validate(arguments: args, files: files, directory: root))
        XCTAssertEqual(try WorkflowPreparationInspectionToolContract.fileLimit(path: "input/preparation/manifest.json"), 2_000_000)
        XCTAssertEqual(try WorkflowPreparationInspectionToolContract.fileLimit(path: "input/preparation/table.json"), 50_000_000)
        for bad in [files + [file("extra.json")], [file("manifest.json")], [file("manifest.json", bytes: 2_000_001), file("a.json")],
                    [file("manifest.json"), file("a.json", bytes: 50_000_001)],
                    [file("manifest.json")] + (0..<5).map { file("large\($0).json", bytes: 50_000_000) },
                    [file("manifest.json"), file("a.json"), file("a.json")]] {
            XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.validate(arguments: args, files: bad, directory: root))
        }
        for badPath in ["input/graph.json", "input/preparation/../escape", "input/preparation/nested/a.json", "input/preparation/" + String(repeating: "x", count: 256)] {
            XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.fileLimit(path: badPath))
        }
        for extra in [["--registry", "/tmp/registry"], ["--output", "/tmp/output"], ["--source-bindings", "/tmp/bindings"], ["--execute-dataflow", "/tmp/graph"]] {
            XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.validate(arguments: args + extra, files: files, directory: root))
        }
        var outside = args; outside[1] = "/tmp/manifest.json"
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.validate(arguments: outside, files: files, directory: root))
        var wrongPin = args; wrongPin[3] = String(repeating: "b", count: 64)
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.validate(arguments: wrongPin, files: files, directory: root))
        let request = WorkflowToolRequest(operationID: root.lastPathComponent,
            projectID: root.deletingLastPathComponent().lastPathComponent, requestIdentity: digest,
            toolMode: .preparationInspection, arguments: args, inputFiles: files,
            executionBudget: try .resolve(global: .factory, projectMaximumRunSeconds: nil), remainingSeconds: 60)
        XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()), request)
    }
    func testCompletedHistoricalInventoryIsExactAndReverified() throws {
        let root = try directory()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("input/preparation"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) }
        let table = Data("{\"old\":true}".utf8), path = "input/preparation/table.json"
        func inventory(status: String = "completed") throws -> [WorkflowArchiveFile] {
            let manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": status,
                "graphDigest": String(repeating: "b", count: 64),
                "artifacts": [["path": "table.json", "bytes": table.count, "sha256": WorkflowArchiveCodec.hash(table), "kind": "table"]]]
            let bytes = try JSONSerialization.data(withJSONObject: manifest)
            try bytes.write(to: root.appendingPathComponent("input/preparation/manifest.json"))
            try table.write(to: root.appendingPathComponent(path))
            return [.init(path: "input/preparation/manifest.json", bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes)),
                    .init(path: path, bytes: table.count, sha256: WorkflowArchiveCodec.hash(table))]
        }
        let valid = try inventory(), args = arguments(root, pin: valid[0].sha256)
        XCTAssertNoThrow(try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: args, files: valid, directory: root))
        let extra = Data("{}".utf8); try extra.write(to: root.appendingPathComponent("input/preparation/extra.json"))
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: args,
            files: valid + [.init(path: "input/preparation/extra.json", bytes: extra.count, sha256: WorkflowArchiveCodec.hash(extra))], directory: root))
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: args, files: [valid[0]], directory: root))
        try Data("tampered".utf8).write(to: root.appendingPathComponent(path))
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: args, files: valid, directory: root))
        _ = try inventory()
        try FileManager.default.removeItem(at: root.appendingPathComponent(path))
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent(path), withDestinationURL: root.appendingPathComponent("input/preparation/manifest.json"))
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: args, files: valid, directory: root))
        try FileManager.default.removeItem(at: root.appendingPathComponent(path))
        let incomplete = try inventory(status: "failed")
        XCTAssertThrowsError(try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: arguments(root, pin: incomplete[0].sha256), files: incomplete, directory: root))
    }
}
