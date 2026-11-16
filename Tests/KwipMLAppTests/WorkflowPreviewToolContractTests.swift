import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPreviewToolContractTests: XCTestCase {
    private let fixtureHash = String(repeating: "a", count: 64)
    private var directory: URL { URL(fileURLWithPath: "/private/tmp/workspace/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)") }
    private func file(_ path: String, _ bytes: Int = 2) -> WorkflowArchiveFile { .init(path: path, bytes: bytes, sha256: fixtureHash) }
    private func base(_ root: URL) -> [String] { ["--graph", root.appendingPathComponent("input/graph.json").path, "--output", root.appendingPathComponent("artifacts").path, "--device", "cpu", "--seed", "42"] }
    private let sampling = ["--sample-limit", "0", "--split-seed", "42", "--validation-percent", "20"]
    func testBundledCSVWeightsAndResumeRequireExactInventories() throws {
        let root = directory, base = base(root)
        let preview = ["--preview", "--stop-at", "hidden"]
        XCTAssertNoThrow(try WorkflowPreviewToolContract.validate(arguments: base + preview + sampling, files: [file("input/graph.json")], directory: root))
        let csv = ["--dataset-csv", root.appendingPathComponent("input/data/my sample.csv").path, "--feature-columns-json", "[\"x\",\"y\"]", "--target-column", "label"]
        let csvFiles = [file("input/graph.json"), file("input/data/my sample.csv")]
        XCTAssertNoThrow(try WorkflowPreviewToolContract.validate(arguments: base + preview + sampling + csv, files: csvFiles, directory: root))
        let resume = ["--resume-from", "hidden", "--cache", root.appendingPathComponent("input/cache/activation-cache.pt").path,
                      "--checkpoint", root.appendingPathComponent("input/weights/checkpoint.pt").path]
        let files = [file("input/graph.json"), file("input/cache/activation-cache.pt"), file("input/cache/activation-cache.json"), file("input/weights/checkpoint.pt"), file("input/weights/graph.json")]
        XCTAssertNoThrow(try WorkflowPreviewToolContract.validate(arguments: base + resume + sampling, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: base + resume + sampling, files: Array(files.dropLast()), directory: root))
        XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: base + preview + sampling, files: csvFiles, directory: root))
    }
    func testCSVRetainsLongHeaderCompatibilityWithinArgumentBound() throws {
        let root = directory, longName = String(repeating: "x", count: 201), target = String(repeating: "y", count: 201)
        let names = String(decoding: try JSONEncoder().encode([longName, "second"]), as: UTF8.self)
        let args = base(root) + ["--preview", "--stop-at", "hidden"] + sampling + ["--dataset-csv", root.appendingPathComponent("input/data/source.csv").path, "--feature-columns-json", names, "--target-column", target]
        XCTAssertNoThrow(try WorkflowPreviewToolContract.validate(arguments: args,
            files: [file("input/graph.json"), file("input/data/source.csv")], directory: root))
        for key in ["--sample-limit", "--validation-percent"] {
            var changed = args
            let index = try XCTUnwrap(changed.firstIndex(of: key)) + 1
            changed[index] = "+" + changed[index]
            XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: changed,
                files: [file("input/graph.json"), file("input/data/source.csv")], directory: root))
        }
    }
    func testRejectsMixedModesDuplicateUnknownFlagsAndEscapedPaths() throws {
        let root = directory, valid = base(root) + ["--preview", "--stop-at", "hidden"] + sampling
        for extra in [["--preview"], ["--epochs", "1"], ["--registry", "/tmp/registry"], ["--resume-from", "hidden", "--cache", root.appendingPathComponent("input/cache/activation-cache.pt").path], ["--checkpoint", "/tmp/checkpoint.pt"]] {
            XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: valid + extra, files: [file("input/graph.json")], directory: root))
        }
        for replacement in ["/tmp/graph.json", root.appendingPathComponent("input/../graph.json").path] {
            var args = valid; args[1] = replacement
            XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: args, files: [file("input/graph.json")], directory: root))
        }
        XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: valid, files: [file("input/graph.json"), file("input/graph.json")], directory: root))
    }
    func testPreparedTupleAndInventoryBounds() throws {
        let root = directory
        let prepared = ["--prepared-dataflow", root.appendingPathComponent("input/preparation/manifest.json").path,
                        "--prepared-manifest-sha256", fixtureHash, "--prepared-graph-digest", fixtureHash,
                        "--prepared-training-node", "split", "--prepared-training-port", "training",
                        "--prepared-validation-node", "split", "--prepared-validation-port", "validation",
                        "--feature-columns-json", "[\"x\",\"y\"]", "--target-column", "label"]
        let args = base(root) + ["--preview", "--stop-at", "hidden"] + prepared
        let files = [file("input/graph.json"), file("input/preparation/manifest.json"), file("input/preparation/table.json")]
        XCTAssertNoThrow(try WorkflowPreviewToolContract.validate(arguments: args, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: args + sampling, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: args, files: files + [file("input/preparation/large.json", 50_000_001)], directory: root))
        let oversized = (0..<5).map { file("input/preparation/table\($0).json", 50_000_000) }
        XCTAssertThrowsError(try WorkflowPreviewToolContract.validate(arguments: args, files: files + oversized, directory: root))
        XCTAssertThrowsError(try WorkflowPreviewToolContract.fileLimit(path: "input/preparation/../outside"))
        XCTAssertEqual(try WorkflowPreviewToolContract.fileLimit(path: "input/weights/checkpoint.pt"), 100_000_000)
        XCTAssertEqual(try WorkflowPreviewToolContract.fileLimit(path: "input/data/source.csv"), 10_000_000)
    }
    func testStreamedNestedInputVerificationRejectsChangedAndLinkedFiles() throws {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil)); defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("input/data"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data(repeating: 65, count: 2_100_000)
        let path = root.appendingPathComponent("input/data/source.csv")
        try bytes.write(to: path)
        let entry = WorkflowArchiveFile(path: "input/data/source.csv", bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes))
        XCTAssertNoThrow(try WorkflowPreviewToolContract.verifyInputs(files: [entry], directory: root))
        try Data("changed".utf8).write(to: path)
        XCTAssertThrowsError(try WorkflowPreviewToolContract.verifyInputs(files: [entry], directory: root))
        try FileManager.default.removeItem(at: path)
        let outside = root.appendingPathComponent("source.csv"); try bytes.write(to: outside)
        try FileManager.default.createSymbolicLink(at: path, withDestinationURL: outside)
        XCTAssertThrowsError(try WorkflowPreviewToolContract.verifyInputs(files: [entry], directory: root))
    }
    func testPreviewEnvelopeHasNoArbitraryModeCeiling() throws {
        let root = directory, project = root.deletingLastPathComponent().lastPathComponent
        let request = WorkflowToolRequest(operationID: root.lastPathComponent, projectID: project, requestIdentity: fixtureHash,
            toolMode: .modelPreview, arguments: base(root) + ["--preview", "--stop-at", "hidden"] + sampling,
            inputFiles: [file("input/graph.json")], executionBudget: try .resolve(global: .factory, projectMaximumRunSeconds: nil), remainingSeconds: 500)
        XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()), request)
    }
}
