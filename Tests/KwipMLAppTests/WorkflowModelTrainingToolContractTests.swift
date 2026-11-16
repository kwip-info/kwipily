import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowModelTrainingToolContractTests: XCTestCase {
    private let digest = String(repeating: "a", count: 64)
    private var directory: URL { URL(fileURLWithPath: "/private/tmp/workspace/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)") }
    private func file(_ path: String, _ size: Int = 2) -> WorkflowArchiveFile { .init(path: path, bytes: size, sha256: digest) }
    private func base(_ root: URL) -> [String] {
        ["--graph", root.appendingPathComponent("input/graph.json").path, "--output", root.appendingPathComponent("artifacts").path,
         "--epochs", "10", "--seed", "42", "--device", "cpu", "--batch-size", "64", "--learning-rate", "0.03"]
    }
    private let sampling = ["--sample-limit", "0", "--split-seed", "42", "--validation-percent", "20"]
    private func prepared(_ root: URL) -> [String] {
        ["--prepared-dataflow", root.appendingPathComponent("input/preparation/manifest.json").path,
         "--prepared-manifest-sha256", digest, "--prepared-graph-digest", digest,
         "--prepared-training-node", "split", "--prepared-training-port", "training",
         "--prepared-validation-node", "split", "--prepared-validation-port", "validation",
         "--feature-columns-json", "[\"x\",\"y\"]", "--target-column", "label"]
    }
    private func replacing(_ arguments: [String], _ key: String, _ value: String) -> [String] {
        var result = arguments; result[result.firstIndex(of: key)! + 1] = value; return result
    }
    func testFreshBundledAndCSVRequireExactOwnedInputsAndRejectMixedModes() throws {
        let root = directory, bundled = base(root) + sampling, graph = file("input/graph.json")
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: bundled, files: [graph], directory: root))
        let csv = ["--dataset-csv", root.appendingPathComponent("input/data/my sample.csv").path,
                   "--feature-columns-json", "[\"x\",\"y\"]", "--target-column", "label"]
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: bundled + csv,
            files: [graph, file("input/data/my sample.csv", 10_000_000)], directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: bundled + csv, files: [graph], directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: bundled, files: [graph, file("input/data/unused.csv")], directory: root))
        for extra in [["--preview"], ["--export-code"], ["--registry", "/tmp/registry.json"], ["--cache", "/tmp/cache.pt"],
                      ["--checkpoint", "/tmp/model.pt"], ["--epochs", "2"], ["--execute-dataflow", "/tmp/flow.json"]] {
            XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: bundled + extra, files: [graph], directory: root))
        }
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: replacing(bundled, "--output", "/tmp/out"), files: [graph], directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: replacing(bundled, "--graph", root.appendingPathComponent("input/../graph.json").path), files: [graph], directory: root))
    }
    func testScalarAndGradientBoundsPreserveSupportedTrainingControls() throws {
        let root = directory, args = base(root) + sampling, files = [file("input/graph.json")]
        for (key, values) in ["--epochs": ["0", "10001", "01"], "--batch-size": ["0", "100001"],
                              "--seed": ["-1", "4294967296"], "--split-seed": ["-1"], "--sample-limit": ["19", "100001"],
                              "--validation-percent": ["4", "51"], "--learning-rate": ["0", "-0.1", "1.01", "nan", "inf"]] {
            for value in values { XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: replacing(args, key, value), files: files, directory: root), "\(key)=\(value)") }
        }
        for value in ["1", "1e-6", "0.0001"] {
            XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: replacing(args, "--learning-rate", value), files: files, directory: root))
        }
        let gradient = ["--capture-gradients", "--gradient-every-epochs", "10000", "--gradient-max-batches-per-epoch", "8", "--gradient-max-captures", "1024"]
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: args + gradient, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + Array(gradient.dropFirst()), files: files, directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + ["--capture-gradients"], files: files, directory: root))
        for (key, value) in [("--gradient-every-epochs", "0"), ("--gradient-max-batches-per-epoch", "9"), ("--gradient-max-captures", "1025")] {
            XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: replacing(args + gradient, key, value), files: files, directory: root))
        }
    }
    func testContinuationAndCompositeRequireCompletePinnedOwnedBundles() throws {
        let root = directory, args = base(root) + sampling
        let parent = [file("input/continuation-parent/manifest.json", 20_000_000), file("input/continuation-parent/checkpoint.pt", 100_000_000), file("input/continuation-parent/metrics.json", 4_000_000)]
        let composite = ["--composite-source", root.appendingPathComponent("input/composite/source.json").path, "--composite-source-sha256", digest]
        let files = [file("input/graph.json"), file("input/composite/source.json", 2_000_000)] + parent
        for mode in ["--resume-checkpoint", "--finetune-checkpoint"] {
            let continuation = [mode, root.appendingPathComponent("input/continuation-parent/checkpoint.pt").path]
            XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: args + continuation + composite, files: files, directory: root))
            for missing in parent {
                XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + continuation + composite, files: files.filter { $0.path != missing.path }, directory: root))
            }
            XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + [mode, "/tmp/checkpoint.pt"] + composite, files: files, directory: root))
        }
        let both = ["--resume-checkpoint", root.appendingPathComponent("input/continuation-parent/checkpoint.pt").path,
                    "--finetune-checkpoint", root.appendingPathComponent("input/continuation-parent/checkpoint.pt").path]
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + both + composite, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + composite,
            files: [file("input/graph.json"), .init(path: "input/composite/source.json", bytes: 2, sha256: String(repeating: "b", count: 64))], directory: root))
    }
    func testPreparedTrainingNeverResplitsAndAcceptsCompleteMaximumStructuralEnvelope() throws {
        let root = directory
        let continuation = ["--resume-checkpoint", root.appendingPathComponent("input/continuation-parent/checkpoint.pt").path]
        let composite = ["--composite-source", root.appendingPathComponent("input/composite/source.json").path, "--composite-source-sha256", digest]
        let files = [file("input/graph.json", 2_000_000), file("input/preparation/manifest.json", 2_000_000),
                     file("input/continuation-parent/manifest.json", 20_000_000), file("input/continuation-parent/checkpoint.pt", 100_000_000),
                     file("input/continuation-parent/metrics.json", 4_000_000), file("input/composite/source.json", 2_000_000)]
            + (0..<300).map { file("input/preparation/table\($0).json", $0 < 4 ? 50_000_000 : 0) }
        let args = base(root) + prepared(root) + continuation + composite
        XCTAssertEqual(files.count, 306); XCTAssertEqual(files.reduce(0) { $0 + $1.bytes }, 330_000_000)
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: args, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + sampling, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args, files: files + [file("input/preparation/extra.json", 0)], directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.fileLimit(path: "input/preparation/../escape"))
        XCTAssertEqual(try WorkflowModelTrainingToolContract.fileLimit(path: "input/continuation-parent/metrics.json"), 4_000_000)
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.fileLimit(path: "input/weights/checkpoint.pt"))
    }
    func testPinnedInputVerificationRejectsChangedFilesAndIncompletePreparedInventory() throws {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil)); defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString).appendingPathComponent("workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("input/preparation"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()) }
        let bytes = Data("{}".utf8)
        let manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "graphDigest": digest,
            "artifacts": [["path": "table.json", "bytes": bytes.count, "sha256": WorkflowArchiveCodec.hash(bytes)]]]
        let raw = try JSONSerialization.data(withJSONObject: manifest), pin = WorkflowArchiveCodec.hash(raw)
        let values = ["input/graph.json": bytes, "input/preparation/manifest.json": raw, "input/preparation/table.json": bytes]
        for (path, data) in values { try data.write(to: root.appendingPathComponent(path)) }
        let files = values.map { WorkflowArchiveFile(path: $0.key, bytes: $0.value.count, sha256: WorkflowArchiveCodec.hash($0.value)) }
        let args = replacing(base(root) + prepared(root), "--prepared-manifest-sha256", pin)
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.verifyInputs(arguments: args, files: files, directory: root))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.verifyInputs(arguments: args, files: files.filter { $0.path != "input/preparation/table.json" }, directory: root))
        try Data("xx".utf8).write(to: root.appendingPathComponent("input/preparation/table.json"))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.verifyInputs(arguments: args, files: files, directory: root))
    }
}
