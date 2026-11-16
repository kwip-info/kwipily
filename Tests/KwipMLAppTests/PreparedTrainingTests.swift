import XCTest
import CryptoKit
@testable import KwipMLApp
import KwipMLCore

final class PreparedTrainingTests: XCTestCase {
    @MainActor func testRealPreparationToModelRunUsesExactPartitions() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        let ready = await withCheckedContinuation { continuation in workspace.refreshCapabilities { continuation.resume(returning: $0) } }
        XCTAssertTrue(ready, workspace.capabilityStatus)
        workspace.loadExample()
        workspace.setExecution { $0.device = "cpu"; $0.epochs = 2; $0.batchSize = 8 }
        let preparation = DataFlowWorkspaceStore(workspace: workspace)
        preparation.loadExample(columns: ["length", "width"])
        preparation.selectNode("cast")
        preparation.configure(["rows":.array([("length","float64"),("width","float64"),("label","int64")].map {name,type in
            .object(["source":.string(name),"included":true,"outputName":.string(name),"targetType":.string(type)])
        }),"castFailure":.string("error"),"trimText":false,"newColumns":.string("keep"),"missingPolicy":.string("error")])
        preparation.selectNode("split")
        preparation.configure(["strategy":.string("random"),"seed":.integer(17),"validationPercent":.integer(20),"testPercent":.integer(20),"stratifyColumn":.string("label")])
        let csv = directory.appendingPathComponent("source.csv")
        try (["length,width,label"] + (0..<40).map { "\($0),\($0 * 2 + 1),\($0 % 2)" }).joined(separator: "\n").write(to: csv, atomically: true, encoding: .utf8)
        preparation.project.sourceBindings["source"] = csv.path
        preparation.runPreparation()
        let deadline = Date().addingTimeInterval(30)
        while preparation.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        let url = try XCTUnwrap(preparation.manifestURL, preparation.statusMessage)
        // Forty independent rows split explicitly 60/20/20; validation remains eight.
        for (node,expected) in [("train",24),("validation",8),("test",8)] {
            let table=try XCTUnwrap(preparation.previews[node]?["table"] as? [String:Any])
            let preview=try XCTUnwrap(table["preview"] as? [String:Any])
            XCTAssertEqual(preview["rowCount"] as? Int,expected,node)
        }
        let data = try Data(contentsOf: url)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let selection = PreparedTrainingSelection(manifestPath: url.path, manifestSHA256: SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined(), graphDigest: try XCTUnwrap(manifest["graphDigest"] as? String), trainingNode:"train", trainingPort:"table", validationNode:"validation", validationPort:"table", featureColumns:["length","width"], targetColumn:"label")
        try workspace.usePreparedTraining(selection)
        workspace.run()
        while workspace.isRunning && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertTrue(workspace.lastCompleted, workspace.runError ?? workspace.statusMessage)
        XCTAssertEqual(workspace.metrics.last?.epoch, 2)
        let output = try XCTUnwrap(workspace.outputDirectory)
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("preparation/manifest.json").path))
        let saved = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("manifest.json"))) as? [String: Any])
        XCTAssertEqual((saved["settings"] as? [String: Any])?["partitionMode"] as? String, "prepared")
        // A completed training manifest alone cannot authorize a damaged portable data snapshot.
        workspace.trainingWorkerLauncher = { executable, arguments, cwd, line, exit in
            let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
            let operation = URL(fileURLWithPath: arguments[index + 1])
            return try WorkflowExecutionSession.launchWorker(executable, arguments, cwd, line, { code, message in
                do { try Data("{}".utf8).write(to: operation.appendingPathComponent("artifacts/preparation/manifest.json")) }
                catch { XCTFail("Could not inject prepared-artifact damage: \(error)") }
                exit(code, message)
            })
        }
        workspace.run()
        let rejectedDeadline = Date().addingTimeInterval(30)
        while workspace.isRunning && Date() < rejectedDeadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(workspace.isRunning)
        XCTAssertFalse(workspace.lastCompleted)
        XCTAssertNil(workspace.outputDirectory)
        XCTAssertNotNil(workspace.runError)
        XCTAssertEqual(workspace.experiments.first?.status, "failed")
    }
    @MainActor func testPreparedArgumentsPreservePartitionAndRejectMutatedManifest() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifest = directory.appendingPathComponent("manifest.json")
        let data = Data("{\"schemaVersion\":1,\"kind\":\"dataflowExecution\",\"status\":\"completed\",\"graphDigest\":\"\(String(repeating:"a",count:64))\"}".utf8)
        try data.write(to: manifest)
        let selection = PreparedTrainingSelection(manifestPath: manifest.path, manifestSHA256: SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined(), graphDigest:String(repeating:"a",count:64), trainingNode:"train", trainingPort:"table", validationNode:"validation", validationPort:"table", featureColumns:["length","width"], targetColumn:"label")
        let store = WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        try store.usePreparedTraining(selection)
        let args = try store.trainingDataArguments()
        XCTAssertTrue(args.contains("--prepared-dataflow"))
        XCTAssertFalse(args.contains("--split-seed"))
        XCTAssertFalse(args.contains("--sample-limit"))
        XCTAssertEqual(store.execution.preparedTraining, selection)
        try Data("{}".utf8).write(to:manifest)
        XCTAssertThrowsError(try store.trainingDataArguments())
    }
    @MainActor func testDirectDataRetainsExplicitSplitBudget() throws {
        let store = WorkspaceStore(rootOverride:FileManager.default.temporaryDirectory,detectCapabilities:false)
        let args = try store.trainingDataArguments()
        XCTAssertTrue(args.contains("--split-seed"))
        XCTAssertTrue(args.contains("--sample-limit"))
        XCTAssertFalse(args.contains("--prepared-dataflow"))
    }
}
