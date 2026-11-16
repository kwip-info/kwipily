import XCTest
import CryptoKit
@testable import KwipMLApp
import KwipMLCore

final class PreparedPreviewTests: XCTestCase {
    @MainActor func testRealFrozenValidationPreviewResumeAndArtifactInvalidation() async throws {
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
        workspace.selectNode("merge");workspace.previewSelected()
        try await previewFinished(workspace)
        XCTAssertNotNil(workspace.previewCachePath,workspace.statusMessage)
        XCTAssertTrue(workspace.cacheValid,workspace.statusMessage)
        XCTAssertTrue(workspace.statusMessage.contains("validation"))
        XCTAssertTrue(workspace.statusMessage.contains("8"))
        workspace.selectNode("merge");workspace.resumeSelected()
        try await previewFinished(workspace)
        XCTAssertFalse(workspace.previewReused.isEmpty,workspace.statusMessage)
        XCTAssertNotNil(workspace.previews["output"])
        try "changed external source".write(to:csv,atomically:true,encoding:.utf8)
        XCTAssertTrue(workspace.cacheValid,"Frozen prepared data must not depend on later source edits")
        workspace.setExecution{$0.epochs=2};workspace.run()
        let trainDeadline=Date().addingTimeInterval(30)
        while workspace.isRunning && Date()<trainDeadline{try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertTrue(workspace.lastCompleted,workspace.runError ?? workspace.statusMessage)
        workspace.selectNode("output");workspace.previewSelected();try await previewFinished(workspace)
        XCTAssertEqual(workspace.previewWeightLabel,"Inference · trained checkpoint")
        workspace.selectNode("merge");workspace.resumeSelected();try await previewFinished(workspace)
        XCTAssertFalse(workspace.previewReused.isEmpty,workspace.statusMessage)
        XCTAssertEqual(workspace.previewWeightLabel,"Inference · trained checkpoint")
        let captured=try XCTUnwrap(workspace.previewCheckpoint)
        XCTAssertNotEqual(captured,workspace.outputDirectory?.appendingPathComponent("checkpoint.pt").path)
        try Data("changed weights".utf8).write(to:URL(fileURLWithPath:captured))
        XCTAssertFalse(workspace.cacheValid)
        workspace.previewSelected();try await previewFinished(workspace)
        XCTAssertTrue(workspace.cacheValid)
        let entry=try XCTUnwrap((manifest["artifacts"] as? [[String:Any]])?.first{$0["nodeID"] as? String == "validation"})
        let artifact=url.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(entry["path"] as? String))
        try Data("{}".utf8).write(to:artifact)
        XCTAssertFalse(workspace.cacheValid)
        workspace.resumeSelected();XCTAssertFalse(workspace.isPreviewing)
    }
    @MainActor private func previewFinished(_ store:WorkspaceStore) async throws {
        let deadline=Date().addingTimeInterval(30)
        while store.isPreviewing && Date()<deadline{try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(store.isPreviewing)
    }
}
