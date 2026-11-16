import XCTest
import CryptoKit
@testable import KwipMLApp
import KwipMLCore
final class CheckpointActionsTests:XCTestCase {
    @MainActor func testLegacyParentMatchesMigratedBiasTrueButRejectsFalseWithoutMutation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        var legacy = ExampleFlows.parallelClassifier; legacy.schemaVersion = 1
        legacy.execution = ExecutionSettings(); legacy.execution?.device = "cpu"; legacy.execution?.epochs = 2
        for index in legacy.nodes.indices { legacy.nodes[index].config.removeValue(forKey: "bias") }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let original = try encoder.encode(legacy)
        let parent = CheckpointParent(directory: root, runID: "legacy", completedEpoch: 2, manifestSHA256: "pinned", checkpointSHA256: "pinned", metricsSHA256: "pinned", graph: legacy, device: "cpu", lineageMode: "newTraining", parentRunID: nil)
        XCTAssertTrue(store.beginProject(legacy))
        store.setExecution { $0.epochs = 3 }
        XCTAssertNil(store.checkpointCompatibility(CheckpointContinuation(parent: parent, mode: .exactResume)))
        XCTAssertNil(store.checkpointCompatibility(CheckpointContinuation(parent: parent, mode: .fineTune)))
        store.selectNode("branch-a"); XCTAssertTrue(store.setConfigValue("bias", value: .bool(false), nodeID: "branch-a"))
        XCTAssertNotNil(store.checkpointCompatibility(CheckpointContinuation(parent: parent, mode: .exactResume)))
        XCTAssertNotNil(store.checkpointCompatibility(CheckpointContinuation(parent: parent, mode: .fineTune)))
        XCTAssertEqual(try encoder.encode(parent.graph), original)
    }
    @MainActor func testNativeResumeAndFineTunePreserveParentAndRejectIncompatibleRequest()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation {c in store.refreshCapabilities{c.resume(returning:$0)}};XCTAssertTrue(ready)
        store.loadExample();store.setExecution{$0.device="cpu";$0.epochs=2;$0.sampleLimit=40}
        store.run();try await finished(store)
        let directory=try XCTUnwrap(store.outputDirectory)
        let parent=try store.checkpointParent(at:directory)
        let original=try Data(contentsOf:directory.appendingPathComponent("checkpoint.pt"))
        XCTAssertEqual(parent.completedEpoch,2)
        XCTAssertThrowsError(try store.checkpointArguments(CheckpointContinuation(parent:parent,mode:.exactResume),requestDirectory:storage.appendingPathComponent("reject")))
        store.run(continuation:CheckpointContinuation(parent:parent,mode:.exactResume))
        XCTAssertFalse(store.isRunning);XCTAssertNotNil(store.runError)
        store.setExecution{$0.device="auto";$0.epochs=4}
        XCTAssertNotNil(store.checkpointCompatibility(CheckpointContinuation(parent:parent,mode:.exactResume)))
        store.setExecution{$0.device="cpu";$0.seed=123}
        XCTAssertNotNil(store.checkpointCompatibility(CheckpointContinuation(parent:parent,mode:.exactResume)))
        store.setExecution{$0.seed=42;$0.epochs=4}
        store.run(continuation:CheckpointContinuation(parent:parent,mode:.exactResume));try await finished(store)
        let resumed=try XCTUnwrap(store.outputDirectory);XCTAssertNotEqual(resumed,directory)
        XCTAssertEqual(store.metrics.map(\.epoch),[1,2,3,4])
        var manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:resumed.appendingPathComponent("manifest.json"))) as? [String:Any])
        XCTAssertEqual((manifest["lineage"] as? [String:Any])?["mode"] as? String,"exactResume")
        XCTAssertEqual((manifest["settings"] as? [String:Any])?["initialEpoch"] as? Int,2)
        store.setExecution{$0.epochs=2;$0.learningRate=0.01;$0.seed=71}
        store.run(continuation:CheckpointContinuation(parent:parent,mode:.fineTune));try await finished(store)
        manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:try XCTUnwrap(store.outputDirectory).appendingPathComponent("manifest.json"))) as? [String:Any])
        XCTAssertEqual((manifest["lineage"] as? [String:Any])?["mode"] as? String,"weightsOnlyFineTune")
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("checkpoint.pt")),original)
        try Data("changed".utf8).write(to:directory.appendingPathComponent("checkpoint.pt"))
        XCTAssertThrowsError(try store.checkpointArguments(CheckpointContinuation(parent:parent,mode:.fineTune),requestDirectory:storage.appendingPathComponent("reject-corrupt")))
    }
    @MainActor func testPreparedSelectionSurvivesExactNativeResume()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:storage)}
        try FileManager.default.createDirectory(at:storage,withIntermediateDirectories:true)
        let store=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation{c in store.refreshCapabilities{c.resume(returning:$0)}};XCTAssertTrue(ready)
        store.loadExample();store.setExecution{$0.device="cpu";$0.epochs=1}
        let prep=DataFlowWorkspaceStore(workspace:store);prep.loadExample(columns:["x","y"]);prep.selectNode("cast")
        prep.configure(["rows":.array([("x","float64"),("y","float64"),("label","int64")].map {name,type in
            .object(["source":.string(name),"included":true,"outputName":.string(name),"targetType":.string(type)])
        }),"castFailure":.string("error"),"trimText":false,"newColumns":.string("keep"),"missingPolicy":.string("error")])
        prep.selectNode("split");prep.configure(["strategy":.string("random"),"seed":.integer(42),"validationPercent":.integer(20),"testPercent":.integer(20),"stratifyColumn":.string("label")])
        let source=storage.appendingPathComponent("source.csv")
        try (["x,y,label"]+(0..<40).map{"\($0+1),\($0*$0+1),\($0%2)"}).joined(separator:"\n").write(to:source,atomically:true,encoding:.utf8)
        prep.project.sourceBindings["source"]=source.path;prep.runPreparation()
        let deadline=Date().addingTimeInterval(30);while prep.isBusy && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        let manifestURL=try XCTUnwrap(prep.manifestURL,prep.statusMessage),bytes=try Data(contentsOf:manifestURL)
        let preparation=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        let selected=PreparedTrainingSelection(manifestPath:manifestURL.path,manifestSHA256:SHA256.hash(data:bytes).map{String(format:"%02x",$0)}.joined(),graphDigest:try XCTUnwrap(preparation["graphDigest"] as? String),trainingNode:"train",trainingPort:"table",validationNode:"validation",validationPort:"table",featureColumns:["x","y"],targetColumn:"label")
        try store.usePreparedTraining(selected);store.run();try await finished(store)
        let parent=try store.checkpointParent(at:XCTUnwrap(store.outputDirectory))
        store.setExecution{$0.epochs=2};store.run(continuation:CheckpointContinuation(parent:parent,mode:.exactResume));try await finished(store)
        XCTAssertEqual(store.execution.preparedTraining,selected)
        let output=try XCTUnwrap(store.outputDirectory)
        XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("preparation/manifest.json")),bytes)
        let result=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("manifest.json"))) as? [String:Any])
        XCTAssertEqual((result["settings"] as? [String:Any])?["partitionMode"] as? String,"prepared")
        XCTAssertEqual((result["lineage"] as? [String:Any])?["mode"] as? String,"exactResume")
    }
    @MainActor private func finished(_ store:WorkspaceStore)async throws {
        let deadline=Date().addingTimeInterval(30)
        while store.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(store.isRunning);XCTAssertTrue(store.lastCompleted,store.runError ?? store.statusMessage)
    }
}
