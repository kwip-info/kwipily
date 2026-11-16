import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Native store → actual supervised Python worker → artifacts. No mock events.
final class AppRunnerIntegrationTests: XCTestCase {
    private func canonicalDirectory(_ url: URL) throws -> URL {
        let path = try XCTUnwrap(realpath(url.path, nil))
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path), isDirectory: true)
    }
    private func temporaryRuntimeFixture() throws -> (repository: URL, storage: URL) {
        let repository = try canonicalDirectory(URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent())
        guard FileManager.default.isExecutableFile(atPath: repository.appendingPathComponent(".venv/bin/python").path) else {
            throw XCTSkip("Project Python runtime is absent; run scripts/setup.sh to enable native runner integration tests.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("KwipML-run-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        // Inventory the real worker tree; only writable test state belongs in temporary storage.
        return (repository, try canonicalDirectory(root))
    }
    @MainActor private func cleanup(_ root:URL,store:WorkspaceStore) {
        if let id=store.activeProjectID {
            var paths=UserDefaults.standard.dictionary(forKey:"lastRunByProject") as? [String:String] ?? [:]
            paths.removeValue(forKey:id);UserDefaults.standard.set(paths,forKey:"lastRunByProject")
        }
        try? FileManager.default.removeItem(at:root)
    }
    @MainActor private func waitUntil(_ predicate:()->Bool,timeout:TimeInterval=45) async throws {
        let deadline=Date().addingTimeInterval(timeout)
        while !predicate() {
            if Date() >= deadline { throw NSError(domain:"AppRunnerIntegrationTests",code:1,userInfo:[NSLocalizedDescriptionKey:"Timed out waiting for the native worker."]) }
            try await Task.sleep(nanoseconds:25_000_000)
        }
    }
    @MainActor func testNativeStoreTrainsMappedCSVAndPublishesRealArtifacts() async throws {
        let fixture = try temporaryRuntimeFixture()
        let root = fixture.storage
        let store = WorkspaceStore(rootOverride: fixture.repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        defer { cleanup(root,store:store) }
        let csv=root.appendingPathComponent("fixture.csv")
        var lines=["length,width,category"]
        for i in 0..<40 {
            let label=i<20 ? 0 : 1
            let length=(label == 0 ? -1.0 : 1.0)+Double(i%20)*0.02
            let width=Double(i)*0.031+0.1
            lines.append("\(length),\(width),\(label)")
        }
        try lines.joined(separator:"\n").write(to:csv,atomically:true,encoding:.utf8)
        store.graph=ExampleFlows.parallelClassifier
        store.graph.execution=ExecutionSettings(device:"cpu",epochs:5,datasetCSV:csv.path,featureColumns:["length","width"],targetColumn:"category")
        store.run()
        XCTAssertTrue(store.isRunning)
        do { try await waitUntil({!store.isRunning}) }
        catch { store.cancelRun();try? await waitUntil({!store.isRunning},timeout:10);throw error }
        XCTAssertNil(store.runError,store.runError ?? "")
        XCTAssertTrue(store.lastCompleted)
        XCTAssertEqual(store.metrics.count,5)
        XCTAssertEqual(store.metrics.last?.epoch,5)
        XCTAssertFalse(store.generatedCode.isEmpty)
        XCTAssertFalse(store.previews.isEmpty)
        let output=try XCTUnwrap(store.outputDirectory)
        for file in ["graph.json","checkpoint.pt","model.py","metrics.json","manifest.json"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath:output.appendingPathComponent(file).path),file)
        }
        let data=try Data(contentsOf:output.appendingPathComponent("manifest.json"))
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
        let provenance=try XCTUnwrap(manifest["provenance"] as? [String:Any])
        XCTAssertEqual(provenance["dataset"] as? String,"local-csv-v1")
        XCTAssertEqual(provenance["featureColumns"] as? [String],["length","width"])
        XCTAssertEqual(provenance["targetColumn"] as? String,"category")
        XCTAssertEqual(provenance["samples"] as? Int,40)
        XCTAssertEqual(provenance["trainingSamples"] as? Int,32)
        XCTAssertEqual(provenance["validationSamples"] as? Int,8)
        XCTAssertEqual((manifest["runtime"] as? [String:Any])?["device"] as? String,"cpu")
        let saved=try store.projectLibrary.load(id:XCTUnwrap(store.activeProjectID))
        XCTAssertEqual(saved.execution,store.graph.execution)
    }
    @MainActor func testNativeCancellationNeverPublishesCompletedState() async throws {
        let fixture = try temporaryRuntimeFixture()
        let root = fixture.storage
        let store = WorkspaceStore(rootOverride: fixture.repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        defer { cleanup(root,store:store) }
        store.graph=ExampleFlows.parallelClassifier
        store.graph.execution=ExecutionSettings(device:"cpu",epochs:10000)
        store.run()
        do {
            // Observe actual progress before cancellation, not merely process launch.
            try await waitUntil({!store.metrics.isEmpty || !store.isRunning})
            XCTAssertTrue(store.isRunning,"Long run should still be active when the first epoch arrives.")
            XCTAssertFalse(store.metrics.isEmpty)
            store.cancelRun()
            try await waitUntil({!store.isRunning},timeout:15)
        } catch { store.cancelRun();try? await waitUntil({!store.isRunning},timeout:10);throw error }
        XCTAssertTrue(store.wasCancelled)
        XCTAssertNil(store.runError)
        XCTAssertFalse(store.lastCompleted)
        XCTAssertNil(store.outputDirectory)
        XCTAssertNil(store.resultSignature)
        XCTAssertTrue(store.statusMessage.lowercased().contains("cancel"))
        let runs=root.appendingPathComponent("runs")
        if let enumerator=FileManager.default.enumerator(at:runs,includingPropertiesForKeys:nil) {
            let files=enumerator.allObjects.compactMap {$0 as? URL}
            XCTAssertFalse(files.contains {$0.lastPathComponent == "manifest.json"},"Cancelled worker must not publish a success manifest.")
        }
    }
    @MainActor func testIterationFlagsReachRealManifestAndPersistedHistory() async throws {
        let fixture = try temporaryRuntimeFixture()
        let root = fixture.storage
        let store = WorkspaceStore(rootOverride: fixture.repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        defer { cleanup(root, store: store) }
        store.graph = ExampleFlows.parallelClassifier
        let settings = ExecutionSettings(device: "cpu", epochs: 3, batchSize: 7, sampleLimit: 40, seed: 9, splitSeed: 17, validationPercent: 25, learningRate: 0.012)
        store.graph.execution = settings
        store.run()
        XCTAssertTrue(store.isRunning)
        do { try await waitUntil({ !store.isRunning }) }
        catch { store.cancelRun(); try? await waitUntil({ !store.isRunning }, timeout: 10); throw error }
        XCTAssertNil(store.runError, store.runError ?? "")
        XCTAssertTrue(store.lastCompleted)
        let output = try XCTUnwrap(store.outputDirectory)
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("manifest.json"))) as? [String: Any])
        let actual = try XCTUnwrap(manifest["settings"] as? [String: Any])
        let provenance = try XCTUnwrap(manifest["provenance"] as? [String: Any])
        XCTAssertEqual(manifest["recipeVersion"] as? Int, 2)
        XCTAssertEqual(manifest["seed"] as? Int, 9)
        XCTAssertEqual(actual["epochs"] as? Int, 3)
        XCTAssertEqual(actual["requestedBatchSize"] as? Int, 7)
        XCTAssertEqual(actual["batchSize"] as? Int, 7)
        XCTAssertEqual(actual["lastBatchSize"] as? Int, 2)
        XCTAssertEqual(actual["batchesPerEpoch"] as? Int, 5)
        XCTAssertEqual(actual["sampleLimit"] as? Int, 40)
        XCTAssertEqual(actual["splitSeed"] as? Int, 17)
        XCTAssertEqual(actual["validationPercent"] as? Int, 25)
        XCTAssertEqual(try XCTUnwrap(actual["learningRate"] as? Double), 0.012, accuracy: 0.0000001)
        XCTAssertEqual(provenance["trainingSamples"] as? Int, 30)
        XCTAssertEqual(provenance["validationSamples"] as? Int, 10)
        let trainIDs = try XCTUnwrap(provenance["trainingRowIDs"] as? [Int])
        let validationIDs = try XCTUnwrap(provenance["validationRowIDs"] as? [Int])
        XCTAssertTrue(Set(trainIDs).isDisjoint(with: validationIDs))
        let recorded = try XCTUnwrap(store.experiments.first)
        XCTAssertEqual(recorded.status, "completed")
        XCTAssertEqual(recorded.settings, settings)
        XCTAssertEqual(recorded.trainingSamples, 30)
        XCTAssertEqual(recorded.validationSamples, 10)
        XCTAssertEqual(recorded.splitHash, provenance["splitHash"] as? String)
        XCTAssertEqual(recorded.sourceHash, provenance["sourceHash"] as? String)
        XCTAssertEqual(recorded.epoch, 3)
        XCTAssertEqual(recorded.directory, output.path)
        XCTAssertGreaterThanOrEqual(recorded.finishedAt, recorded.startedAt)
        store.refreshExperiments()
        XCTAssertEqual(store.experiments.first?.id, recorded.id, "History must survive a disk reload.")
        XCTAssertEqual(store.experiments.first?.settings, settings)
        let saved = try store.projectLibrary.load(id: XCTUnwrap(store.activeProjectID))
        XCTAssertEqual(saved.execution, settings)
    }

    @MainActor func testRestoredLegacyRecipeIsHistoricalDespiteMatchingGraph() async throws {
        let fixture = try temporaryRuntimeFixture()
        let root = fixture.storage
        let store = WorkspaceStore(rootOverride: fixture.repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        defer { cleanup(root, store: store) }
        store.graph = ExampleFlows.parallelClassifier
        store.graph.execution = ExecutionSettings(device: "cpu", epochs: 1)
        store.run()
        do { try await waitUntil({ !store.isRunning }) }
        catch { store.cancelRun(); try? await waitUntil({ !store.isRunning }, timeout: 10); throw error }
        XCTAssertTrue(store.lastCompleted)
        XCTAssertFalse(store.resultsStale)
        let manifestURL = try XCTUnwrap(store.outputDirectory).appendingPathComponent("manifest.json")
        var manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any])
        manifest.removeValue(forKey: "recipeVersion")
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL, options: .atomic)
        store.restoreLastRun()
        XCTAssertTrue(store.lastCompleted, "Historical artifacts remain accessible.")
        XCTAssertTrue(store.resultsStale, "A matching graph does not make an older data-selection recipe current.")
        XCTAssertFalse(store.metrics.isEmpty)
    }

}
