import XCTest
import Foundation
import KwipMLCore
@testable import KwipMLApp

final class CodeGenerationTests:XCTestCase {
    @MainActor func testInvalidGraphOffersActionableErrorWithoutLaunchingRuntime() {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        store.graph=GraphDocument()
        store.refreshDraftCode()
        XCTAssertFalse(store.isGeneratingCode);XCTAssertTrue(store.draftCode.isEmpty)
        XCTAssertTrue(store.draftCodeError?.contains("connections") == true)
    }
    @MainActor private func controlledStore() throws -> (URL, WorkspaceStore, CodeWorkerProbe) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let bin = root.appendingPathComponent(".venv/bin"), runner = root.appendingPathComponent("runner")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: runner, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python"), withDestinationURL: URL(fileURLWithPath: "/bin/sh"))
        try Data("fixture".utf8).write(to: runner.appendingPathComponent("main.py"))
        let scripts = root.appendingPathComponent("scripts")
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        try Data("fixture launcher".utf8).write(to: scripts.appendingPathComponent("runtime_launch.py"))
        let resources = root.appendingPathComponent("Sources/KwipMLCore/Resources")
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: resources.appendingPathComponent("components.json"))
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        try installFixtureExecutionContract(store, version: "0.0.0+fixture")
        store.installProjectSession(store.graph, id: nil, owned: false)
        let probe = CodeWorkerProbe()
        store.codeWorkerLauncher = probe.launch
        return (root, store, probe)
    }
    @MainActor func testChangedGraphCannotPublishEarlierGeneration() async throws {
        let (root, store, probe) = try controlledStore()
        defer { try? FileManager.default.removeItem(at: root) }
        store.refreshDraftCode(); XCTAssertTrue(store.isGeneratingCode, store.draftCodeError ?? "No diagnostic")
        try await probe.waitForLaunches(1)
        guard store.isGeneratingCode else { return }
        store.graph.nodes[1].config["out_features"] = 9
        probe.emit(0, code: "old source"); probe.finish(0)
        XCTAssertFalse(store.isGeneratingCode); XCTAssertTrue(store.draftCode.isEmpty); XCTAssertNil(store.codeSignature)
    }
    @MainActor func testRenamedNodeCannotPublishOrReuseEarlierSource() async throws {
        let (root, store, probe) = try controlledStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let signature = store.semanticSignature(store.graph)
        store.refreshDraftCode()
        try await probe.waitForLaunches(1)
        store.graph.nodes[1].name = "Renamed layer"
        XCTAssertEqual(store.semanticSignature(store.graph), signature)
        probe.emit(0, code: "old aliases"); probe.finish(0)
        XCTAssertTrue(store.draftCode.isEmpty)
        store.refreshDraftCode(); XCTAssertTrue(store.isGeneratingCode, store.draftCodeError ?? "No diagnostic")
        try await probe.waitForLaunches(2)
        guard store.isGeneratingCode else { return }
        probe.emit(1, code: "new aliases"); probe.finish(1)
        XCTAssertEqual(store.draftCode, "new aliases")
    }
    @MainActor func testSupersededWorkerCannotClearNewHandleAndTimeoutCannotPublish() async throws {
        let (root, store, probe) = try controlledStore()
        defer { try? FileManager.default.removeItem(at: root) }
        store.refreshDraftCode()
        try await probe.waitForLaunches(1)
        store.graph.nodes[1].name = "New name"
        store.refreshDraftCode()
        try await probe.waitForLaunches(2)
        XCTAssertEqual(probe.cancellations, [0])
        probe.emit(0, code: "stale"); probe.finish(0)
        XCTAssertTrue(store.isGeneratingCode); XCTAssertNotNil(store.codeGenerationWorker)
        probe.emit(1, code: "provisional"); probe.finish(1, status: 124)
        XCTAssertFalse(store.isGeneratingCode); XCTAssertNil(store.codeGenerationWorker)
        XCTAssertTrue(store.draftCode.isEmpty)
        XCTAssertTrue(store.draftCodeError?.contains("time") == true)
    }
    @MainActor func testProjectSwitchCancelsAndRetainsOwnedInputsUntilActualExit() async throws {
        let (root, store, probe) = try controlledStore()
        defer { try? FileManager.default.removeItem(at: root) }
        store.refreshDraftCode()
        try await probe.waitForLaunches(1)
        let input = probe.folders[0].appendingPathComponent("input/graph.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: input.path))
        store.installProjectSession(store.graph, id: nil, owned: false)
        XCTAssertEqual(probe.cancellations, [0])
        XCTAssertTrue(FileManager.default.fileExists(atPath: input.path))
        probe.emit(0, code: "stale"); probe.finish(0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input.path))
        XCTAssertTrue(store.draftCode.isEmpty); XCTAssertFalse(store.isGeneratingCode)
    }
    @MainActor func testInvalidReplacementCancelsExistingWorker() async throws {
        let (root, store, probe) = try controlledStore()
        defer { try? FileManager.default.removeItem(at: root) }
        store.refreshDraftCode()
        try await probe.waitForLaunches(1)
        store.graph = GraphDocument(); store.refreshDraftCode()
        XCTAssertEqual(probe.cancellations, [0])
        probe.finish(0)
        XCTAssertTrue(store.draftCodeError?.contains("connections") == true)
    }
    @MainActor func testSynchronousExitDoesNotResurrectWorkerHandle() async throws {
        let (root, store, probe) = try controlledStore()
        defer { try? FileManager.default.removeItem(at: root) }
        probe.synchronous = true
        store.refreshDraftCode()
        try await probe.waitForLaunches(1)
        XCTAssertEqual(store.draftCode, "synchronous code")
        XCTAssertFalse(store.isGeneratingCode); XCTAssertNil(store.codeGenerationWorker)
    }
    @MainActor func testRealSourceGenerationBeforeTrainingAndWithoutDataset() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Python runtime absent")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let store=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(store)
        store.graph.execution=ExecutionSettings(datasetCSV:"/nonexistent/should-not-be-read.csv")
        store.refreshDraftCode()
        let deadline=Date().addingTimeInterval(20)
        while store.isGeneratingCode && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(store.isGeneratingCode);XCTAssertNil(store.draftCodeError,store.draftCodeError ?? "")
        XCTAssertTrue(store.draftCode.contains("import torch"));XCTAssertTrue(store.draftCode.contains("def forward"))
        XCTAssertEqual(store.codeSignature,store.semanticSignature(store.graph))
        XCTAssertFalse(store.lastCompleted);XCTAssertTrue(store.metrics.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent(".local/runs").path))
    }
}

@MainActor private final class CodeWorkerProbe {
    var lines: [(String) -> Void] = []
    var exits: [(Int32, String) -> Void] = []
    var cancellations: [Int] = []
    var folders: [URL] = []
    var synchronous = false
    func launch(_ executable: URL, _ arguments: [String], _ directory: URL,
                _ line: @escaping (String) -> Void, _ exit: @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle {
        let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
        let folder = URL(fileURLWithPath: arguments[index + 1])
        let request = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("request.json"))) as? [String: Any])
        XCTAssertEqual(request["toolMode"] as? String, "codeExport")
        let item = lines.count; lines.append(line); exits.append(exit); folders.append(folder)
        let ready: [String: Any] = ["event": "operationReady", "operationID": try XCTUnwrap(request["operationID"]),
            "toolMode": "codeExport", "requestSHA256": arguments[index + 2], "ownershipSHA256": arguments[index + 3]]
        line(String(decoding: try JSONSerialization.data(withJSONObject: ready), as: UTF8.self))
        if synchronous { emit(item, code: "synchronous code"); finish(item) }
        return .init(cancel: { self.cancellations.append(item) })
    }
    func waitForLaunches(_ count: Int) async throws {
        let deadline = Date().addingTimeInterval(10)
        while lines.count < count && Date() < deadline { try await Task.sleep(nanoseconds: 10_000_000) }
        guard lines.count >= count else { throw WorkflowArchiveError("Expected code worker did not launch.") }
    }
    func emit(_ item: Int, code: String) {
        guard lines.indices.contains(item) else { XCTFail("Missing launched code worker"); return }
        lines[item](String(decoding: try! JSONSerialization.data(withJSONObject: ["event": "codeExport", "code": code]), as: UTF8.self))
    }
    func finish(_ item: Int, status: Int32 = 0) { guard exits.indices.contains(item) else { XCTFail("Missing launched code worker"); return }; exits[item](status, "") }
}
