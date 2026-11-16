import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RegistryFieldEditingTests: XCTestCase {
    @MainActor func testBooleanEditRejectsNumericAndStaleIdentityPersistsAndUndoes() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        store.loadExample(); store.selectNode("branch-a")
        XCTAssertFalse(store.setConfigValue("bias", value: .integer(0), nodeID: "branch-a"))
        XCTAssertTrue(store.setConfigValue("bias", value: .bool(false), nodeID: "branch-a"))
        XCTAssertEqual(store.graph.nodes.first { $0.id == "branch-a" }?.config["bias"], .bool(false))
        XCTAssertTrue(store.canvasSummary(try XCTUnwrap(store.selectedNode)).contains("no bias"))
        store.undo(); XCTAssertEqual(store.graph.nodes.first { $0.id == "branch-a" }?.config["bias"], .bool(true))
        store.redo(); XCTAssertEqual(store.graph.nodes.first { $0.id == "branch-a" }?.config["bias"], .bool(false))
        store.selectNode("classifier")
        XCTAssertFalse(store.setConfigValue("bias", value: .bool(true), nodeID: "branch-a"))
        XCTAssertFalse(store.setConfigValue("out_features", value: .bool(true), nodeID: "classifier"))
        let id = try XCTUnwrap(store.activeProjectID)
        let saved = try store.projectLibrary.load(id: id)
        XCTAssertEqual(saved.schemaVersion, 2)
        XCTAssertEqual(saved.nodes.first { $0.id == "branch-a" }?.config["bias"], .bool(false))
        XCTAssertTrue(store.beginProject(saved, id: id))
        XCTAssertEqual(store.graph, saved)
    }
    @MainActor func testBiaslessLinearExportsAndTrainsThroughActualNativeBoundary() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        await verifyRuntimeForIntegration(store)
        store.loadExample(); store.selectNode("branch-a")
        XCTAssertTrue(store.setConfigValue("bias", value: .bool(false), nodeID: "branch-a"))
        store.refreshDraftCode()
        var deadline = Date().addingTimeInterval(20)
        while store.isGeneratingCode && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertNil(store.draftCodeError)
        XCTAssertTrue(store.draftCode.contains("bias=False"), store.draftCodeError ?? store.draftCode)
        store.setExecution { $0.device = "cpu"; $0.epochs = 2 }
        store.run(); deadline = Date().addingTimeInterval(20)
        while store.isRunning && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertTrue(store.lastCompleted, store.runError ?? store.statusMessage)
        let output = try XCTUnwrap(store.outputDirectory)
        let archived = try JSONDecoder().decode(GraphDocument.self, from: Data(contentsOf: output.appendingPathComponent("graph.json")))
        XCTAssertEqual(archived.nodes.first { $0.id == "branch-a" }?.config["bias"], .bool(false))
    }
    @MainActor func testEditableLegacyOpenMigratesWithoutChangingInputDocument() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        var legacy = ExampleFlows.parallelClassifier; legacy.schemaVersion = 1
        for index in legacy.nodes.indices { legacy.nodes[index].config.removeValue(forKey: "bias") }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let oldBytes = try encoder.encode(legacy)
        XCTAssertTrue(store.beginProject(legacy))
        XCTAssertEqual(store.graph.schemaVersion, 2)
        XCTAssertTrue(store.graph.nodes.filter { $0.type == "linear" }.allSatisfy { $0.config["bias"] == .bool(true) })
        XCTAssertEqual(try encoder.encode(legacy), oldBytes)
    }
}
