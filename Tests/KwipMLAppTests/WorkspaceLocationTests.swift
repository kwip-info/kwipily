import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkspaceLocationTests: XCTestCase {
    @MainActor func testUserStorageIsSeparateFromRuntimeCheckout() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let checkout = base.appendingPathComponent("read-only-runtime")
        let storage = base.appendingPathComponent("user data")
        let store = WorkspaceStore(rootOverride: checkout, storageOverride: storage, detectCapabilities: false)
        store.beginProject(.init(name: "Untitled flow"))
        XCTAssertEqual(store.workspaceDirectory, storage)
        XCTAssertTrue(try XCTUnwrap(store.projectURL).path.hasPrefix(storage.path + "/projects/"))
        store.recordExperiment(status: "failed", directory: storage.appendingPathComponent("runs/test"), error: "fixture")
        XCTAssertTrue(try XCTUnwrap(store.experimentIndex).path.hasPrefix(storage.path + "/experiment-index/"))
        XCTAssertEqual(store.experiments.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkout.path))
    }
    @MainActor func testDevelopmentStorageRemainsCompatible() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        XCTAssertEqual(store.workspaceDirectory, root.appendingPathComponent(".local"))
        store.beginProject(.init(name: "Untitled flow"))
        XCTAssertTrue(try XCTUnwrap(store.projectURL).path.hasPrefix(root.appendingPathComponent(".local/projects").path))
    }
    @MainActor func testPresentBrokenBundleNeverUsesCheckoutRuntime() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let resource = root.appendingPathComponent("Resources")
        let payload = resource.appendingPathComponent("KwipMLRuntime")
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        try Data("malformed manifest".utf8).write(to: payload.appendingPathComponent("runtime-manifest.json"))
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root.appendingPathComponent("user"), resourcesOverride: resource, detectCapabilities: false)
        XCTAssertNil(store.runtime)
        XCTAssertTrue(store.runtimeError?.contains("Bundled runtime is invalid") == true)
        store.refreshDraftCode()
        XCTAssertFalse(store.isGeneratingCode)
        XCTAssertTrue(store.draftCodeError?.contains("Bundled runtime is invalid") == true)
    }

}
