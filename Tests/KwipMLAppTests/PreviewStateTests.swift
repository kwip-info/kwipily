import XCTest
import KwipMLCore
@testable import KwipMLApp
final class PreviewStateTests: XCTestCase {
    func testCacheRequiresArtifactAndMatchingArchitecture() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            XCTAssertFalse(store.cacheValid)
            let cache = root.appendingPathComponent("activation-cache.pt")
            try Data("fixture".utf8).write(to: cache)
            let sidecar = cache.deletingPathExtension().appendingPathExtension("json")
            try Data("fixture metadata".utf8).write(to: sidecar)
            store.previewCachePath = cache.path
            store.capturePreviewArtifactHashes()
            store.previewSignature = store.semanticSignature(store.graph)
            XCTAssertTrue(store.cacheValid)
            store.selectNode("output"); store.previewExecuted = ["input"]
            XCTAssertFalse(store.canResumeSelected)
            store.previewExecuted = store.validation.orderedNodeIDs
            XCTAssertTrue(store.canResumeSelected)
            store.selectNode("branch-a"); store.renameNode("A user label")
            XCTAssertTrue(store.cacheValid)
            store.setConfig("out_features", value: 10)
            XCTAssertFalse(store.cacheValid)
            store.undo(); XCTAssertTrue(store.cacheValid)
            try FileManager.default.removeItem(at: cache)
            XCTAssertFalse(store.cacheValid)
        }
    }
    @MainActor func testCacheRefusesSymlinkBeforeHashAndCapture() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer{try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let cache=root.appendingPathComponent("cache.pt"),metadata=cache.deletingPathExtension().appendingPathExtension("json")
        try Data("cached".utf8).write(to:cache);try Data("metadata".utf8).write(to:metadata)
        store.previewCachePath=cache.path;store.previewSignature=store.semanticSignature(store.graph);store.capturePreviewArtifactHashes()
        XCTAssertTrue(store.cacheValid)
        let moved=root.appendingPathComponent("actual.pt")
        try FileManager.default.moveItem(at:cache,to:moved)
        try FileManager.default.createSymbolicLink(at:cache,withDestinationURL:moved)
        XCTAssertFalse(store.cacheValid,"Do not follow nonregular artifact paths to hash their contents")
        store.capturePreviewArtifactHashes();XCTAssertNil(store.previewArtifactHashes[cache.path])
    }

}
