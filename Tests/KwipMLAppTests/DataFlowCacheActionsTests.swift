import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowCacheActionsTests: XCTestCase {
    @MainActor func testRealMaterializeReuseAndTamperedPin() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        workspace.activeProjectID = UUID().uuidString
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadExample(columns: ["x", "y"])
        let csv = directory.appendingPathComponent("source.csv")
        try (["x,y"] + (0..<40).map { "\($0),\($0+1)" }).joined(separator: "\n").write(to: csv, atomically: true, encoding: .utf8)
        store.project.sourceBindings["source"] = csv.path
        XCTAssertEqual(store.cachePolicy, .materialize)
        store.runPreparation()
        let firstDeadline = Date().addingTimeInterval(20)
        while store.isBusy && Date() < firstDeadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertNotNil(store.lastCacheReference, store.statusMessage)
        XCTAssertEqual(store.cacheReport?.executedNodeIDs.count, 7)
        XCTAssertEqual(Set(store.cacheReport?.executedNodeIDs ?? []),["source","cast","split","fit","train","validation","test"])
        store.setCachePolicy(.reuse)
        store.runPreparation()
        let secondDeadline = Date().addingTimeInterval(20)
        while store.isBusy && Date() < secondDeadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertEqual(store.cacheReport?.reusedNodeIDs.count, 6, store.statusMessage)
        XCTAssertEqual(store.cacheReport?.executedNodeIDs, ["source"])
        XCTAssertEqual(Set(store.cacheReport?.reusedNodeIDs ?? []),["cast","split","fit","train","validation","test"])
        XCTAssertEqual(store.previewStatus(for: "fit"), "Reused")
        store.save()
        let reopened = DataFlowWorkspaceStore(workspace: workspace)
        XCTAssertEqual(reopened.cachePolicy, .reuse)
        XCTAssertEqual(reopened.lastCacheReference, store.lastCacheReference)
        let reference = try XCTUnwrap(store.lastCacheReference)
        try Data("{}".utf8).write(to: URL(fileURLWithPath: reference.manifestPath))
        XCTAssertThrowsError(try store.preparationCacheArguments())
        store.setCachePolicy(.materialize)
        XCTAssertNoThrow(try store.preparationCacheArguments())
    }
}
