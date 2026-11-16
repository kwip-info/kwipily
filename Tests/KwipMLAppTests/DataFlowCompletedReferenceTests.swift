import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowCompletedReferenceTests: XCTestCase {
    @MainActor func testOperationReferencesRoundTripAndRejectWrongProjectOrVersion() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        let projectID = UUID().uuidString
        workspace.activeProjectID = projectID
        let store = DataFlowWorkspaceStore(workspace: workspace)
        let relative = "workflow-operations/\(projectID)/\(UUID().uuidString)/artifacts/manifest.json"
        let manifest = directory.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: manifest.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["graphDigest": String(repeating: "a", count: 64)]).write(to: manifest)
        try DataFlowCodec.encodeGraph(store.project.graph).write(to: manifest.deletingLastPathComponent().appendingPathComponent("dataflow.json"))
        // macOS workspace aliases and canonical worker paths must address the same result.
        let pointer = try XCTUnwrap(realpath(manifest.path, nil))
        defer { free(pointer) }
        let canonical = URL(fileURLWithPath: String(cString: pointer))
        try store.persistCompletedResult(manifest: canonical, graph: store.project.graph, bindings: [:], digests: [:])
        let saved = try XCTUnwrap(store.completedResult)
        XCTAssertEqual(saved.schemaVersion, 2)
        XCTAssertEqual(saved.manifestRelativePath, relative)
        XCTAssertEqual(try store.readCompletedReference().0, saved)
        for path in [relative.replacingOccurrences(of: projectID, with: UUID().uuidString),
                     relative.replacingOccurrences(of: projectID, with: projectID.lowercased()),
                     "workflow-operations/\(projectID)/not-a-uuid/artifacts/manifest.json",
                     "workflow-operations/\(projectID)/../artifacts/manifest.json"] {
            var invalid = saved; invalid.manifestRelativePath = path
            try JSONEncoder().encode(invalid).write(to: store.completedResultURL)
            XCTAssertThrowsError(try store.readCompletedReference(), path)
        }
        for version in [1, 3] {
            var invalid = saved; invalid.schemaVersion = version
            try JSONEncoder().encode(invalid).write(to: store.completedResultURL)
            XCTAssertThrowsError(try store.readCompletedReference())
        }
        // A failed cross-project save must not replace the last successful pointer.
        try JSONEncoder().encode(saved).write(to: store.completedResultURL)
        let other = directory.appendingPathComponent(relative.replacingOccurrences(of: projectID, with: UUID().uuidString))
        try FileManager.default.createDirectory(at: other.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(contentsOf: manifest).write(to: other)
        XCTAssertThrowsError(try store.persistCompletedResult(manifest: other, graph: store.project.graph, bindings: [:], digests: [:]))
        XCTAssertEqual(try store.readCompletedReference().0, saved)
    }
}
