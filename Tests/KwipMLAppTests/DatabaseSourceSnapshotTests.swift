import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DatabaseSourceSnapshotTests: XCTestCase {
    private func workspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func receipt(_ data: Data, hash: String? = nil) throws -> Data {
        try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "kind": "databaseCapture", "format": "ipc",
            "sampleOnly": false, "sha256": hash ?? WorkflowArchiveCodec.hash(data), "bytes": data.count, "rows": 1, "columns": 1])
    }
    func testReservationSpansCaptureAndRetainedBytesAreAccountedAfterCommit() throws {
        let root = try workspace(), ledger = try WorkflowStorageAdmission(workspace: root)
        let capture = try DatabaseSourceSnapshot(workspace: root, storageLimitBytes: 20_000_000)
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 10_000_000)
        let data = Data("private captured table".utf8)
        try capture.output.write(contentsOf: data)
        try capture.verify(receipt: receipt(data))
        capture.finish(retain: true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: capture.destination.path))
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertEqual(try ledger.snapshot().retainedBytes, Int64(data.count))
        XCTAssertThrowsError(try DatabaseSourceSnapshot(workspace: root, storageLimitBytes: 10_000_000))
    }
    func testFailedIntegrityAndCancellationReleaseOnlyOwnedPrivateCapture() throws {
        let root = try workspace(), ledger = try WorkflowStorageAdmission(workspace: root)
        let capture = try DatabaseSourceSnapshot(workspace: root, storageLimitBytes: 20_000_000)
        let data = Data("private".utf8); try capture.output.write(contentsOf: data)
        XCTAssertThrowsError(try capture.verify(receipt: receipt(data, hash: String(repeating: "0", count: 64))))
        XCTAssertThrowsError(try capture.verify(receipt: receipt(data), checkCancellation: { throw CancellationError() }))
        capture.finish(retain: false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: capture.destination.deletingLastPathComponent().path))
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 0)
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
    }
    func testNamespaceSymlinkCannotRedirectCaptureOutsideManagedStorage() throws {
        let root = try workspace(), external = try workspace()
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("database-source-snapshots"), withDestinationURL: external)
        XCTAssertThrowsError(try DatabaseSourceSnapshot(workspace: root, storageLimitBytes: 20_000_000))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: external.path), [])
    }
    func testReplacedDirectoryCannotBeAcceptedOrDeletedAsOriginal() throws {
        let root = try workspace()
        let capture = try DatabaseSourceSnapshot(workspace: root, storageLimitBytes: 20_000_000)
        let data = Data("private".utf8); try capture.output.write(contentsOf: data)
        let original = capture.destination.deletingLastPathComponent(), moved = original.appendingPathExtension("moved")
        try FileManager.default.moveItem(at: original, to: moved)
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: false)
        let foreign = original.appendingPathComponent("foreign.txt"); try Data("preserve".utf8).write(to: foreign)
        XCTAssertThrowsError(try capture.verify(receipt: receipt(data)))
        capture.finish(retain: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: foreign.path))
    }
}
