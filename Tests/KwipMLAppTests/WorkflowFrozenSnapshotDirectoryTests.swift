import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowFrozenSnapshotDirectoryTests: XCTestCase {
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).standardizedFileURL
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    func testAbandonedNativeOwnershipDeletesSnapshotButNeverReplacement() throws {
        let root = try fixture(), directory = root.appendingPathComponent("native")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        var ownership: WorkflowFrozenSnapshotDirectory? = try .init(directory: directory)
        XCTAssertNotNil(ownership)
        try Data("owned".utf8).write(to: directory.appendingPathComponent("payload"))
        ownership = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        ownership = try .init(directory: directory)
        let moved = root.appendingPathComponent("moved")
        try FileManager.default.moveItem(at: directory, to: moved)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try Data("replacement".utf8).write(to: directory.appendingPathComponent("payload"))
        ownership?.remove(); ownership?.remove(); ownership = nil
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("payload")), Data("replacement".utf8))
        XCTAssertTrue(FileManager.default.fileExists(atPath: moved.path))
    }
    func testGenericSnapshotDoesNotOwnPersistedWorkerInput() throws {
        let directory = try fixture(), pin = String(repeating: "a", count: 64)
        let reference = WorkflowFrozenModelReference(projectID: UUID().uuidString, directoryID: UUID().uuidString,
            runID: UUID().uuidString, requestIdentity: pin, outcomeSHA256: pin, modelManifestSHA256: pin,
            preparationManifestSHA256: pin, checkpointSHA256: pin)
        var snapshot: WorkflowFrozenModelSnapshot? = .init(reference: reference, directory: directory, files: [], receiptSHA256: pin)
        XCTAssertNotNil(snapshot)
        snapshot = nil
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
    }
}
