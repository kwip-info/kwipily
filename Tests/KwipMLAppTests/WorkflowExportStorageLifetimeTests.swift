import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExportStorageLifetimeTests: XCTestCase {
    private func fixture() throws -> (URL, URL, WorkflowStorageAdmission, WorkflowStorageAdmission.Reservation) {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let scope = try WorkflowStorageAdmission.Scope(namespace: .workflowEvaluationExports, components: [UUID().uuidString, UUID().uuidString])
        let token = try ledger.reserve(scopes: [scope], peakBytes: 623_000_000, limitBytes: 623_000_000)
        let directory = scope.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 10).write(to: directory.appendingPathComponent("package"))
        return (root, directory, ledger, token)
    }
    func testReviewLifetimeRetainsAllowanceUntilExplicitRemovalOrAbandonment() throws {
        for explicit in [false, true] {
            let (_, directory, ledger, token) = try fixture()
            var lifetime: WorkflowExportStorageLifetime? = try .init(directory: directory, reservation: token)
            XCTAssertNotNil(lifetime)
            XCTAssertEqual(try ledger.snapshot().committedBytes, 623_000_000)
            XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("package")).count, 10)
            if explicit {
                lifetime?.remove(); lifetime?.remove()
                XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
                XCTAssertEqual(try ledger.snapshot().committedBytes, 0)
            }
            lifetime = nil
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
            XCTAssertEqual(try ledger.snapshot().committedBytes, 0)
        }
    }
    func testReplacementSurvivesAndRetainedBytesRemainAccounted() throws {
        let (root, directory, ledger, token) = try fixture()
        var lifetime: WorkflowExportStorageLifetime? = try .init(directory: directory, reservation: token)
        let original = root.appendingPathComponent("moved-original")
        try FileManager.default.moveItem(at: directory, to: original)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try Data(repeating: 2, count: 7).write(to: directory.appendingPathComponent("foreign"))
        lifetime?.remove(); lifetime = nil
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("foreign")).count, 7)
        XCTAssertEqual(try Data(contentsOf: original.appendingPathComponent("package")).count, 10)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 7)
    }
    func testBoundAuthorityRetiresBeforeDirectoryRemovalAndCanRetryAfterOwnerExit() throws {
        let (_, directory, ledger, token) = try fixture()
        let owner = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive, create: true)
        let activity = try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .exclusive, create: true)
        defer { owner.close(); activity.close() }
        let scope = try WorkflowStorageAdmission.Scope(namespace: .workflowEvaluationExports,
            components: [directory.deletingLastPathComponent().lastPathComponent, directory.lastPathComponent])
        try token.bind(to: .init(scope: scope, directoryIdentity: owner.parentIdentity,
            ownerLockIdentity: owner.identity, activityLockIdentity: activity.identity))
        let lifetime = try WorkflowExportStorageLifetime(directory: directory, reservation: token)
        XCTAssertThrowsError(try token.retire())
        lifetime.remove()
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 623_000_000)
        owner.close(); activity.close()
        lifetime.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 0)
        XCTAssertNoThrow(try token.retire(), "Retirement remains idempotent after authority-directory cleanup")
    }
    func testAbandonedReviewNeverDeletesAnActiveBoundAuthority() throws {
        let (_, directory, ledger, token) = try fixture()
        let owner = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive, create: true)
        let activity = try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .exclusive, create: true)
        defer { owner.close(); activity.close() }
        let scope = try WorkflowStorageAdmission.Scope(namespace: .workflowEvaluationExports,
            components: [directory.deletingLastPathComponent().lastPathComponent, directory.lastPathComponent])
        try token.bind(to: .init(scope: scope, directoryIdentity: owner.parentIdentity,
            ownerLockIdentity: owner.identity, activityLockIdentity: activity.identity))
        var lifetime: WorkflowExportStorageLifetime? = try .init(directory: directory, reservation: token)
        XCTAssertNotNil(lifetime)
        lifetime = nil
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 623_000_000)
        owner.close(); activity.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("package")).count, 10)
    }
    func testExplicitWriterLifetimeWithoutReservationStillCleansItsOwnedPackage() throws {
        let (root, _, _, token) = try fixture()
        token.close()
        let directory = root.appendingPathComponent("unlimited-export")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        var lifetime: WorkflowExportStorageLifetime? = try .init(directory: directory, reservation: nil)
        XCTAssertNotNil(lifetime)
        lifetime = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }
}
