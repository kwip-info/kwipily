import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class WorkflowStorageBindingTests: XCTestCase {
    private func fixture() throws -> (URL, WorkflowStorageAdmission, WorkflowStorageAdmission.Scope, URL, WorkflowFileLease, WorkflowFileLease) {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil)); defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let scope = try WorkflowStorageAdmission.Scope(namespace: .workflowOperations, components: [UUID().uuidString, UUID().uuidString])
        let directory = scope.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 10).write(to: directory.appendingPathComponent("payload"))
        let owner = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive, create: true)
        let activity = try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .exclusive, create: true)
        return (root, ledger, scope, directory, owner, activity)
    }
    private func authority(_ scope: WorkflowStorageAdmission.Scope, _ owner: WorkflowFileLease, _ activity: WorkflowFileLease) -> WorkflowStorageAdmission.Authority {
        .init(scope: scope, directoryIdentity: owner.parentIdentity, ownerLockIdentity: owner.identity, activityLockIdentity: activity.identity)
    }
    func testBoundReservationSurvivesLaunchGapAndIndependentWorkerAfterNativeOwnerCloses() throws {
        let (root, ledger, scope, directory, owner, activity) = try fixture()
        defer { owner.close(); activity.close() }
        let reservation = try ledger.reserve(scopes: [scope], peakBytes: 80, limitBytes: 100)
        try reservation.bind(to: authority(scope, owner, activity))
        reservation.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80)
        activity.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80, "Native owner covers the release-to-supervisor activity gap")
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Development Python required") }
        let process = Process(), output = Pipe(), lifetime = Pipe()
        process.executableURL = python
        process.arguments = ["-c", "import fcntl,os,signal,sys\nsignal.alarm(5)\nf=os.open(sys.argv[1],os.O_RDWR)\nfcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)\nprint('READY',flush=True)\nsys.stdin.buffer.read(1)", directory.appendingPathComponent("activity.lock").path]
        process.standardOutput = output; process.standardInput = lifetime; process.standardError = FileHandle.nullDevice
        try process.run()
        defer { try? lifetime.fileHandleForWriting.close(); if process.isRunning { process.terminate() }; process.waitUntilExit() }
        XCTAssertEqual(String(decoding: output.fileHandleForReading.availableData, as: UTF8.self), "READY\n")
        owner.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80, "Worker activity holds the allowance after app ownership is gone")
        XCTAssertThrowsError(try ledger.reserve(scopes: [try .init(namespace: .modelTrainingStaging, components: [UUID().uuidString])], peakBytes: 21, limitBytes: 100))
        try lifetime.fileHandleForWriting.close(); process.waitUntilExit()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("payload")).count, 10)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("storage-accounting/reservations/" + reservation.id).path))
    }
    func testBindingRejectsDifferentScopeIdentityAndSharedReaderCannotImpersonateOwner() throws {
        let (_, ledger, scope, directory, owner, activity) = try fixture()
        defer { owner.close(); activity.close() }
        let reservation = try ledger.reserve(scopes: [scope], peakBytes: 80, limitBytes: 100)
        defer { reservation.close() }
        let foreign = try WorkflowStorageAdmission.Scope(namespace: .workflowOperations, components: [UUID().uuidString, UUID().uuidString])
        XCTAssertThrowsError(try reservation.bind(to: authority(foreign, owner, activity)))
        let wrong = WorkflowStorageAdmission.Authority(scope: scope, directoryIdentity: owner.parentIdentity,
            ownerLockIdentity: .init(device: owner.identity.device, inode: owner.identity.inode + 1), activityLockIdentity: activity.identity)
        XCTAssertThrowsError(try reservation.bind(to: wrong))
        try reservation.bind(to: authority(scope, owner, activity))
        XCTAssertThrowsError(try reservation.bind(to: authority(scope, owner, activity)), "Binding is immutable")
        owner.close(); activity.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80, "Native staging lease covers finalization even when bound authority is closed")
        reservation.close()
        let reader = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .shared)
        defer { reader.close() }
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10, "A shared reader must not be mistaken for exclusive native ownership")
    }
    func testBindingRequiresLiveExclusiveOwnerAndCompletedControlRecordsAreReclaimed() throws {
        let (root, ledger, scope, _, owner, activity) = try fixture()
        let proof = authority(scope, owner, activity)
        owner.close(); activity.close()
        let invalid = try ledger.reserve(scopes: [scope], peakBytes: 80, limitBytes: 100)
        XCTAssertThrowsError(try invalid.bind(to: proof)); invalid.close()
        _ = try ledger.snapshot()
        for _ in 0..<20 {
            let stage = try WorkflowStorageAdmission.Scope(namespace: .modelTrainingStaging, components: [UUID().uuidString])
            let reservation = try ledger.reserve(scopes: [stage], peakBytes: 20, limitBytes: 100)
            reservation.close()
            XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("storage-accounting/reservations").path), [])
    }
    func testPartialOrphansReclaimOnlyWhenTheirLeaseIsInactive() throws {
        let (root, ledger, _, directory, owner, activity) = try fixture()
        defer { owner.close(); activity.close() }
        let parent = root.appendingPathComponent("storage-accounting/reservations")
        let empty = parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: false)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        XCTAssertFalse(FileManager.default.fileExists(atPath: empty.path))
        let orphan = parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: orphan, withIntermediateDirectories: false)
        let lease = try WorkflowFileLease(directory: orphan, name: "owner.lock", mode: .exclusive, create: true)
        defer { lease.close() }
        XCTAssertThrowsError(try ledger.snapshot(), "An untracked live staging lease cannot be silently ignored")
        XCTAssertTrue(FileManager.default.fileExists(atPath: orphan.path))
        lease.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("payload")).count, 10)
    }

    func testSequentialRevisionRequiresIdleWorkerAndChargesCompletedVariantBytes() throws {
        let (root, ledger, scope, _, owner, activity) = try fixture()
        defer { owner.close(); activity.close() }
        let first = try WorkflowStorageAdmission.Scope(namespace: .workflowRuns, components: [UUID().uuidString, UUID().uuidString])
        let next = try WorkflowStorageAdmission.Scope(namespace: .workflowRuns, components: [UUID().uuidString, UUID().uuidString])
        let reservation = try ledger.reserve(scopes: [scope, first], peakBytes: 80, limitBytes: 100)
        defer { reservation.close() }
        try reservation.bind(to: authority(scope, owner, activity))
        XCTAssertThrowsError(try reservation.revise(scopes: [scope, next], peakBytes: 60, limitBytes: 100), "Cannot change allowance while worker activity is held")
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80)
        activity.close()
        let completed = first.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }
        try FileManager.default.createDirectory(at: completed, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 30).write(to: completed.appendingPathComponent("model"))
        XCTAssertThrowsError(try reservation.revise(scopes: [next], peakBytes: 60, limitBytes: 100), "Bound authority scope must remain included")
        XCTAssertThrowsError(try reservation.revise(scopes: [scope, next], peakBytes: 71, limitBytes: 100))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80, "Failed revision keeps original allowance")
        try reservation.revise(scopes: [scope, next], peakBytes: 70, limitBytes: 100)
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 40)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 100, "Completed variant bytes remain charged outside next variant scope")
        owner.close()
        XCTAssertThrowsError(try reservation.revise(scopes: [scope, next], peakBytes: 70, limitBytes: 100), "Lost native owner cannot revise a bound reservation")
        reservation.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 40)
    }
    func testRevisionRejectsCompetingScopeAndClosedStage() throws {
        let (_, ledger, scope, _, owner, activity) = try fixture()
        defer { owner.close(); activity.close() }
        let stage = try WorkflowStorageAdmission.Scope(namespace: .modelTrainingStaging, components: [UUID().uuidString])
        let reservation = try ledger.reserve(scopes: [scope], peakBytes: 40, limitBytes: 100)
        let other = try ledger.reserve(scopes: [stage], peakBytes: 20, limitBytes: 100)
        defer { reservation.close(); other.close() }
        XCTAssertThrowsError(try reservation.revise(scopes: [scope, stage], peakBytes: 60, limitBytes: 100))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 60)
        try reservation.revise(scopes: [scope], peakBytes: 50, limitBytes: 100)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 70)
        reservation.close()
        XCTAssertThrowsError(try reservation.revise(scopes: [scope], peakBytes: 60, limitBytes: 100))
    }

    func testPhaseHandoffPreservesAllowanceUntilPriorAuthorityIsInactiveAndCapacityFits() throws {
        let (root, ledger, scope, _, owner, activity) = try fixture()
        let token = try ledger.reserve(scopes: [scope], peakBytes: 80, limitBytes: 100)
        defer { token.close(); owner.close(); activity.close() }
        try token.bind(to: authority(scope, owner, activity))
        XCTAssertThrowsError(try token.prepareNextPhase(limitBytes: 100))
        owner.close()
        XCTAssertThrowsError(try token.prepareNextPhase(limitBytes: 100), "Worker still holds activity")
        activity.close()
        let other = root.appendingPathComponent("runs/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 21).write(to: other.appendingPathComponent("payload"))
        XCTAssertThrowsError(try token.prepareNextPhase(limitBytes: 100))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 101, "Failure retains finalization allowance")
        try token.prepareNextPhase(limitBytes: 101)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 101)
        token.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 31)
    }

}
