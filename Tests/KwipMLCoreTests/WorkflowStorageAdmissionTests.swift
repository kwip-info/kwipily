import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class WorkflowStorageAdmissionTests: XCTestCase {
    private func fixture() throws -> URL {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func scope() throws -> WorkflowStorageAdmission.Scope {
        try .init(namespace: .workflowOperations, components: [UUID().uuidString, UUID().uuidString])
    }
    private func write(_ bytes: Int, root: URL, scope: WorkflowStorageAdmission.Scope) throws {
        let directory = scope.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(repeating: 1, count: bytes).write(to: directory.appendingPathComponent("payload"))
    }
    func testWorkspaceDoesNotRequireListingAncestors() throws {
        let ancestor = try fixture()
        let root = ancestor.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        XCTAssertEqual(chmod(ancestor.path, 0o100), 0)
        defer { _ = chmod(ancestor.path, 0o700) }
        let admission = try WorkflowStorageAdmission(workspace: root)
        XCTAssertEqual(try admission.snapshot().committedBytes, 0)
        let reservation = try admission.reserve(scopes: [scope()], peakBytes: 80, limitBytes: 100)
        defer { reservation.close() }
        XCTAssertEqual(try admission.snapshot().reservedGrowthBytes, 80)
    }
    func testInstalledOCRArchivesRemainChargedAfterDownloadStagingEnds()throws {
        let root=try fixture(),ledger=try WorkflowStorageAdmission(workspace:root)
        let installed=root.appendingPathComponent("ocr-packs/english-printed-v1")
        try FileManager.default.createDirectory(at:installed,withIntermediateDirectories:true)
        try Data(repeating:1,count:70).write(to:installed.appendingPathComponent("reviewed.zip"))
        XCTAssertEqual(try ledger.snapshot().retainedBytes,70)
        XCTAssertThrowsError(try ledger.reserve(scopes:[scope()],peakBytes:31,limitBytes:100))
    }

    func testAggregateReservationsAcrossInstancesDoNotDoubleChargeMaterializedGrowth() throws {
        let root = try fixture(), first = try WorkflowStorageAdmission(workspace: root), second = try WorkflowStorageAdmission(workspace: root)
        let owned = try scope(), reservation = try first.reserve(scopes: [owned], peakBytes: 80, limitBytes: 100)
        defer { reservation.close() }
        try write(20, root: root, scope: owned)
        let snapshot = try second.snapshot()
        XCTAssertEqual(snapshot.retainedBytes, 20); XCTAssertEqual(snapshot.reservedGrowthBytes, 60); XCTAssertEqual(snapshot.committedBytes, 80)
        XCTAssertThrowsError(try second.reserve(scopes: [scope()], peakBytes: 21, limitBytes: 100))
        let remaining = try second.reserve(scopes: [scope()], peakBytes: 20, limitBytes: 100)
        defer { remaining.close() }
        XCTAssertEqual(try first.snapshot().committedBytes, 100)
        try write(90, root: root, scope: owned)
        XCTAssertEqual(try second.snapshot().committedBytes, 110, "Accounting reports an overrun instead of hiding bytes beyond a reservation")
        XCTAssertThrowsError(try first.reserve(scopes: [scope()], peakBytes: 1, limitBytes: 100))
    }
    func testQuotaRefusalReportsCapturedAllowanceDemandAndRemedy() throws {
        let admission = try WorkflowStorageAdmission(workspace: fixture())
        XCTAssertThrowsError(try admission.reserve(scopes: [scope()], peakBytes: 120, limitBytes: 100)) { error in
            let message = error.localizedDescription
            XCTAssertTrue(message.contains("100 bytes"))
            XCTAssertTrue(message.contains("120 bytes"))
            XCTAssertTrue(message.contains("Settings"))
            XCTAssertTrue(message.contains("Existing work was retained"))
        }
        XCTAssertEqual(try admission.snapshot().committedBytes, 0)
    }

    func testPopulatedProposedScopeIsChargedOnceAndCannotHideExistingBytes() throws {
        let root = try fixture(), ledger = try WorkflowStorageAdmission(workspace: root), owned = try scope()
        try write(50, root: root, scope: owned)
        let reservation = try ledger.reserve(scopes: [owned], peakBytes: 80, limitBytes: 80)
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 50)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80)
        XCTAssertThrowsError(try ledger.reserve(scopes: [scope()], peakBytes: 21, limitBytes: 100))
        reservation.close()
        try write(90, root: root, scope: owned)
        XCTAssertThrowsError(try ledger.reserve(scopes: [owned], peakBytes: 80, limitBytes: 80))
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 90)
    }
    func testStaleStageReservationReleasesOnlyAllowanceAndNeverDeletesPayload() throws {
        let root = try fixture(), ledger = try WorkflowStorageAdmission(workspace: root), owned = try scope()
        var reservation: WorkflowStorageAdmission.Reservation? = try ledger.reserve(scopes: [owned], peakBytes: 80, limitBytes: 100)
        try write(37, root: root, scope: owned)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 80)
        reservation = nil
        XCTAssertNil(reservation)
        let reopened = try WorkflowStorageAdmission(workspace: root), snapshot = try reopened.snapshot()
        XCTAssertEqual(snapshot.retainedBytes, 37); XCTAssertEqual(snapshot.reservedGrowthBytes, 0); XCTAssertEqual(snapshot.committedBytes, 37)
        let path = owned.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }.appendingPathComponent("payload")
        XCTAssertEqual(try Data(contentsOf: path).count, 37)
        let next = try reopened.reserve(scopes: [scope()], peakBytes: 63, limitBytes: 100); next.close()
    }
    func testClosedNamespacesExcludeRuntimeLogsAndRejectSharedOrOverlappingScopes() throws {
        let root = try fixture(), ledger = try WorkflowStorageAdmission(workspace: root), owned = try scope()
        try Data(repeating: 0, count: 1000).write(to: root.appendingPathComponent("test.log"))
        try FileManager.default.createDirectory(at: root.appendingPathComponent("runtime"), withIntermediateDirectories: true)
        try Data(repeating: 0, count: 1000).write(to: root.appendingPathComponent("runtime/payload"))
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 0)
        XCTAssertNil(WorkflowStorageAdmission.Namespace(rawValue: "runtime"))
        XCTAssertThrowsError(try WorkflowStorageAdmission.Scope(namespace: .workflowOperations, components: []))
        XCTAssertThrowsError(try WorkflowStorageAdmission.Scope(namespace: .workflowOperations, components: [UUID().uuidString, "..", UUID().uuidString]))
        XCTAssertThrowsError(try WorkflowStorageAdmission.Scope(namespace: .sourceSnapshots, components: [String(repeating: "a", count: 64)]))
        let live = try ledger.reserve(scopes: [owned], peakBytes: 10, limitBytes: 100); defer { live.close() }
        XCTAssertThrowsError(try ledger.reserve(scopes: [owned], peakBytes: 10, limitBytes: 100))
        XCTAssertThrowsError(try ledger.reserve(scopes: [scope(), owned], peakBytes: 10, limitBytes: 100))
        let shared = root.appendingPathComponent("source-snapshots")
        try FileManager.default.createDirectory(at: shared, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 5).write(to: shared.appendingPathComponent("captured.csv"))
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 5)
    }
    func testSparseFilesCountLogicalBytesAndTraversalRejectsLinksOrExceededBounds() throws {
        let root = try fixture(), owned = try scope(), ledger = try WorkflowStorageAdmission(workspace: root)
        try write(0, root: root, scope: owned)
        let directory = owned.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }, path = directory.appendingPathComponent("payload")
        let descriptor = open(path.path, O_RDWR)
        XCTAssertGreaterThanOrEqual(descriptor, 0); defer { if descriptor >= 0 { close(descriptor) } }
        XCTAssertEqual(ftruncate(descriptor, 134_217_728), 0)
        XCTAssertEqual(try ledger.snapshot().retainedBytes, 134_217_728)
        XCTAssertThrowsError(try ledger.reserve(scopes: [scope()], peakBytes: 1, limitBytes: 134_217_728))
        try FileManager.default.linkItem(at: path, to: directory.appendingPathComponent("hardlink"))
        XCTAssertThrowsError(try ledger.snapshot())
        try FileManager.default.removeItem(at: directory.appendingPathComponent("hardlink"))
        try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent("link"), withDestinationURL: path)
        XCTAssertThrowsError(try ledger.snapshot())
        try FileManager.default.removeItem(at: directory.appendingPathComponent("link"))
        let bounded = try WorkflowStorageAdmission(workspace: root, limits: .init(maximumEntries: 2, maximumDepth: 8))
        XCTAssertThrowsError(try bounded.snapshot())
    }
    func testGlobalAdmissionLeaseRefusesCompetingProcessWithoutPayloadWrites() throws {
        let root = try fixture(), ledger = try WorkflowStorageAdmission(workspace: root)
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Development Python required for process lock test") }
        let process = Process(), output = Pipe(), lifetime = Pipe()
        process.executableURL = python
        process.arguments = ["-c", "import fcntl,os,signal,sys\nsignal.alarm(5)\nf=os.open(sys.argv[1],os.O_RDWR)\nfcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nprint('READY',flush=True)\nsys.stdin.buffer.read(1)", root.appendingPathComponent("storage-accounting/owner.lock").path]
        process.standardOutput = output; process.standardInput = lifetime; process.standardError = FileHandle.nullDevice
        try process.run()
        defer { try? lifetime.fileHandleForWriting.close(); if process.isRunning { process.terminate() }; process.waitUntilExit() }
        XCTAssertEqual(String(decoding: output.fileHandleForReading.availableData, as: UTF8.self), "READY\n")
        XCTAssertThrowsError(try ledger.reserve(scopes: [scope()], peakBytes: 80, limitBytes: 100))
        try lifetime.fileHandleForWriting.close(); process.waitUntilExit()
        let admitted = try ledger.reserve(scopes: [scope()], peakBytes: 80, limitBytes: 100); admitted.close()
    }
    func testExternalNamespacesShareGlobalRetainedAndReservedAllowance() throws {
        let app = try fixture(), a = app.appendingPathComponent("external-projects/" + UUID().uuidString)
        let b = app.appendingPathComponent("external-projects/" + UUID().uuidString)
        let local = try WorkflowStorageAdmission(workspace: app, accountingRoot: app)
        let first = try WorkflowStorageAdmission(workspace: a, accountingRoot: app)
        let second = try WorkflowStorageAdmission(workspace: b, accountingRoot: app)
        let sameScope = try scope()
        try write(30, root: app, scope: scope())
        try write(20, root: a, scope: sameScope)
        let one = try first.reserve(scopes: [sameScope], peakBytes: 40, limitBytes: 100)
        defer { one.close() }
        XCTAssertEqual(try second.snapshot().committedBytes, 70)
        XCTAssertThrowsError(try second.reserve(scopes: [sameScope], peakBytes: 31, limitBytes: 100))
        let two = try second.reserve(scopes: [sameScope], peakBytes: 30, limitBytes: 100)
        XCTAssertEqual(try local.snapshot().committedBytes, 100)
        two.close(); one.close()
        XCTAssertEqual(try local.snapshot().retainedBytes, 50)
        XCTAssertEqual(try first.snapshot().committedBytes, 50)
    }
    func testExternalAccountingRejectsUnrelatedAndLinkedNamespaces() throws {
        let app = try fixture(), other = try fixture()
        XCTAssertThrowsError(try WorkflowStorageAdmission(workspace: other, accountingRoot: app))
        let parent = app.appendingPathComponent("external-projects")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        let link = parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: other)
        XCTAssertThrowsError(try WorkflowStorageAdmission(workspace: link, accountingRoot: app))
        let global = try WorkflowStorageAdmission(workspace: app, accountingRoot: app)
        XCTAssertThrowsError(try global.snapshot())
    }

    func testAccountingMemberReplacementCannotRedirectControlWrites() throws {
        for link in [true, false] {
            let app = try fixture(), outside = try fixture()
            let member = app.appendingPathComponent("external-projects/" + UUID().uuidString)
            try FileManager.default.createDirectory(at: member, withIntermediateDirectories: true)
            let global = try WorkflowStorageAdmission(workspace: app, accountingRoot: app)
            global.beforeAccountingMemberOpen = { selected in
                XCTAssertEqual(selected.path, member.path)
                try FileManager.default.removeItem(at: member)
                if link { try FileManager.default.createSymbolicLink(at: member, withDestinationURL: outside) }
                else { try FileManager.default.createDirectory(at: member, withIntermediateDirectories: false) }
            }
            XCTAssertThrowsError(try global.snapshot())
            XCTAssertFalse(FileManager.default.fileExists(atPath: outside.appendingPathComponent("storage-accounting").path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: member.appendingPathComponent("storage-accounting").path))
        }
    }
    func testAccountingScanLimitIsSharedAcrossMembers() throws {
        let app = try fixture(), member = app.appendingPathComponent("external-projects/" + UUID().uuidString)
        try write(1, root: app, scope: scope())
        try write(1, root: member, scope: scope())
        let global = try WorkflowStorageAdmission(workspace: app, accountingRoot: app, limits: .init(maximumEntries: 6))
        XCTAssertThrowsError(try global.snapshot()) { error in
            guard case WorkflowStorageAdmission.Failure.scanLimit = error else { return XCTFail("Expected shared scan limit, got \(error)") }
        }
    }

}
