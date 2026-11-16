import XCTest
import Darwin
@testable import KwipMLCore

final class WorkflowScratchCleanupTests: XCTestCase {
    private func root() throws -> URL {
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil)); defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func stage(_ root: URL) throws -> (URL, WorkflowScratchCleanup.Lifetime) {
        let directory = root.appendingPathComponent("preview-staging").appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("input"), withIntermediateDirectories: true)
        let lease = try WorkflowScratchCleanup.register(directory: directory)
        try Data([1,2,3]).write(to: directory.appendingPathComponent("input/data"))
        return (directory, lease)
    }
    func testModelDownloadStageNeedsLiveLeaseThenIsReclaimable()throws {
        let root=try root(),directory=root.appendingPathComponent("model-download-staging").appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let lifetime=try WorkflowScratchCleanup.register(directory:directory)
        try Data([1,2,3]).write(to:directory.appendingPathComponent("model.safetensors"))
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace:root).candidates.isEmpty)
        lifetime.close()
        let report=try WorkflowScratchCleanup.scan(workspace:root)
        XCTAssertEqual(report.candidates.count,1)
        XCTAssertEqual(try WorkflowScratchCleanup.remove(report.candidates,workspace:root),3)
    }
    func testLiveStageRetainedThenInactiveStageRemovedWithoutHistoryOrRecovery() throws {
        let root = try root(), (directory, lease) = try stage(root)
        for path in ["workflow-runs/pinned", "projects/project", ".kwipflow-publication-recovery", "source-snapshots/original", "workflow-operations/record"] {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([4]).write(to: url)
        }
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace: root).candidates.isEmpty)
        lease.close()
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        XCTAssertEqual(report.candidates.count, 1); XCTAssertEqual(report.reclaimableBytes, 3)
        XCTAssertEqual(try WorkflowScratchCleanup.remove(report.candidates, workspace: root), 3)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-runs/pinned").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(".kwipflow-publication-recovery").path))
    }
    func testUnregisteredAndSymlinkAndHardlinkStagesRetained() throws {
        let root = try root(), (directory, lease) = try stage(root); lease.close()
        let external = root.appendingPathComponent("external")
        try Data([7]).write(to: external)
        let link = directory.appendingPathComponent("input/link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: external)
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace: root).candidates.isEmpty)
        try FileManager.default.removeItem(at: link)
        XCTAssertEqual(Darwin.link(external.path, link.path), 0)
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace: root).candidates.isEmpty)
        try FileManager.default.removeItem(at: link)
        try FileManager.default.removeItem(at: directory.appendingPathComponent(".kwip-scratch.json"))
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace: root).candidates.isEmpty)
        XCTAssertEqual(try Data(contentsOf: external), Data([7]))
    }
    func testChangedInventoryAndReacquiredLockBlockReviewedCleanup() throws {
        let root = try root(), (directory, lifetime) = try stage(root); lifetime.close()
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        let lock = try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .exclusive, requireLocalAPFS: true)
        XCTAssertThrowsError(try WorkflowScratchCleanup.remove(report.candidates, workspace: root)); lock.close()
        try Data([9]).write(to: directory.appendingPathComponent("input/new"))
        XCTAssertThrowsError(try WorkflowScratchCleanup.remove(report.candidates, workspace: root))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("input/data").path))
    }
    func testCancellationLeavesReceiptForRetry() throws {
        let root = try root(), (directory, lifetime) = try stage(root); lifetime.close()
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        var checks = 0
        XCTAssertThrowsError(try WorkflowScratchCleanup.remove(report.candidates, workspace: root, checkCancellation: {
            checks += 1
            if checks == 5 { throw CancellationError() }
        }))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent(".kwip-scratch.json").path))
        let retry = try WorkflowScratchCleanup.scan(workspace: root)
        _ = try WorkflowScratchCleanup.remove(retry.candidates, workspace: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }
    func testSameSizeMutationAfterPreflightIsRetained() throws {
        let root = try root(), (directory, lifetime) = try stage(root); lifetime.close()
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        let file = directory.appendingPathComponent("input/data")
        var checks = 0, mutated = false
        XCTAssertThrowsError(try WorkflowScratchCleanup.remove(report.candidates, workspace: root, checkCancellation: {
            checks += 1
            if checks == 6 {
                let handle = try FileHandle(forWritingTo: file)
                try handle.write(contentsOf: Data([8,8,8])); try handle.close()
                // Deterministically alter mtime even on a coarse-clock fixture.
                try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 42)], ofItemAtPath: file.path)
                mutated = true
            }
        }))
        XCTAssertTrue(mutated)
        XCTAssertEqual(try Data(contentsOf: file), Data([8,8,8]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent(".kwip-scratch.json").path))
    }
    func testNestedDirectoryReplacedAfterPreflightIsNotTraversed() throws {
        let root = try root(), (directory, lifetime) = try stage(root); lifetime.close()
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        let input = directory.appendingPathComponent("input"), saved = root.appendingPathComponent("saved-input")
        var checks = 0, replaced = false
        XCTAssertThrowsError(try WorkflowScratchCleanup.remove(report.candidates, workspace: root, checkCancellation: {
            checks += 1
            if checks == 6 {
                try FileManager.default.moveItem(at: input, to: saved)
                try FileManager.default.createDirectory(at: input, withIntermediateDirectories: false)
                try Data([9,9,9]).write(to: input.appendingPathComponent("data"))
                replaced = true
            }
        }))
        XCTAssertTrue(replaced)
        XCTAssertEqual(try Data(contentsOf: input.appendingPathComponent("data")), Data([9,9,9]))
        XCTAssertEqual(try Data(contentsOf: saved.appendingPathComponent("data")), Data([1,2,3]))
    }
    func testCrashedProcessReleasesStageLeaseForExplicitCleanup() throws {
        let root = try root(), (directory, lifetime) = try stage(root); lifetime.close()
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = repository.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Development Python required for process crash test") }
        let process = Process(), ready = Pipe(), input = Pipe()
        process.executableURL = python
        process.arguments = ["-c", "import fcntl,os,sys,signal\nsignal.alarm(10)\nf=os.open(sys.argv[1],os.O_RDWR)\nfcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nprint('READY',flush=True)\nsys.stdin.buffer.read(1)\nos._exit(79)", directory.appendingPathComponent("owner.lock").path]
        process.standardOutput = ready; process.standardInput = input; process.standardError = FileHandle.nullDevice
        try process.run()
        defer { try? input.fileHandleForWriting.close(); if process.isRunning { process.terminate() }; process.waitUntilExit() }
        XCTAssertEqual(String(decoding: ready.fileHandleForReading.availableData, as: UTF8.self), "READY\n")
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace: root).candidates.isEmpty)
        try input.fileHandleForWriting.write(contentsOf: Data([1])); process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 79)
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        XCTAssertEqual(report.candidates.count, 1)
        XCTAssertEqual(try WorkflowScratchCleanup.remove(report.candidates, workspace: root), 3)
    }
    func testReplacedStageAndSymlinkAncestorRefused() throws {
        let root = try root(), (directory, lifetime) = try stage(root); lifetime.close()
        let report = try WorkflowScratchCleanup.scan(workspace: root)
        let moved = root.appendingPathComponent("saved")
        try FileManager.default.moveItem(at: directory, to: moved)
        try FileManager.default.createSymbolicLink(at: directory, withDestinationURL: moved)
        XCTAssertThrowsError(try WorkflowScratchCleanup.remove(report.candidates, workspace: root))
        XCTAssertTrue(FileManager.default.fileExists(atPath: moved.appendingPathComponent("input/data").path))
        XCTAssertThrowsError(try WorkflowScratchCleanup.register(directory: root.appendingPathComponent("workflow-runs").appendingPathComponent(UUID().uuidString)))
    }
}
