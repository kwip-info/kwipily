import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class WorkflowPublicationCleanupTests: XCTestCase {
    private let childKey = "KWIP_REC_PUBLICATION_CHILD"
    private func fixture() throws -> URL {
        let raw = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil)); defer { free(raw) }
        let root = URL(fileURLWithPath: String(cString: raw)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        for name in ["owner.lock", "activity.lock"] { let lease = try WorkflowFileLease(directory: root, name: name, mode: .exclusive, create: true); lease.close() }
        return root
    }
    private func crashPublication(in root: URL, publication: String = "staged") throws {
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["xctest", "-XCTest", "KwipMLCoreTests.WorkflowPublicationCleanupTests/testActualPublicationCrashCleanupPreservesEvidence", Bundle(for: Self.self).bundleURL.path]
        var environment = ProcessInfo.processInfo.environment; environment[childKey] = root.path; environment["KWIP_REC_PUBLICATION_STAGE"] = publication; process.environment = environment
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        try process.run(); process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 78, "Fixture must exit without Swift cleanup after durable publication staging.")
    }
    func testActualPublicationCrashCleanupPreservesEvidence() throws {
        if let root = ProcessInfo.processInfo.environment[childKey] {
            let owner = try WorkflowFileLease(directory: URL(fileURLWithPath: root), name: "owner.lock", mode: .exclusive)
            let temporary = try owner.makePublicationTemporary(target: "group.json", maximumBytes: 256_000)
            try withExtendedLifetime((owner, temporary)) {
                try temporary.write(Data("unfinished metadata publication".utf8))
                let stage = ProcessInfo.processInfo.environment["KWIP_REC_PUBLICATION_STAGE"]
                if stage == "linked" { guard link(URL(fileURLWithPath: root).appendingPathComponent(temporary.name).path, URL(fileURLWithPath: root).appendingPathComponent("group.json").path) == 0 else { Darwin._exit(79) } }
                if stage == "renamed" { guard rename(URL(fileURLWithPath: root).appendingPathComponent(temporary.name).path, URL(fileURLWithPath: root).appendingPathComponent("group.json").path) == 0 else { Darwin._exit(79) } }
                Darwin._exit(78)
            }
        }
        let root = try fixture()
        let preserved = ["group.json", "outcome.json", "inputs.json", "partial-model.pt", ".ownership-\(UUID().uuidString).tmp", "notes.txt"]
        for name in preserved { try Data("preserve \(name)".utf8).write(to: root.appendingPathComponent(name)) }
        try crashPublication(in: root)
        let owner = try WorkflowFileLease(directory: root, name: "owner.lock", mode: .exclusive)
        let activity = try WorkflowFileLease(directory: root, name: "activity.lock", mode: .exclusive)
        defer { activity.close(); owner.close() }
        let report = try owner.cleanupPublicationTemporaries(activity: activity)
        XCTAssertEqual(report.removedTemporaryFiles, 1); XCTAssertEqual(report.removedReceipts, 1); XCTAssertTrue(report.retainedReceipts.isEmpty)
        for name in preserved { XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent(name)), Data("preserve \(name)".utf8)) }
        XCTAssertEqual(try owner.cleanupPublicationTemporaries(activity: activity).removedTemporaryFiles, 0)
        owner.close(); XCTAssertThrowsError(try owner.cleanupPublicationTemporaries(activity: activity))
    }
    func testPublishedCanonicalLinkAndRenamedTargetSurviveCleanup() throws {
        for stage in ["linked", "renamed"] {
            let root = try fixture(); try crashPublication(in: root, publication: stage)
            let owner = try WorkflowFileLease(directory: root, name: "owner.lock", mode: .exclusive)
            let activity = try WorkflowFileLease(directory: root, name: "activity.lock", mode: .exclusive)
            defer { activity.close(); owner.close() }
            let report = try owner.cleanupPublicationTemporaries(activity: activity)
            XCTAssertEqual(report.removedTemporaryFiles, stage == "linked" ? 1 : 0)
            XCTAssertEqual(report.removedReceipts, 1); XCTAssertTrue(report.retainedReceipts.isEmpty)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("group.json")), Data("unfinished metadata publication".utf8))
        }
    }
    func testReplacedSymlinkIncompleteOrExtraLinkedEvidenceIsRetained() throws {
        for mutation in ["replacement", "symlink", "incomplete", "extraLink"] {
            let root = try fixture(); try crashPublication(in: root)
            let names = try FileManager.default.contentsOfDirectory(atPath: root.path)
            let temp = root.appendingPathComponent(try XCTUnwrap(names.first { $0.hasSuffix(".tmp") }))
            let receipt = root.appendingPathComponent(try XCTUnwrap(names.first { $0.hasSuffix(".receipt.json") }))
            let victim = root.appendingPathComponent("retained-evidence")
            if mutation == "replacement" || mutation == "symlink" {
                try FileManager.default.moveItem(at: temp, to: victim)
                if mutation == "replacement" { try Data("unrelated replacement".utf8).write(to: temp) }
                else { try FileManager.default.createSymbolicLink(at: temp, withDestinationURL: victim) }
            } else if mutation == "incomplete" { try Data("{".utf8).write(to: receipt) }
            else { XCTAssertEqual(link(temp.path, victim.path), 0) }
            let owner = try WorkflowFileLease(directory: root, name: "owner.lock", mode: .exclusive)
            let activity = try WorkflowFileLease(directory: root, name: "activity.lock", mode: .exclusive)
            defer { activity.close(); owner.close() }
            let report = try owner.cleanupPublicationTemporaries(activity: activity)
            XCTAssertEqual(report.removedTemporaryFiles, 0, mutation); XCTAssertEqual(report.removedReceipts, 0, mutation)
            XCTAssertEqual(report.retainedReceipts.count, 1, mutation)
            XCTAssertTrue(FileManager.default.fileExists(atPath: receipt.path), mutation)
            XCTAssertTrue(FileManager.default.fileExists(atPath: temp.path), mutation)
        }
    }
    func testRequiresBothExclusiveUnchangedLocksAndRejectsArbitraryTargets() throws {
        let root = try fixture(); try crashPublication(in: root)
        let owner = try WorkflowFileLease(directory: root, name: "owner.lock", mode: .exclusive)
        let shared = try WorkflowFileLease(directory: root, name: "activity.lock", mode: .shared)
        defer { owner.close(); shared.close() }
        XCTAssertThrowsError(try owner.cleanupPublicationTemporaries(activity: shared))
        XCTAssertThrowsError(try owner.makePublicationTemporary(target: "checkpoint.pt", maximumBytes: 100))
        XCTAssertThrowsError(try owner.makePublicationTemporary(target: "../group.json", maximumBytes: 100))
        shared.close()
        let activity = try WorkflowFileLease(directory: root, name: "activity.lock", mode: .exclusive)
        defer { activity.close() }
        try FileManager.default.moveItem(at: root.appendingPathComponent("activity.lock"), to: root.appendingPathComponent("old-activity.lock"))
        try Data().write(to: root.appendingPathComponent("activity.lock"))
        XCTAssertThrowsError(try owner.cleanupPublicationTemporaries(activity: activity))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasSuffix(".receipt.json") }.count, 1)
    }
}
