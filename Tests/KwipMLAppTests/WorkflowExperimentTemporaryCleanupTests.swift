import XCTest
import Darwin
@testable import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExperimentTemporaryCleanupTests: XCTestCase {
    private let childKey = "KWIP_REC_GROUP_PUBLICATION_CHILD"
    func testOptionalCleanupRunsAfterRevocationAndRetainsPartialEvidence() throws {
        if let path = ProcessInfo.processInfo.environment[childKey] {
            let owner = try WorkflowFileLease(directory: URL(fileURLWithPath: path), name: "owner.lock", mode: .exclusive)
            let scratch = try owner.makePublicationTemporary(target: "group.json", maximumBytes: 256_000)
            try withExtendedLifetime((owner, scratch)) { try scratch.write(Data("uncommitted publication bytes".utf8)); Darwin._exit(78) }
        }
        let f = try WorkflowExperimentRecoveryCoordinatorTests().fixture()
        addTeardownBlock { try? FileManager.default.removeItem(at: f.root) }
        let run = UUID(), directoryID = UUID()
        try f.store.begin(ordinal: 1, runID: run, directoryID: directoryID)
        let phase = f.root.appendingPathComponent("workflow-runs/\(f.store.record.projectID)/\(directoryID.uuidString)/preparation")
        let launch = try f.owner.publishLaunch(projectID: f.store.record.projectID, ordinal: 1, runID: run.uuidString,
            directoryID: directoryID.uuidString, requestIdentity: f.requests[0].requestIdentity, phase: .preparation,
            arguments: ["--execute-dataflow", phase.appendingPathComponent("request.json").path,
                        "--source-bindings", phase.appendingPathComponent("bindings.json").path,
                        "--output", phase.appendingPathComponent("artifacts").path])
        let preserved = f.directory.appendingPathComponent("partial-evidence.bin")
        try Data("partial evidence".utf8).write(to: preserved)
        let inputs = try Data(contentsOf: f.directory.appendingPathComponent("authoring.json"))
        f.owner.close()
        let child = Process(); child.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        child.arguments = ["xctest", "-XCTest", "KwipMLAppTests.WorkflowExperimentTemporaryCleanupTests/testOptionalCleanupRunsAfterRevocationAndRetainsPartialEvidence", Bundle(for: Self.self).bundleURL.path]
        var environment = ProcessInfo.processInfo.environment; environment[childKey] = f.directory.path; child.environment = environment
        child.standardOutput = FileHandle.nullDevice; child.standardError = FileHandle.nullDevice
        try child.run(); child.waitUntilExit(); XCTAssertEqual(child.terminationStatus, 78)
        func receipts() throws -> [String] { try FileManager.default.contentsOfDirectory(atPath: f.directory.path).filter { $0.hasSuffix(".receipt.json") } }
        XCTAssertEqual(try receipts().count, 1)
        let recovery = try WorkflowGroupOwnership.acquireRecovery(directory: f.directory, groupID: f.store.record.groupID)
        XCTAssertThrowsError(try recovery.cleanupPublicationTemporaries())
        XCTAssertEqual(try receipts().count, 1); recovery.close()
        let activity = try WorkflowFileLease(directory: f.directory, name: "activity.lock", mode: .shared)
        XCTAssertThrowsError(try WorkflowExperimentRecoveryCoordinator.recover(directory: f.directory, workspaceDirectory: f.root))
        XCTAssertEqual(try receipts().count, 1); activity.close()
        let result = try WorkflowExperimentRecoveryCoordinator.recover(directory: f.directory, workspaceDirectory: f.root, cleanupOwnedTemporaries: false)
        XCTAssertEqual(result.variants.map(\.status), [.interrupted, .cancelled]); XCTAssertEqual(try receipts().count, 1)
        let claim = try JSONDecoder().decode(WorkflowLaunchClaim.self, from: Data(contentsOf: f.directory.appendingPathComponent("claim-\(launch.nonce).json")))
        XCTAssertEqual(claim.state, "revoked")
        let again = try WorkflowExperimentRecoveryCoordinator.recover(directory: f.directory, workspaceDirectory: f.root)
        XCTAssertEqual(again, result); XCTAssertTrue(try receipts().isEmpty)
        XCTAssertEqual(try Data(contentsOf: preserved), Data("partial evidence".utf8))
        XCTAssertEqual(try Data(contentsOf: f.directory.appendingPathComponent("authoring.json")), inputs)
    }
}
