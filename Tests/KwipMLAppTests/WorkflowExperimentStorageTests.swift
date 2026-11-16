import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExperimentStorageTests: XCTestCase {
    func testCollectionGroupFundsMemberBytesAndLargerInventoryBeforeWriting() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let count = Int64(WorkflowCapturedSourceArchive.maximumAggregateBytes)
        let baseline = count + 5_000_000 + 2 * 520_000
        let project = UUID().uuidString, group = UUID().uuidString
        XCTAssertThrowsError(try WorkflowExperimentStorageAdmission(workspace: root, projectID: project,
            groupID: group, sourceBytes: count, variantCount: 1, limitBytes: Int64.max))
        XCTAssertThrowsError(try WorkflowExperimentStorageAdmission(workspace: root, projectID: project,
            groupID: group, sourceBytes: count, variantCount: 1, hasFileCollections: true, limitBytes: baseline - 1))
        let admission = try WorkflowExperimentStorageAdmission(workspace: root, projectID: project,
            groupID: group, sourceBytes: count, variantCount: 1, hasFileCollections: true, limitBytes: baseline)
        defer { admission.close() }
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, baseline)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-experiments/\(project)/\(group)").path))
    }
    func testGroupAdmissionPrecedesCreationAndSequentialRunsKeepCompletedBytesCharged() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let project = UUID().uuidString, groupID = UUID().uuidString
        let baseline: Int64 = 10 + 3_000_000 + 2 * 2 * 520_000
        XCTAssertThrowsError(try WorkflowExperimentStorageAdmission(workspace: root, projectID: project,
            groupID: groupID, sourceBytes: 10, variantCount: 2, limitBytes: baseline - 1))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("workflow-experiments").path))
        let storage = try WorkflowExperimentStorageAdmission(workspace: root, projectID: project,
            groupID: groupID, sourceBytes: 10, variantCount: 2, limitBytes: baseline + 100)
        let group = try WorkflowGroupOwnership.createGroup(workspace: root, projectID: project, groupID: groupID)
        defer { group.owner.close(); storage.close() }
        try storage.bind(to: group.owner)
        let ledger = try WorkflowStorageAdmission(workspace: root)
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline)
        let first = UUID().uuidString, second = UUID().uuidString
        try storage.admitVariant(directoryID: first, runPeakBytes: 80)
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline + 80)
        let run = root.appendingPathComponent("workflow-runs/\(project)/\(first)")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 30).write(to: run.appendingPathComponent("model"))
        XCTAssertThrowsError(try storage.admitVariant(directoryID: second, runPeakBytes: 71))
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline + 80)
        let activity = try WorkflowFileLease(directory: group.directory, name: "activity.lock", mode: .shared,
            expectedIdentity: group.owner.metadata.activityLockIdentity, requireLocalAPFS: true)
        XCTAssertThrowsError(try storage.admitVariant(directoryID: second, runPeakBytes: 70)) { error in
            guard case WorkflowFileLeaseError.busy = error else { return XCTFail("Expected active worker refusal: \(error)") }
        }
        activity.close()
        try storage.admitVariant(directoryID: second, runPeakBytes: 70)
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline + 100)
        group.owner.close(); storage.close()
        XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
        XCTAssertEqual(try Data(contentsOf: run.appendingPathComponent("model")).count, 30)
    }
}
