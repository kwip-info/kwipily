import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCaptureStorageReservationTests: XCTestCase {
    private func workspace() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    func testExactModePeaksAndAdditionalPrivateStageAreAccountedOnce() throws {
        let root = try workspace(), project = UUID().uuidString
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let cases: [(WorkflowPreviewCapture.Purpose, Int64)] = [
            (.preview, 33_100_000), (.modelTraining, 443_000_000), (.preparation, 302_000_000),
            (.sourceInspection, 0), (.preparationInspection, 100_000_000)
        ]
        for (purpose, outputBytes) in cases {
            let operation = UUID().uuidString, stage = UUID()
            let extra = try WorkflowStorageAdmission.Scope(namespace: .preparationSourceStaging, components: [UUID().uuidString])
            let peak = outputBytes + 1_000_000 + 100 + 50
            let context = try WorkflowCaptureStorageReservation(workspace: root, projectID: project, operationID: operation,
                purpose: purpose, stageID: stage, maximumCaptureBytes: 100, additionalScopes: [extra], additionalBytes: 50, limitBytes: peak)
            XCTAssertEqual(context.stageID, stage)
            XCTAssertEqual(context.projectID, project); XCTAssertEqual(context.operationID, operation)
            XCTAssertEqual(context.purpose, purpose); XCTAssertEqual(context.maximumCaptureBytes, 100)
            XCTAssertEqual(try ledger.snapshot().committedBytes, peak)
            XCTAssertThrowsError(try ledger.reserve(scopes: [extra], peakBytes: 1, limitBytes: Int64.max))
            let stageURL = root.appendingPathComponent(purpose.namespace).appendingPathComponent(stage.uuidString)
            try FileManager.default.createDirectory(at: stageURL, withIntermediateDirectories: true)
            try Data(repeating: 1, count: 100).write(to: stageURL.appendingPathComponent("captured"))
            XCTAssertEqual(try ledger.snapshot().committedBytes, peak)
            let token = try context.take(workspace: root, projectID: project, operationID: operation, purpose: purpose)
            token.close()
            XCTAssertEqual(try ledger.snapshot().committedBytes, 100)
            try FileManager.default.removeItem(at: stageURL)
        }
    }
    func testTransferIsOneShotAndForeignContextsDoNotReleaseAllowance() throws {
        let root = try workspace(), foreign = try workspace(), project = UUID().uuidString, operation = UUID().uuidString
        let ledger = try WorkflowStorageAdmission(workspace: root)
        var context: WorkflowCaptureStorageReservation? = try .init(workspace: root, projectID: project, operationID: operation,
            purpose: .sourceInspection, maximumCaptureBytes: 20, limitBytes: 1_000_020)
        XCTAssertThrowsError(try context!.take(workspace: foreign, projectID: project, operationID: operation, purpose: .sourceInspection))
        XCTAssertThrowsError(try context!.take(workspace: root, projectID: UUID().uuidString, operationID: operation, purpose: .sourceInspection))
        XCTAssertThrowsError(try context!.take(workspace: root, projectID: project, operationID: UUID().uuidString, purpose: .sourceInspection))
        XCTAssertThrowsError(try context!.take(workspace: root, projectID: project, operationID: operation, purpose: .preparationInspection))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_000_020)
        let token = try context!.take(workspace: root, projectID: project, operationID: operation, purpose: .sourceInspection)
        XCTAssertThrowsError(try context!.take(workspace: root, projectID: project, operationID: operation, purpose: .sourceInspection))
        context!.close(); context!.close()
        context = nil
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_000_020, "Transferred token outlives its admission context")
        token.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 0)
    }
    func testAbandonedContextReleasesOnlyGrowthAndRefusesReplacedWorkspace() throws {
        let root = try workspace(), project = UUID().uuidString, operation = UUID().uuidString, stage = UUID()
        var context: WorkflowCaptureStorageReservation? = try .init(workspace: root, projectID: project, operationID: operation,
            purpose: .preparation, stageID: stage, maximumCaptureBytes: 10, limitBytes: 303_000_010)
        let stageURL = root.appendingPathComponent("preparation-staging/" + stage.uuidString)
        try FileManager.default.createDirectory(at: stageURL, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 10).write(to: stageURL.appendingPathComponent("retained"))
        let moved = root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: moved) }
        try FileManager.default.moveItem(at: root, to: moved)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        XCTAssertThrowsError(try context!.take(workspace: root, projectID: project, operationID: operation, purpose: .preparation))
        context = nil
        let ledger = try WorkflowStorageAdmission(workspace: moved)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        XCTAssertEqual(try Data(contentsOf: moved.appendingPathComponent("preparation-staging/" + stage.uuidString + "/retained")).count, 10)
    }
    func testInvalidBoundsAndOverflowDoNotCreateReservations() throws {
        let root = try workspace(), project = UUID().uuidString
        for capture in [-1, WorkflowPreviewCapture.Purpose.sourceInspection.totalBytes + 1, Int.max] {
            XCTAssertThrowsError(try WorkflowCaptureStorageReservation(workspace: root, projectID: project,
                purpose: .sourceInspection, maximumCaptureBytes: capture, limitBytes: Int64.max))
        }
        for additional in [Int64(-1), Int64.max] {
            XCTAssertThrowsError(try WorkflowCaptureStorageReservation(workspace: root, projectID: project,
                purpose: .sourceInspection, maximumCaptureBytes: 10, additionalBytes: additional, limitBytes: Int64.max))
        }
        XCTAssertThrowsError(try WorkflowCaptureStorageReservation(workspace: root, projectID: "invalid",
            purpose: .sourceInspection, maximumCaptureBytes: 1, limitBytes: Int64.max))
        XCTAssertThrowsError(try WorkflowCaptureStorageReservation(workspace: root, projectID: project, operationID: "invalid",
            purpose: .sourceInspection, maximumCaptureBytes: 1, limitBytes: Int64.max))
        XCTAssertThrowsError(try WorkflowCaptureStorageReservation(workspace: root, projectID: project,
            purpose: .sourceInspection, maximumCaptureBytes: 1, limitBytes: 1_000_000))
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 0)
    }
    func testSourceInspectionCollectionBudgetIsReservedAndReleasedWithoutGrantingFileAccess() throws {
        let root = try workspace(), project = UUID().uuidString
        let ceiling = WorkflowPreviewCapture.Purpose.sourceInspection.totalBytes
        let context = try WorkflowCaptureStorageReservation(workspace: root, projectID: project,
            purpose: .sourceInspection, maximumCaptureBytes: ceiling, limitBytes: Int64(ceiling) + 1_000_000)
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, Int64(ceiling) + 1_000_000)
        context.close()
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 0)
    }

}

extension WorkflowCaptureStorageReservationTests {
    func testCapturedWorkloadReservesScratchAndRetentionAndRejectsUndersizedTransfer() throws {
        let root = try workspace(),project = UUID().uuidString,operation = UUID().uuidString
        let workload = try ExecutionWorkloadPolicy(preset:.balanced)
        let growth = workload.limit(.scratchBytes)+workload.limit(.retainedBytes)+2_000_000
        let context = try WorkflowCaptureStorageReservation(workspace:root,projectID:project,operationID:operation,
            purpose:.preparationInspection,maximumCaptureBytes:250_000_000,limitBytes:growth+251_000_000,workload:workload)
        XCTAssertEqual(try WorkflowStorageAdmission(workspace:root).snapshot().committedBytes,growth+251_000_000)
        let token = try context.take(workspace:root,projectID:project,operationID:operation,purpose:.preparationInspection,workload:workload)
        token.close()
        let small = try WorkflowCaptureStorageReservation(workspace:root,projectID:project,operationID:operation,
            purpose:.preparationInspection,maximumCaptureBytes:100,limitBytes:101_000_100)
        XCTAssertThrowsError(try small.take(workspace:root,projectID:project,operationID:operation,purpose:.preparationInspection,workload:workload))
        small.close()
        XCTAssertEqual(try WorkflowStorageAdmission(workspace:root).snapshot().committedBytes,0)
    }
}
