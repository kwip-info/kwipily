import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationStorageReservationTests: XCTestCase {
    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
    func testFixedPeakScopesAndOneShotTransfer() throws {
        let root = try root(), foreign = try self.root(), project = UUID().uuidString, evaluation = UUID(), stage = UUID()
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let context = try WorkflowEvaluationStorageReservation(workspace: root, projectID: project, evaluationID: evaluation, stageID: stage, limitBytes: 1_227_512_000)
        XCTAssertEqual(context.evaluationID, evaluation); XCTAssertEqual(context.stageID, stage)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_227_512_000)
        let scope = try WorkflowStorageAdmission.Scope(namespace: .evaluationStaging, components: [stage.uuidString])
        XCTAssertThrowsError(try ledger.reserve(scopes: [scope], peakBytes: 1, limitBytes: Int64.max))
        XCTAssertThrowsError(try context.take(workspace: foreign, projectID: project, evaluationID: evaluation))
        XCTAssertThrowsError(try context.take(workspace: root, projectID: UUID().uuidString, evaluationID: evaluation))
        XCTAssertThrowsError(try context.take(workspace: root, projectID: project, evaluationID: UUID()))
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_227_512_000)
        let token = try context.take(workspace: root, projectID: project, evaluationID: evaluation)
        XCTAssertThrowsError(try context.take(workspace: root, projectID: project, evaluationID: evaluation))
        context.close(); context.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 1_227_512_000)
        token.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 0)
    }
    func testInsufficientAdmissionAndAbandonmentRetainExistingBytes() throws {
        let root = try root(), project = UUID().uuidString, evaluation = UUID(), stage = UUID()
        XCTAssertThrowsError(try WorkflowEvaluationStorageReservation(workspace: root, projectID: project, evaluationID: evaluation, stageID: stage, limitBytes: 1_227_511_999))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("evaluation-staging").path))
        var context: WorkflowEvaluationStorageReservation? = try .init(workspace: root, projectID: project, evaluationID: evaluation, stageID: stage, limitBytes: 1_227_512_000)
        XCTAssertEqual(context?.stageID, stage)
        let directory = root.appendingPathComponent("evaluation-staging/" + stage.uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 9).write(to: directory.appendingPathComponent("retained"))
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 1_227_512_000)
        context = nil
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 9)
        XCTAssertEqual(try Data(contentsOf: directory.appendingPathComponent("retained")).count, 9)
    }
}
