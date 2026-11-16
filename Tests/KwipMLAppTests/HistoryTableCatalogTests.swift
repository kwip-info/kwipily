import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class HistoryTableCatalogTests: XCTestCase {
    private let project = "project"
    private func training(_ suffix: Int, date: String? = nil, status: WorkflowExecutionPhase? = .completed, verification: WorkflowHistoryVerification = .recordedV2) -> WorkflowHistoryEntry {
        .init(reference: .init(projectID: project, directoryID: UUID(uuidString: String(format: "00000000-0000-0000-0000-%012d", suffix))!),
              status: status, runID: "recorded-run-\(suffix)", finishedAt: date, verification: verification, message: nil)
    }
    private func group(_ statuses: [WorkflowExperimentVariantStatus], date: Double? = nil) -> WorkflowExperimentGroupRecord {
        .init(schemaVersion: 1, createdAtUnixSeconds: date, groupID: "recorded-group", projectID: project,
              train: .init(domain: .control, localID: "train"), sourceRequestIdentity: "source", inputsManifestSHA256: "inputs", authoringSHA256: "authoring", revision: 0, stopRequested: false,
              variants: statuses.enumerated().map { .init(ordinal: $0.offset + 1, requestIdentity: "request-\($0.offset)", status: $0.element) })
    }
    func testAllKindsHaveStableTargetsAndHonestRecordedOrVerifiedEvidenceWithoutArchiveReads() {
        let evaluation = WorkflowEvaluationHistoryEntry(reference: .init(projectID: project, evaluationID: UUID()), status: .completed, verification: .recordedUnverified, message: "select to verify")
        let backend = BackendFitHistoryStore.Entry(id: "backend-operation", backendID: "sklearn", completedAt: Date(timeIntervalSince1970: 100), evaluation: [:],
            artifactDirectory: URL(fileURLWithPath: "/nonexistent-must-not-open"), manifestSHA256: "recorded", workerRequest: Data(),
            document: .init(projectID: project, name: "My recorded logistic flow"), artifacts: [])
        let run = training(1)
        let rows = HistoryTableCatalog.rows(training: [run], evaluation: [evaluation], experiments: [group([.completed])], backends: [backend])
        XCTAssertEqual(rows.count, 4)
        XCTAssertEqual(Set(rows.map(\.id)).count, 4)
        XCTAssertEqual(rows.first { $0.kind == .training }?.target, .training(run.reference))
        XCTAssertEqual(rows.first { $0.kind == .evaluation }?.target, .evaluation(evaluation.reference))
        XCTAssertEqual(rows.first { $0.kind == .experiments }?.target, .experiment("recorded-group"))
        XCTAssertEqual(rows.first { $0.kind == .backend }?.target, .backend("backend-operation"))
        XCTAssertEqual(rows.first { $0.kind == .backend }?.title, "My recorded logistic flow")
        for row in rows where row.kind != .backend { XCTAssertFalse(row.evidenceLabel.lowercased().contains("verified at scan")) }
        XCTAssertTrue(rows.first { $0.kind == .backend }!.evidenceLabel.contains("Verified at scan"))
        XCTAssertTrue(rows.first { $0.kind == .experiments }!.evidenceLabel.contains("select to verify"))
    }
    func testIncompleteUnavailableAndInUseDoNotInferTerminalCompletion() {
        let evaluation = WorkflowEvaluationHistoryEntry(reference: .init(projectID: project, evaluationID: UUID()), status: nil, verification: .inUse, message: "lease busy")
        let rows = HistoryTableCatalog.rows(training: [training(1, status: nil, verification: .incomplete), training(2, verification: .unavailable)], evaluation: [evaluation])
        XCTAssertEqual(rows.first { $0.target == .training(training(1).reference) }?.status, .incomplete)
        XCTAssertEqual(rows.first { $0.target == .training(training(2).reference) }?.status, .unavailable)
        XCTAssertEqual(rows.first { $0.kind == .evaluation }?.statusLabel, "In use")
        XCTAssertEqual(HistoryTableCatalog.filtered(rows, status: .completed).count, 0)
    }
    func testExperimentStatusUsesRecordedMembershipAndNeverClaimsLiveRunning() {
        for (states, expected) in [([WorkflowExperimentVariantStatus.completed, .completed], HistoryTableCatalog.Status.completed),
                                   ([.completed, .failed], .failed), ([.completed, .cancelled], .cancelled),
                                   ([.interrupted], .incomplete), ([.pending, .running], .recorded), ([], .incomplete)] {
            let row = HistoryTableCatalog.rows(experiments: [group(states)])[0]
            XCTAssertEqual(row.status, expected)
            XCTAssertTrue(row.evidenceLabel.contains("Recorded ledger"))
            XCTAssertNotEqual(row.statusLabel, "Running", "An archived claim cannot imply a live process")
        }
    }
    func testDateParsingSortingUnknownLastAndStableTieBreakBothDirections() {
        let entries = [training(4), training(3, date: "not a timestamp"), training(2, date: "2026-10-07T20:00:00Z"), training(1, date: "2026-10-07T20:00:00.000Z"), training(5, date: "2025-01-01T00:00:00Z")]
        let rows = HistoryTableCatalog.rows(training: entries)
        let newest = HistoryTableCatalog.filtered(rows, sort: .newest)
        let oldest = HistoryTableCatalog.filtered(rows, sort: .oldest)
        XCTAssertEqual(newest.prefix(3).compactMap(\.recordedAt).count, 3)
        XCTAssertEqual(oldest.prefix(3).compactMap(\.recordedAt).count, 3)
        XCTAssertEqual(newest.suffix(2).map(\.id), oldest.suffix(2).map(\.id))
        XCTAssertEqual(newest.prefix(2).map(\.id), newest.prefix(2).map(\.id).sorted())
        XCTAssertLessThan(oldest[0].recordedAt!, oldest[1].recordedAt!)
        XCTAssertEqual(HistoryTableCatalog.filtered(Array(rows.reversed()), sort: .newest).map(\.id), newest.map(\.id))
        XCTAssertFalse(rows.first { $0.recordedAt == nil }!.timeLabel.isEmpty)
    }
    func testCombinedFiltersSearchAcrossIdentityEvidenceTypeAndStatus() {
        let rows = HistoryTableCatalog.rows(training: [training(1, status: .failed), training(2, verification: .legacyV1)], experiments: [group([.completed])])
        XCTAssertEqual(HistoryTableCatalog.filtered(rows, kind: .training, status: .failed, query: " FAILED ").count, 1)
        XCTAssertEqual(HistoryTableCatalog.filtered(rows, query: "recorded-run-1").count, 1)
        XCTAssertEqual(HistoryTableCatalog.filtered(rows, query: "legacy").count, 1)
        XCTAssertEqual(HistoryTableCatalog.filtered(rows, kind: .experiments, query: "Training").count, 0)
        XCTAssertTrue(HistoryTableCatalog.filtered(rows, query: "not present").isEmpty)
        XCTAssertEqual(HistoryTableCatalog.filtered(rows, query: " ").count, 3)
    }
    func testTimestampUnavailableDoesNotCreateFakeEpochOrCurrentTime() {
        let rows = HistoryTableCatalog.rows(training: [training(1)], experiments: [group([.pending], date: .nan)])
        XCTAssertTrue(rows.allSatisfy { $0.recordedAt == nil })
        XCTAssertTrue(rows.allSatisfy { $0.timeLabel == "Time not recorded" })
    }
}
