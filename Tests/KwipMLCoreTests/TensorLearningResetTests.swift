import XCTest
@testable import KwipMLCore

final class TensorLearningResetTests: XCTestCase {
    func testResetClearsOnlyReviewedCompletionPointerAndRetainsImmutableAttempt() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repository = TensorLearningReceiptRepository(directory: directory)
        let first = try TensorLearningReceiptTests().fixture()
        let peer = try TensorLearningReceiptTests().fixture()
        try repository.save(first); try repository.save(peer)
        let plan = try XCTUnwrap(repository.previewReset(projectID: first.projectID))
        XCTAssertEqual(plan.attemptID, first.attempt.id)
        XCTAssertEqual(plan.objectiveID, first.attempt.objectiveID)
        try repository.applyReset(plan)
        XCTAssertNil(try repository.latest(projectID: first.projectID))
        XCTAssertEqual(try repository.load(projectID: first.projectID, attemptID: first.attempt.id).attempt, first.attempt)
        XCTAssertEqual(try repository.latest(projectID: peer.projectID)?.attempt, peer.attempt)
        XCTAssertThrowsError(try repository.applyReset(plan))
    }
    func testNewAttemptAfterPreviewRefusesResetAndPreservesLatest() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repository = TensorLearningReceiptRepository(directory: directory)
        var receipt = try TensorLearningReceiptTests().fixture()
        try repository.save(receipt)
        let plan = try XCTUnwrap(repository.previewReset(projectID: receipt.projectID))
        receipt.attempt.id = UUID().uuidString
        try repository.save(receipt)
        XCTAssertThrowsError(try repository.applyReset(plan))
        XCTAssertEqual(try repository.latest(projectID: receipt.projectID)?.attempt.id, receipt.attempt.id)
    }
}
