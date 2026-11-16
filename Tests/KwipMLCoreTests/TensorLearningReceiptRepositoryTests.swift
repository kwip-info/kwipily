import XCTest
@testable import KwipMLCore

final class TensorLearningReceiptRepositoryTests:XCTestCase {
    func testImmutableAttemptAndLatestPointerReopen()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=TensorLearningReceiptRepository(directory:root)
        let first=try TensorLearningReceiptTests().fixture()
        XCTAssertNil(try repository.latest(projectID:first.projectID))
        try repository.save(first);try repository.save(first)
        XCTAssertEqual(try repository.latest(projectID:first.projectID)?.attempt,first.attempt)
        var second=first;second.attempt.id=UUID().uuidString;second.attempt.timestamp=Date().addingTimeInterval(1)
        try repository.save(second)
        let reopened=TensorLearningReceiptRepository(directory:root)
        XCTAssertEqual(try reopened.latest(projectID:first.projectID)?.attempt.id,second.attempt.id)
        XCTAssertEqual(try reopened.load(projectID:first.projectID,attemptID:first.attempt.id).attempt,first.attempt)
        var conflict=first;conflict.attempt.mode = .practice
        XCTAssertThrowsError(try repository.save(conflict))
        XCTAssertEqual(try repository.latest(projectID:first.projectID)?.attempt.id,second.attempt.id)
    }
    func testCorruptOrForeignPointerAndSymlinkAreRejectedWithoutReplacement()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=TensorLearningReceiptRepository(directory:root),receipt=try TensorLearningReceiptTests().fixture()
        try repository.save(receipt)
        let pointer=root.appendingPathComponent("\(receipt.projectID)/tensor-receipts/latest.json")
        try Data("invalid".utf8).write(to:pointer)
        XCTAssertThrowsError(try repository.latest(projectID:receipt.projectID))
        let file=root.appendingPathComponent("\(receipt.projectID)/tensor-receipts/\(receipt.attempt.id).json")
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createSymbolicLink(at:file,withDestinationURL:root.appendingPathComponent("missing"))
        XCTAssertThrowsError(try repository.load(projectID:receipt.projectID,attemptID:receipt.attempt.id))
        XCTAssertThrowsError(try repository.save(receipt))
        XCTAssertEqual(try Data(contentsOf:pointer),Data("invalid".utf8))
        XCTAssertThrowsError(try repository.latest(projectID:"../outside"))
    }
}
