import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorLearningDraftSessionTests:XCTestCase {
    @MainActor func testEditsAndPendingTextSurviveReopenWithoutBecomingASubmission() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=TensorLearningDraftRepository(directory:root)
        let id=UUID().uuidString
        let session=TensorLearningDraftSession(repository:repository)
        XCTAssertTrue(session.create(projectID:id,operation:"transpose",mode:.guided))
        XCTAssertTrue(session.update { $0.notes="Explain P transpose"; $0.mode = .practice; $0.pendingFields["axis"]="-" })
        XCTAssertTrue(session.hasUnsavedChanges)
        XCTAssertTrue(session.save())
        XCTAssertFalse(session.hasUnsavedChanges)
        let reopened=TensorLearningDraftSession(repository:repository)
        XCTAssertTrue(reopened.open(projectID:id))
        XCTAssertEqual(reopened.draft,session.draft)
        XCTAssertThrowsError(try XCTUnwrap(reopened.draft).submission())
        let before=reopened.draft
        XCTAssertFalse(reopened.update {$0.operation="custom"})
        XCTAssertEqual(reopened.draft,before)
        XCTAssertFalse(reopened.hasUnsavedChanges)
    }
    @MainActor func testStaleSaveRetainsLocalEditsAndBlocksProjectSwitchUntilExplicitDiscard() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=TensorLearningDraftRepository(directory:root)
        let session=TensorLearningDraftSession(repository:repository),other=TensorLearningDraftSession(repository:repository)
        let first=UUID().uuidString,second=UUID().uuidString
        XCTAssertTrue(session.create(projectID:first,operation:"reshape",mode:.guided))
        XCTAssertTrue(other.open(projectID:first))
        XCTAssertTrue(other.update {$0.notes="newer disk version"});XCTAssertTrue(other.save())
        XCTAssertTrue(session.update {$0.notes="local work"})
        XCTAssertFalse(session.save());XCTAssertTrue(session.hasUnsavedChanges)
        XCTAssertFalse(session.create(projectID:second,operation:"flatten",mode:.assessment))
        XCTAssertFalse(session.open(projectID:first))
        XCTAssertEqual(session.draft?.notes,"local work")
        XCTAssertEqual(try repository.load(projectID:first).draft.notes,"newer disk version")
        session.discardLocalChanges()
        XCTAssertTrue(session.open(projectID:first))
        XCTAssertEqual(session.draft?.notes,"newer disk version")
    }
    @MainActor func testFailedOpenAndDuplicateCreatePreserveSource() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=TensorLearningDraftRepository(directory:root),id=UUID().uuidString
        let session=TensorLearningDraftSession(repository:repository)
        XCTAssertTrue(session.create(projectID:id,operation:"flatten",mode:.guided))
        let original=try repository.load(projectID:id)
        XCTAssertFalse(session.create(projectID:id,operation:"transpose",mode:.practice))
        XCTAssertEqual(try repository.load(projectID:id),original)
        let file=root.appendingPathComponent("\(id)/tensor.json")
        try Data("invalid".utf8).write(to:file)
        XCTAssertFalse(session.open(projectID:id))
        XCTAssertNil(session.draft)
        XCTAssertFalse(session.save())
        XCTAssertEqual(try Data(contentsOf:file),Data("invalid".utf8))
    }
}
