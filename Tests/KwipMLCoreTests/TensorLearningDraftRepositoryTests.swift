import XCTest
@testable import KwipMLCore

final class TensorLearningDraftRepositoryTests:XCTestCase {
    func testReopenStaleSaveAndInvalidSavePreserveLastDraft()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=TensorLearningDraftRepository(directory:root)
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:"transpose",mode:.practice)
        draft.notes="unfinished";draft.pendingFields=["axis":"-"]
        let first=try store.save(draft,expectedRevision:nil)
        let reopened=try store.load(projectID:draft.projectID)
        XCTAssertEqual(reopened.draft,draft);XCTAssertEqual(reopened.revision,first)
        draft.notes="updated"
        let second=try store.save(draft,expectedRevision:first)
        XCTAssertNotEqual(first,second)
        XCTAssertThrowsError(try store.save(reopened.draft,expectedRevision:first))
        draft.operation="code"
        XCTAssertThrowsError(try store.save(draft,expectedRevision:second))
        XCTAssertEqual(try store.load(projectID:draft.projectID).draft.notes,"updated")
        XCTAssertThrowsError(try store.load(projectID:"../outside"))
    }
    func testMismatchedIdentityAndSymlinkAreNotLoaded()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=TensorLearningDraftRepository(directory:root)
        let draft=TensorLearningDraft(projectID:UUID().uuidString,operation:"reshape",mode:.guided)
        _=try store.save(draft,expectedRevision:nil)
        let file=root.appendingPathComponent(draft.projectID).appendingPathComponent("tensor.json")
        var wrong=draft;wrong.projectID=UUID().uuidString
        try wrong.encoded().write(to:file)
        XCTAssertThrowsError(try store.load(projectID:draft.projectID))
        try FileManager.default.removeItem(at:file)
        let other=root.appendingPathComponent("other.json");try draft.encoded().write(to:other)
        try FileManager.default.createSymbolicLink(at:file,withDestinationURL:other)
        XCTAssertThrowsError(try store.load(projectID:draft.projectID))
        XCTAssertThrowsError(try store.save(draft,expectedRevision:nil))
    }
}
