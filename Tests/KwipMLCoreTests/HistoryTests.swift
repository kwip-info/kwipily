import XCTest
@testable import KwipMLCore
final class HistoryTests: XCTestCase {
    func testUndoRedoAndBranchInvalidation() {
        var history = EditHistory<Int>()
        history.record(before: 1, after: 2)
        XCTAssertEqual(history.undo(current: 2), 1)
        XCTAssertEqual(history.redo(current: 1), 2)
        XCTAssertEqual(history.undo(current: 2), 1)
        history.record(before: 1, after: 3)
        XCTAssertFalse(history.canRedo)
        XCTAssertEqual(history.undo(current: 3), 1)
    }
    func testNoOpDoesNotPolluteHistory() {
        var history = EditHistory<Int>()
        history.record(before: 1, after: 1)
        XCTAssertFalse(history.canUndo)
    }
}
