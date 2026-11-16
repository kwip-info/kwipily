import XCTest
@testable import KwipMLApp

final class NumericDraftGateTests: XCTestCase {
    func testRemovedFieldsCannotReinsertPendingMarkersFromLateFocusEvents() {
        var lifecycle = NumericDraftLifecycle()
        XCTAssertNil(lifecycle.report(origin: "a", pending: true))
        lifecycle.appear()
        XCTAssertEqual(lifecycle.report(origin: "a", pending: true)?.pending, true)
        let removed = lifecycle.disappear(origin: "a")
        XCTAssertEqual(removed.origin, "a"); XCTAssertFalse(removed.pending)
        XCTAssertNil(lifecycle.report(origin: "a", pending: true))
        lifecycle.appear()
        XCTAssertEqual(lifecycle.report(origin: "b", pending: false)?.origin, "b")
    }
    func testPendingInvalidAndRejectedEditsCannotBecomeCommitted() {
        var edit = NumericEditState(value: .decimal(2))
        XCTAssertFalse(edit.needsCommit)
        edit.text = "3"; XCTAssertTrue(edit.needsCommit)
        XCTAssertTrue(edit.commit(domain: .decimal(minimum: -10, maximum: 10), origin: "a", currentOrigin: "a") { _ in true })
        XCTAssertFalse(edit.needsCommit)
        edit.text = "nan"
        XCTAssertFalse(edit.commit(domain: .decimal(minimum: -10, maximum: 10), origin: "a", currentOrigin: "a") { _ in true })
        XCTAssertTrue(edit.needsCommit)
        edit.text = "4"
        XCTAssertFalse(edit.commit(domain: .decimal(minimum: -10, maximum: 10), origin: "a", currentOrigin: "a") { _ in false })
        XCTAssertTrue(edit.needsCommit)
        edit = NumericEditState(value: .decimal(3))
        XCTAssertFalse(edit.needsCommit, "Context reset discards only the uncommitted field draft.")
    }
}
