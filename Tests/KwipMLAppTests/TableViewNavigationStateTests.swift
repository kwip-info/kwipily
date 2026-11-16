import XCTest
@testable import KwipMLApp

final class TableViewNavigationStateTests: XCTestCase {
    func testFilteredSortedPageRemountRetainsItsQueryAndRealBackStack() {
        var state = TableViewNavigationState()
        let sort = TableGridSort(column: nil, descending: true)
        state.begin(offset: 0, search: "needle-end", sort: sort, pageOffset: nil, nextOffset: nil)
        state.begin(offset: 100, search: "needle-end", sort: sort, pageOffset: 0, nextOffset: 100)
        let remounted = state
        XCTAssertEqual(remounted.search, "needle-end")
        XCTAssertEqual(remounted.sort, sort)
        XCTAssertEqual(remounted.offset, 100)
        XCTAssertEqual(remounted.previousOffsets, [0])
        // Retrying a failed page keeps the known navigation history.
        state.begin(offset: 100, search: "needle-end", sort: sort, pageOffset: nil, nextOffset: nil)
        XCTAssertEqual(state, remounted)
        state.begin(offset: 0, search: "needle-end", sort: sort, pageOffset: 100, nextOffset: 200)
        XCTAssertTrue(state.previousOffsets.isEmpty)
    }
    func testNewQueryOrSortResetsHistoryAndDirectJumpNeverInventsPages() {
        var state = TableViewNavigationState()
        state.begin(offset: 100, search: "", sort: .init(), pageOffset: 0, nextOffset: 100)
        XCTAssertEqual(state.previousOffsets, [0])
        state.begin(offset: 400, search: "", sort: .init(), pageOffset: 100, nextOffset: 200)
        XCTAssertTrue(state.previousOffsets.isEmpty)
        state.begin(offset: 0, search: "match", sort: .init(column: 0, descending: true), pageOffset: 400, nextOffset: 500)
        XCTAssertEqual(state.offset, 0)
        XCTAssertEqual(state.search, "match")
        XCTAssertTrue(state.previousOffsets.isEmpty)
        state.begin(offset: 100, search: "match", sort: .init(column: 0, descending: true), pageOffset: 0, nextOffset: 100)
        state.begin(offset: 200, search: "match", sort: .init(column: 0, descending: true), pageOffset: 100, nextOffset: 200)
        state.begin(offset: 100, search: "match", sort: .init(column: 0, descending: true), pageOffset: 200, nextOffset: 300)
        XCTAssertEqual(state.previousOffsets, [0])
    }
}
