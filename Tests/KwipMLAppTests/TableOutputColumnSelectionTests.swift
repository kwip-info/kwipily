import XCTest
@testable import KwipMLApp

final class TableOutputColumnSelectionTests:XCTestCase {
    func testAllModeMaterializesOnlyAnIntentionalRemoveOrReorder() {
        let names=["id","quantity","price","amount"]
        XCTAssertEqual(TableOutputColumnSelection.effective([],available:names),names)
        XCTAssertEqual(TableOutputColumnSelection.removable("price",configured:[],available:names),["id","quantity","amount"])
        XCTAssertEqual(TableOutputColumnSelection.moving("amount",to:1,configured:[],available:names),["id","amount","quantity","price"])
        XCTAssertNil(TableOutputColumnSelection.moving("id",to:0,configured:[],available:names))
    }
    func testLastColumnNeverBecomesEmptySentinelAndAccidentallyIncludesEverything() {
        XCTAssertNil(TableOutputColumnSelection.removable("id",configured:["id"],available:["id","private"]))
        XCTAssertNil(TableOutputColumnSelection.removable("missing",configured:["id","price"],available:[]))
        XCTAssertNil(TableOutputColumnSelection.moving("id",to:2,configured:["id","price"],available:[]))
    }
    func testExplicitOrderAndUnresolvedNamesArePreservedForRepair() {
        XCTAssertEqual(TableOutputColumnSelection.effective(["old","id"],available:["id","new"]),["old","id"])
        XCTAssertEqual(TableOutputColumnSelection.moving("old",to:1,configured:["old","id"],available:["id","new"]),["id","old"])
    }
}
