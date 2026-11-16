import XCTest
@testable import KwipMLApp

final class NumericEditContractTests: XCTestCase {
    func testIntegerParsingRejectsFractionsRangeOverflowAndNonfinite() {
        let domain = NumericDomain.integer(minimum: 1, maximum: 128)
        for text in ["1.5", "0", "129", "nan", "inf", "1e100", "9223372036854775808"] {
            var edit = NumericEditState(value: .integer(8)); edit.text = text
            var calls = 0
            XCTAssertFalse(edit.commit(domain: domain, origin: "node:a", currentOrigin: "node:a") { _ in calls += 1; return true }, text)
            XCTAssertEqual(calls, 0, text); XCTAssertNotNil(edit.error, text)
            XCTAssertEqual(edit.text, text)
        }
    }
    func testRejectedSetterAndChangedOriginNeverPretendToApply() {
        var edit = NumericEditState(value: .integer(8)); edit.text = "12"
        XCTAssertFalse(edit.commit(domain: .integer(minimum: 1, maximum: 128), origin: "a", currentOrigin: "a") { _ in false })
        XCTAssertTrue(edit.error?.contains("not applied") == true)
        var called = false
        XCTAssertFalse(edit.commit(domain: .integer(minimum: 1, maximum: 128), origin: "a", currentOrigin: "b") { _ in called = true; return true })
        XCTAssertFalse(called)
    }
    func testAcceptedDecimalAndIntegerUseExactTypedValues() {
        var edit = NumericEditState(value: .decimal(0)); edit.text = "-2.75"
        var received: NumericValue?
        XCTAssertTrue(edit.commit(domain: .decimal(minimum: nil, maximum: nil), origin: "a", currentOrigin: "a") { received = $0; return true })
        XCTAssertEqual(received, .decimal(-2.75)); XCTAssertNil(edit.error)
        edit.text = "4294967295"
        XCTAssertTrue(edit.commit(domain: .integer(minimum: 0, maximum: 4294967295), origin: "a", currentOrigin: "a") { received = $0; return true })
        XCTAssertEqual(received, .integer(4294967295))
    }
    func testRestoredPendingTextKeepsCommittedNumberUntilValidCommit() {
        var restored=NumericEditState(value:.decimal(0.35),pendingText:"-")
        XCTAssertEqual(restored.text,"-");XCTAssertTrue(restored.needsCommit)
        var applied:[NumericValue]=[]
        XCTAssertFalse(restored.commit(domain:.decimal(minimum:0,maximum:1),origin:"a",currentOrigin:"a",apply:{applied.append($0);return true}))
        XCTAssertTrue(applied.isEmpty)
        restored.text="0.4"
        XCTAssertTrue(restored.commit(domain:.decimal(minimum:0,maximum:1),origin:"a",currentOrigin:"a",apply:{applied.append($0);return true}))
        XCTAssertEqual(applied,[.decimal(0.4)]);XCTAssertFalse(restored.needsCommit)
        XCTAssertFalse(NumericEditState(value:.decimal(0.35)).needsCommit)
    }

}
