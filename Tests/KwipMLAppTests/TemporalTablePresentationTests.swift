import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TemporalTablePresentationTests: XCTestCase {
    private func metadata(_ dtype: String, zone: String = "UTC") -> JSONValue {
        var fields: [String: JSONValue] = ["version": .integer(1), "unit": .string(dtype == "date" ? "day" : "microsecond")]
        if dtype == "datetime" { fields["timezone"] = .string(zone) }
        return .object(fields)
    }
    func testCanonicalDatesAndExactMicrosecondsAcrossEpoch() {
        XCTAssertEqual(TemporalTablePresentation.format(0, dtype: "date", metadata: metadata("date")), "1970-01-01")
        XCTAssertEqual(TemporalTablePresentation.format(-719162, dtype: "date", metadata: metadata("date")), "0001-01-01")
        XCTAssertEqual(TemporalTablePresentation.format(2932896, dtype: "date", metadata: metadata("date")), "9999-12-31")
        XCTAssertEqual(TemporalTablePresentation.format(-1, dtype: "datetime", metadata: metadata("datetime")), "1969-12-31T23:59:59.999999+00:00")
        XCTAssertEqual(TemporalTablePresentation.format(253402300799999999, dtype: "datetime", metadata: metadata("datetime")), "9999-12-31T23:59:59.999999+00:00")
        XCTAssertEqual(TemporalTablePresentation.format(86_399_999_999, dtype: "time", metadata: metadata("time")), "23:59:59.999999")
        XCTAssertEqual(TemporalTablePresentation.format(0, dtype: "datetime", metadata: metadata("datetime", zone: "America/Chicago")), "1969-12-31T18:00:00.000000-06:00")
    }
    func testMalformedMetadataAndValuesDoNotRenderAsValidTemporalValues() {
        XCTAssertNil(TemporalTablePresentation.format(86_400_000_000, dtype: "time", metadata: metadata("time")))
        XCTAssertNil(TemporalTablePresentation.format(0, dtype: "date", metadata: nil))
        XCTAssertNil(TemporalTablePresentation.format(0, dtype: "datetime", metadata: metadata("datetime", zone: "Invalid/Zone")))
    }
    func testCapturedGridShowsISOAndSortsCanonicalIntegers() {
        let column: [String: Any] = ["name": "event", "dtype": "datetime", "nullable": true, "temporal": ["version": 1, "unit": "microsecond", "timezone": "UTC"]]
        let grid = CapturedDataGrid(preview: ["columns": [column], "rows": [[Int64(1)], [Int64(-1)], [NSNull()]]])
        XCTAssertEqual(grid.rows[0].cells[0].text, "1970-01-01T00:00:00.000001+00:00")
        XCTAssertEqual(grid.displayRows(query: "", sort: .init(column: 0, descending: false)).map(\.index), [1, 0, 2])
        XCTAssertEqual(grid.filteredRows("1969-12-31").map(\.index), [1])
    }
}
