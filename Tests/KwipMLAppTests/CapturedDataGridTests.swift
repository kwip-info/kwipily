import XCTest
@testable import KwipMLApp

final class CapturedDataGridTests: XCTestCase {
    func testProjectionUsesCapturedSchemaAndEightRowBudgetWithoutMixingPartitions() {
        let preview: [String: Any] = ["partition": "training", "rowCount": 200,
            "columns": [["name": "x", "displayLabel": "Feature", "dtype": "float64", "nullable": true]],
            "rows": (0..<12).map { [$0] }]
        let grid = CapturedDataGrid(preview: preview)
        XCTAssertEqual(grid.partition, "training")
        XCTAssertEqual(grid.totalRows, 200)
        XCTAssertEqual(grid.rows.count, 8)
        XCTAssertEqual(grid.columns.first?.label, "Feature")
        XCTAssertEqual(grid.columns.first?.dtype, "float64")
        XCTAssertEqual(grid.columns.first?.nullability, "nullable")
    }
    func testNullMissingAndBooleanHaveDistinctCellPresentation() {
        let grid = CapturedDataGrid(preview: ["columns": [["name": "a"], ["name": "b"], ["name": "c"]], "rows": [[NSNull(), true]]])
        XCTAssertEqual(grid.rows.first?.cells.map(\.text), ["null", "true", "not captured"])
        XCTAssertEqual(grid.columns.first?.nullability, "not recorded")
    }
    func testJSONValuesRemainReadableAndUnmodeledExtraValuesAreNotSilentlyLost() {
        let grid = CapturedDataGrid(preview: ["columns": [["name": "a"]], "rows": [[["b": 2, "a": 1], "extra"]]])
        XCTAssertEqual(grid.rows.first?.cells.first?.text, "{\"a\":1,\"b\":2}")
        XCTAssertTrue(grid.hasUnmappedValues)
        XCTAssertEqual(CapturedDataGrid.cell(Double.infinity).text, "∞")
        XCTAssertEqual(CapturedDataGrid.cell(Double.nan).text, "NaN")
        XCTAssertEqual(CapturedDataGrid.cell("").text, "\"\"")
    }
    func testSearchOnlyFiltersRetainedSampleAndPreservesOriginalRowNumbers() {
        let grid = CapturedDataGrid(preview: ["columns": [["name": "a"]], "rows": [["Alpha"], ["Beta"], ["alphabet"]]])
        XCTAssertEqual(grid.filteredRows("ALPHA").map(\.index), [0, 2])
        XCTAssertEqual(grid.filteredRows("").count, 3)
    }
    func testTypedSampleSortingKeepsNullLastAndOriginalRowIdentity() {
        let grid = CapturedDataGrid(preview: ["columns": [["name":"n","dtype":"int64"]], "rows": [[10],[2],[NSNull()],[2]]])
        XCTAssertEqual(grid.displayRows(query: "", sort: .init(column: 0, descending: false)).map(\.index), [1,3,0,2])
        XCTAssertEqual(grid.displayRows(query: "", sort: .init(column: 0, descending: true)).map(\.index), [0,1,3,2])
        XCTAssertEqual(grid.displayRows(query: "2", sort: .init(column: 0, descending: true)).map(\.index), [1,3])
    }

}
