import XCTest
@testable import KwipMLApp

final class Data02SourcePreviewTests: XCTestCase {
    func testPresentationUsesVerifiedRowsAsTextWithoutInferringTypes() {
        let evidence=SourceInspectionResult(columns:[" a","flag"],rows:[["001","true"],["","false"]],rowCount:25,sourceSHA256:String(repeating:"a",count:64))
        let preview=DataSourcePreviewPresentation(result:evidence)
        XCTAssertEqual(preview.grid.columns.map(\.name),[" a","flag"])
        XCTAssertEqual(preview.grid.columns.map(\.dtype),["string","string"])
        XCTAssertEqual(preview.grid.rows[0].cells.map(\.text),["001","true"])
        XCTAssertEqual(preview.grid.totalRows,25)
        XCTAssertTrue(preview.summary.contains("2 of 25"))
        XCTAssertTrue(preview.interpretation.contains("header"))
        XCTAssertTrue(preview.budget.contains("10 MB"))
        XCTAssertFalse(preview.budget.contains("100,000"))
        XCTAssertTrue(preview.budget.contains("1 MiB"))
    }
    func testTypedReaderSchemaIsRecordedRatherThanReinferredFromText() {
        let evidence = SourceInspectionResult(columns:["value"],rows:[["001"]],rowCount:1,sourceSHA256:String(repeating:"a",count:64),schema:[.init(name:"value",dtype:"int64",nullable:nil)])
        let preview = DataSourcePreviewPresentation(result:evidence)
        XCTAssertEqual(preview.grid.columns.first?.dtype,"int64")
        XCTAssertEqual(preview.grid.columns.first?.nullability,"not recorded")
        XCTAssertEqual(preview.grid.rows.first?.cells.first?.text,"001")
        XCTAssertTrue(preview.interpretation.contains("does not infer"))
    }

    func testPreviewDistinguishesExplicitNullFromEmptyText() {
        let evidence = SourceInspectionResult(columns: ["value"], rows: [[""], [""]], rowCount: 2, sourceSHA256: String(repeating: "a", count: 64), nullCells: [[0, 0]])
        let preview = DataSourcePreviewPresentation(result: evidence)
        XCTAssertTrue(preview.grid.rows[0].cells[0].isAbsent)
        XCTAssertEqual(preview.grid.rows[0].cells[0].text, "null")
        XCTAssertFalse(preview.grid.rows[1].cells[0].isAbsent)
        XCTAssertEqual(preview.grid.rows[1].cells[0].text, "\"\"")
    }
    func testNoObservationIsExplicitlyUnavailable() {
        let preview=DataSourcePreviewPresentation(result:nil)
        XCTAssertTrue(preview.grid.columns.isEmpty)
        XCTAssertTrue(preview.summary.contains("not inspected"))
        XCTAssertNil(preview.grid.totalRows)
    }
}
