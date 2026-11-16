import XCTest
import KwipMLCore
@testable import KwipMLApp
final class TableInputOutputPresentationTests: XCTestCase {
    func testBoundedPreviewHasUnknownTotalAndCannotBeMaskedByOlderRun() {
        let preview = SourceInspectionResult(columns: ["sample"], rows: [["new"]], rowCount: 1,
            sourceSHA256: String(repeating: "a", count: 64),
            parsing: .init(validatedRows: 1, previewRows: 1, sampleOnly: true, options: CSVParsingConfiguration.legacyDefaults))
        let run: [String: Any] = ["columns": [["name": "old"]], "rows": [["old"]], "rowCount": 500_001]
        let presentation = TableInputOutputPresentation(run: run, runIsCurrent: true, preview: preview)
        XCTAssertEqual(presentation.label, "Preview")
        XCTAssertEqual(presentation.grid.rows.first?.cells.first?.text, "new")
        XCTAssertNil(presentation.grid.totalRows)
    }

    func testNewerCapturedRunReplacesEarlierPrefixButCannotOverrideStaleConstruction() {
        let preview = SourceInspectionResult(columns: ["value"], rows: [["earlier"]], rowCount: 1,
            sourceSHA256: String(repeating: "a", count: 64),
            parsing: .init(validatedRows: 1, previewRows: 1, sampleOnly: true, options: CSVParsingConfiguration.legacyDefaults))
        let run: [String: Any] = ["columns": [["name": "value"]], "rows": [["captured"]], "rowCount": 1000]
        let newer = TableInputOutputPresentation(run: run, runIsCurrent: true, preview: preview, preferNewerRun: true)
        XCTAssertEqual(newer.label, "Run output")
        XCTAssertEqual(newer.grid.rows.first?.cells.first?.text, "captured")
        XCTAssertEqual(newer.grid.totalRows, 1000)
        let stale = TableInputOutputPresentation(run: run, runIsCurrent: false, preview: preview, preferNewerRun: true)
        XCTAssertEqual(stale.label, "Preview")
    }

    func testOneTableChoosesCurrentEvidenceAndLabelsStaleRun() {
        let run: [String: Any] = ["columns": [["name": "run"]], "rows": [["value"]], "rowCount": 1]
        let preview = SourceInspectionResult(columns: ["preview"], rows: [["sample"]], rowCount: 1, sourceSHA256: String(repeating: "a", count: 64))
        let current = TableInputOutputPresentation(run: run, runIsCurrent: true, preview: preview)
        XCTAssertEqual(current.label, "Run output"); XCTAssertEqual(current.grid.columns.first?.name, "run")
        let refreshed = TableInputOutputPresentation(run: run, runIsCurrent: false, preview: preview)
        XCTAssertEqual(refreshed.label, "Preview"); XCTAssertEqual(refreshed.grid.columns.first?.name, "preview")
        XCTAssertEqual(TableInputOutputPresentation(run: run, runIsCurrent: false, preview: nil).label, "Out of date · run output")
        XCTAssertEqual(TableInputOutputPresentation(run: nil, runIsCurrent: false, preview: nil, retainedPreview: preview).label, "Out of date · preview")
        XCTAssertTrue(TableInputOutputPresentation(run: nil, runIsCurrent: false, preview: nil).grid.rows.isEmpty)
    }
}
