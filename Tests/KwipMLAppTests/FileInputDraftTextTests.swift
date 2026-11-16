import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FileInputDraftTextTests: XCTestCase {
    func testReaderTextRoundTripRetainsLiteralBackslashesAndNewlines() {
        for text in ["\n\r\t", "C:\\notes\\file", "\\n", "end\\", "α,β"] {
            XCTAssertEqual(FileInputDraftText.value(FileInputDraftText.display(.string(text)), kind: "string"), .string(text))
        }
    }
    func testMalformedNumericDraftCannotSilentlyBecomeADefaultOrDropPage() throws {
        XCTAssertEqual(FileInputDraftText.value("wrong", kind: "integer"), .string("wrong"))
        let malformed = FileInputDraftText.value("1, wrong, 2", kind: "integerList")
        XCTAssertThrowsError(try FileInputConfiguration.validateReaderOptions("pdf", ["pages": malformed]))
        XCTAssertEqual(FileInputDraftText.value("", kind: "integerList"), .array([]))
        XCTAssertEqual(FileInputDraftText.value("2, 1", kind: "integerList"), .array([.integer(2), .integer(1)]))
    }
    func testFileContentPreviewDoesNotImplyDecodedImageEvidence() {
        let preview = SourceInspectionResult(columns: ["name"], rows: [["photo.png"]], rowCount: 1, sourceSHA256: String(repeating: "a", count: 64), collection: .init(complete: false, selectedFiles: 1, observedFiles: 9, sourceMode: "directory", readMode: "content", imagePixelsIncluded: false))
        let presentation = TableInputOutputPresentation(run: nil, runIsCurrent: false, preview: preview)
        XCTAssertEqual(presentation.label, "Preview · images are metadata")
        XCTAssertEqual(presentation.grid.rows.first?.cells.first?.text, "photo.png")
    }
}
