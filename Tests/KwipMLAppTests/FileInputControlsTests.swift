import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FileInputControlsTests: XCTestCase {
    func testConfirmingTheCurrentPageModePreservesSelectionAndUnfinishedDraft() {
        let pages: JSONValue = .array([.integer(1), .integer(4)])
        XCTAssertEqual(FileInputPageSelection.selecting("selected", current: pages), pages)
        XCTAssertEqual(FileInputPageSelection.selecting("selected", current: .string("1–")), .string("1–"))
        XCTAssertEqual(FileInputPageSelection.selecting("all", current: pages), .array([]))
        XCTAssertEqual(FileInputPageSelection.selecting("selected", current: .array([])), .string(""))
    }
    func testExistingSingleFileBindingSelectsItsReaderWithoutTreatingDirectoryNamesAsFormats() {
        XCTAssertEqual(FileInputReaderSelection.reader(sourceMode: "file", sourceName: "photo.PNG"), "image")
        XCTAssertEqual(FileInputReaderSelection.reader(sourceMode: "file", sourceName: "report.pdf"), "pdf")
        XCTAssertEqual(FileInputReaderSelection.reader(sourceMode: "file", sourceName: "notes.csv"), "text")
        XCTAssertNil(FileInputReaderSelection.reader(sourceMode: "directory", sourceName: "folder.pdf"))
        XCTAssertNil(FileInputReaderSelection.reader(sourceMode: "file", sourceName: "unknown.bin"))
    }
    func testByteUnitsPreserveExactLimitAndPermitFractionalMiBWithoutRounding() throws {
        XCTAssertEqual(FileInputSizeUnit.preferred(.integer(67_108_864)), .mebibytes)
        XCTAssertEqual(FileInputSizeUnit.mebibytes.display(.integer(67_108_864)), "64")
        XCTAssertEqual(FileInputSizeUnit.mebibytes.parse("0.5"), .integer(524_288))
        XCTAssertEqual(FileInputSizeUnit.kibibytes.parse("512"), .integer(524_288))
        XCTAssertEqual(FileInputSizeUnit.bytes.parse("524288"), .integer(524_288))
        XCTAssertEqual(FileInputSizeUnit.mebibytes.parse("0.1"), .string("0.1"), "A fractional byte must never be rounded silently.")
        for raw in ["", "NaN", "inf", "9999999999999999999999999", "-"] {
            XCTAssertEqual(FileInputSizeUnit.mebibytes.parse(raw), .string(raw))
        }
        XCTAssertThrowsError(try FileInputConfiguration.validate(["maxFileBytes": FileInputSizeUnit.mebibytes.parse("65")]))
    }
    func testDisplayedInclusiveFrameNumbersKeepStoredDecoderIndexesUnchanged() throws {
        XCTAssertEqual(FileInputFrameNumber.display(.integer(0)), "1")
        XCTAssertEqual(FileInputFrameNumber.parse("3"), .integer(2))
        XCTAssertEqual(FileInputFrameNumber.parse(" 2 "), .integer(1))
        XCTAssertEqual(FileInputFrameNumber.parse("0"), .string("0"))
        XCTAssertEqual(FileInputFrameNumber.parse("1.5"), .string("1.5"))
        let options = try FileInputConfiguration.validateReaderOptions("image", ["frames": .string("range"), "frameStart": FileInputFrameNumber.parse("2"), "frameEnd": FileInputFrameNumber.parse("4")])
        XCTAssertEqual(options["frameStart"], .integer(1)); XCTAssertEqual(options["frameEnd"], .integer(3))
        XCTAssertThrowsError(try FileInputConfiguration.validateReaderOptions("image", ["frames": .string("range"), "frameStart": FileInputFrameNumber.parse("4"), "frameEnd": FileInputFrameNumber.parse("2")]))
    }
    func testPageRangesAreOrderedOneBasedUniqueAndBoundedBeforeExpansion() throws {
        let parsed = FileInputPageSelection.parse("4, 1–3, 8-9")
        XCTAssertEqual(parsed, .array([4,1,2,3,8,9].map { .integer(Int64($0)) }))
        XCTAssertEqual(FileInputPageSelection.display(parsed), "4, 1–3, 8–9")
        XCTAssertEqual(FileInputPageSelection.parse(FileInputPageSelection.display(parsed)), parsed)
        XCTAssertEqual(try FileInputConfiguration.validateReaderOptions("pdf", ["pages": parsed])["pages"], parsed)
        for raw in ["", "0", "1,1", "2-1", "1,", "1-1000000", "1000001", "1-999, 999", "1-1000,1001"] {
            XCTAssertEqual(FileInputPageSelection.parse(raw), .string(raw))
        }
        if case .array(let pages) = FileInputPageSelection.parse("1-1000") { XCTAssertEqual(pages.count, 1000); XCTAssertEqual(FileInputPageSelection.display(.array(pages)), "1–1000") }
        else { XCTFail("The exact bounded selection must remain usable.") }
        XCTAssertEqual(FileInputPageSelection.display(.array([])), "")
    }
}
