import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TableEditingTests: XCTestCase {
    func testSelectionIsOrderedExactAndRejectsOutputCollisionsAtomically() throws {
        let result = try TableEditing.selection(columns: [" b ", "a"], rename: [" b ": "a", "a": " b "], available: ["a", " b "])
        XCTAssertEqual(result["columns"], .array([.string(" b "), .string("a")]))
        XCTAssertThrowsError(try TableEditing.selection(columns: ["a", "b"], rename: ["a": "b"], available: ["a", "b"]))
        XCTAssertThrowsError(try TableEditing.selection(columns: ["a"], rename: ["missing": "renamed"], available: ["a"]))
        XCTAssertThrowsError(try TableEditing.selection(columns: ["a", "a"], rename: [:], available: ["a"]))
        XCTAssertThrowsError(try TableEditing.selection(columns: [], rename: [:], available: ["a"]))
        XCTAssertThrowsError(try TableEditing.selection(columns: ["a"], rename: ["a": "   "], available: ["a"]))
    }
    func testFilterPreservesLiteralTypesAndRejectsConfusion() throws {
        XCTAssertNoThrow(try TableEditing.filter(column: "x", comparison: "lt", value: .string("2"), dtype: "string"))
        XCTAssertNoThrow(try TableEditing.filter(column: "x", comparison: "lt", value: .number(2.5), dtype: "int64"))
        XCTAssertNoThrow(try TableEditing.filter(column: "x", comparison: "gt", value: .bool(false), dtype: "bool"))
        XCTAssertThrowsError(try TableEditing.filter(column: "x", comparison: "eq", value: .bool(true), dtype: "int64"))
        XCTAssertThrowsError(try TableEditing.filter(column: "x", comparison: "lt", value: .string("2"), dtype: "float64"))
        XCTAssertThrowsError(try TableEditing.filter(column: "x", comparison: "eq", value: .number(.infinity), dtype: "float64"))
        XCTAssertThrowsError(try TableEditing.filter(column: "x", comparison: "contains", value: .string("2"), dtype: "string"))
    }
}
