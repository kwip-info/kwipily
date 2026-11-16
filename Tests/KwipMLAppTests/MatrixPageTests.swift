import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MatrixPageTests: XCTestCase {
    func testRowMajorCoordinatesAndBoundedPages() throws {
        let tensor = LearningTensor(shape: [6, 7], values: (0..<42).map(Double.init))
        let page = try XCTUnwrap(MatrixPage(tensor: tensor, rowPage: 1, columnPage: 1))
        XCTAssertEqual(Array(page.rows), [4, 5])
        XCTAssertEqual(Array(page.columns), [4, 5, 6])
        XCTAssertEqual(page.value(row: 5, column: 6), 41)
        XCTAssertNil(page.value(row: 6, column: 0))
        XCTAssertEqual(page.rowPageCount, 2)
        XCTAssertEqual(page.columnPageCount, 2)
    }
    func testCompactEditorPagesDoNotTransposeParameters() throws {
        let page = try XCTUnwrap(MatrixPage(tensor: LearningTensor(shape: [3, 4], values: (0..<12).map(Double.init)), rowPage: 1, columnPage: 1, pageSize: 2))
        XCTAssertEqual(Array(page.rows), [2]); XCTAssertEqual(Array(page.columns), [2, 3])
        XCTAssertEqual(page.value(row: 2, column: 2), 10)
        XCTAssertEqual(page.value(row: 2, column: 3), 11)
        XCTAssertNil(MatrixPage(tensor: page.tensor, rowPage: 0, columnPage: 0, pageSize: 0))
    }
    func testInvalidTensorAndStalePageAreSafe() throws {
        XCTAssertNil(MatrixPage(tensor: LearningTensor(shape: [2, 2], values: [1]), rowPage: 0, columnPage: 0))
        XCTAssertNil(MatrixPage(tensor: LearningTensor(shape: [2], values: [1, 2]), rowPage: 0, columnPage: 0))
        let page = try XCTUnwrap(MatrixPage(tensor: LearningTensor(shape: [1, 1], values: [3]), rowPage: 99, columnPage: -1))
        XCTAssertEqual(Array(page.rows), [0]); XCTAssertEqual(Array(page.columns), [0])
        XCTAssertNil(page.value(row: -1, column: 0))
    }
}
