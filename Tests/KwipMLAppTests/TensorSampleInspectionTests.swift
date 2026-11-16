import XCTest
@testable import KwipMLApp

final class TensorSampleInspectionTests: XCTestCase {
    func testCoordinatesUseRowMajorShapeWithoutSemanticAxisGuesses() {
        let sample = TensorSampleInspection(["shape": [2, 3, 2], "dtype": "int64", "values": Array(0..<8)])
        XCTAssertEqual(sample.rows[7].coordinates, [1, 0, 1])
        XCTAssertEqual(sample.axes, ["Axis 0", "Axis 1", "Axis 2"])
        XCTAssertEqual(sample.totalCount, 12)
        XCTAssertTrue(sample.isSampled)
        XCTAssertEqual(sample.rows[7].value, "7")
    }
    func testScalarVectorAndExactLargeIntegers() {
        let scalar = TensorSampleInspection(["shape": [Int](), "dtype": "int64", "values": [Int64(9007199254740993)]])
        XCTAssertEqual(scalar.rows.first?.coordinates, [])
        XCTAssertEqual(scalar.rows.first?.value, "9007199254740993")
        XCTAssertFalse(scalar.isSampled)
        XCTAssertEqual(TensorSampleInspection(["shape": [3], "dtype": "float32", "values": [1.25]]).rows.first?.coordinates, [0])
    }
    func testMissingNullNonfiniteAndUnsupportedStayDistinct() {
        XCTAssertEqual(TensorSampleInspection(["dtype": "float32"]).availability, "Values unavailable")
        let sample = TensorSampleInspection(["shape": [4], "dtype": "float64", "values": [NSNull(), Double.infinity, Double.nan, "bad"]])
        XCTAssertEqual(sample.rows.map(\.state), ["Null", "Nonfinite", "Nonfinite", "Invalid"])
        XCTAssertEqual(TensorSampleInspection(["dtype": "mystery", "values": [1]]).rows.first?.state, "Unsupported type")
        XCTAssertNil(TensorSampleInspection(["values": [1]]).rows.first?.coordinates)
    }
    func testPolicyAndMalformedShapeNeverInventCoordinates() {
        XCTAssertTrue(TensorSampleInspection(["dtype": "float32", "values": [], "captureStatus": "omittedByPolicy"]).rows.isEmpty)
        XCTAssertEqual(TensorSampleInspection(["dtype": "float32", "values": [1], "captureStatus": "omittedByPolicy"]).availability, "Invalid capture: values contradict policy")
        for shape in [[-1], [Int.max, 2], [0], [1]] {
            let sample = TensorSampleInspection(["shape": shape, "dtype": "int64", "values": [1, 2]])
            XCTAssertTrue(sample.rows.allSatisfy { $0.coordinates == nil })
        }
        XCTAssertEqual(TensorSampleInspection(["values": [1], "captureStatus": "mystery"]).availability, "Invalid capture status")
    }
    func testReportsActualRetainedCountWithoutInventingAnEightScalarCapture() {
        let sample = TensorSampleInspection(["shape": [20], "dtype": "int64", "values": Array(0..<20)])
        XCTAssertEqual(sample.retainedCount, 20)
        XCTAssertEqual(sample.rows.count, 20)
        XCTAssertEqual(sample.rows.last?.coordinates, [19])
        XCTAssertFalse(sample.isSampled)
    }
    func testMatrixRefusesMalformedTypedAndNonfiniteCapturedScalars() {
        for value: Any in [Double.infinity, Double.nan, "not a number", NSNull(), true] {
            let sample = TensorSampleInspection(["shape": [1, 1], "dtype": "float64", "values": [value]])
            XCTAssertFalse(sample.canShowMatrix, "A numeric matrix cannot present invalid or nonfinite scalar evidence")
        }
        XCTAssertFalse(TensorSampleInspection(["shape": [1, 1], "dtype": "unknown", "values": [1]]).canShowMatrix)
    }

    func testMatrixRequiresCompleteBoundedTwoDimensionalCapture() {
        let full = TensorSampleInspection(["shape": [2, 3], "dtype": "float32", "values": [1, 2, 3, 4, 5, 6]])
        XCTAssertTrue(full.canShowMatrix)
        XCTAssertEqual(full.rows[5].coordinates, [1, 2])
        XCTAssertFalse(TensorSampleInspection(["shape": [32, 48], "dtype": "float32", "values": Array(0..<8)]).canShowMatrix)
        XCTAssertFalse(TensorSampleInspection(["shape": [1, 300], "dtype": "int64", "values": Array(0..<300)]).canShowMatrix)
        XCTAssertFalse(TensorSampleInspection(["shape": [6], "dtype": "int64", "values": Array(0..<6)]).canShowMatrix)
    }
}

extension TensorSampleInspectionTests {
    func testRankThreeMatrixSlicesUseActualCapturedCoordinatesAndAxisOrder() throws {
        let sample = TensorSampleInspection(["shape": [2, 3, 2], "dtype": "int64", "values": Array(0..<8)])
        let first = try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 0])
        XCTAssertEqual(first.rowIndices, [0, 1, 2]); XCTAssertEqual(first.columnIndices, [0, 1])
        XCTAssertEqual(first.cells.map { $0.map(\.value) }, [["0", "1"], ["2", "3"], ["4", "5"]])
        XCTAssertEqual(first.cells[2][1].coordinates, [0, 2, 1])
        XCTAssertFalse(first.isPartialWindow)
        let transposed = try sample.matrix(rowAxis: 2, columnAxis: 1, fixedIndices: [0: 0])
        XCTAssertEqual(transposed.cells.map { $0.map(\.value) }, [["0", "2", "4"], ["1", "3", "5"]])
        let second = try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 1])
        XCTAssertEqual(second.cells.map { $0.map(\.value) }, [["6", "7"]])
        XCTAssertTrue(second.isPartialWindow)
        XCTAssertEqual(sample.defaultMatrix?.cells.count, 3)
        XCTAssertTrue(sample.isSampled, "A complete first slice does not claim the whole tensor was captured")
    }
    func testMissingSliceInvalidAxesAndWindowBoundsRefuseRatherThanInventValues() throws {
        let sample = TensorSampleInspection(["shape": [4, 2, 2], "dtype": "float32", "values": [1, 2, 3, 4]])
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 1]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 1, fixedIndices: [0: 0, 2: 0]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: -1, columnAxis: 2, fixedIndices: [0: 0]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 3, fixedIndices: [0: 0]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [:]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 0, 1: 0]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: -1]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 4]))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 0], maximumRows: 256, maximumColumns: 256))
        XCTAssertThrowsError(try sample.matrix(rowAxis: 1, columnAxis: 2, fixedIndices: [0: 0], maximumRows: 0))
    }
    func testHugeTensorShapeOnlyInspectsBoundedRetainedWindowAndKeepsExactIntegers() throws {
        let values: [Int64] = [9007199254740993] + Array(repeating: 7, count: 299)
        let sample = TensorSampleInspection(["shape": [1_000_000, 1_000_000], "dtype": "int64", "values": values])
        XCTAssertEqual(sample.rows.count, 256)
        XCTAssertEqual(sample.retainedCount, 300)
        XCTAssertEqual(sample.totalCount, 1_000_000_000_000)
        let matrix = try sample.matrix(rowAxis: 0, columnAxis: 1, fixedIndices: [:])
        XCTAssertEqual(matrix.cells.count, 1)
        XCTAssertEqual(matrix.columnIndices.count, 16)
        XCTAssertEqual(matrix.cells[0][0].value, "9007199254740993")
        XCTAssertTrue(matrix.isPartialWindow)
        XCTAssertFalse(sample.canShowMatrix, "Legacy complete-two-dimensional API stays conservative")
    }
    func testMalformedShapePolicyAndInvalidTypesNeverOfferMatrixEvidence() {
        let invalidShapes:[Any] = [[true, 2] as [Any], [1.5, 2], [-1, 2], [Int.max, 2], [1, 0]]
        for shape in invalidShapes {
            let sample = TensorSampleInspection(["shape": shape, "dtype": "float64", "values": [1]])
            XCTAssertNil(sample.defaultMatrix)
        }
        let invalid = TensorSampleInspection(["shape": [1, 1], "dtype": "float32", "values": [Double.greatestFiniteMagnitude]])
        XCTAssertEqual(invalid.rows.first?.state, "Invalid")
        XCTAssertNil(invalid.defaultMatrix)
        let omitted = TensorSampleInspection(["shape": [1, 1], "dtype": "float64", "values": [1], "captureStatus": "omittedByPolicy"])
        XCTAssertNil(omitted.defaultMatrix)
        XCTAssertThrowsError(try omitted.matrix(rowAxis: 0, columnAxis: 1, fixedIndices: [:]))
        let boolean = TensorSampleInspection(["shape": [1, 2], "dtype": "bool", "values": [true, false]])
        XCTAssertEqual(boolean.defaultMatrix?.cells[0].map(\.value), ["true", "false"])
        XCTAssertNil(boolean.rows.first?.numericValue)
    }
}
