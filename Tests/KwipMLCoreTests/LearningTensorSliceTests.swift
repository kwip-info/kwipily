import XCTest
@testable import KwipMLCore

final class LearningTensorSliceTests:XCTestCase {
    func testHigherRankSlicesPreserveEveryCoordinate()throws {
        let tensor=LearningTensor(shape:[2,3,2,2],values:(0..<24).map(Double.init))
        XCTAssertEqual(tensor.matrixSliceCount,6)
        let slice=try XCTUnwrap(tensor.matrixSlice(at:4))
        XCTAssertEqual(slice.leadingCoordinates,[1,1])
        XCTAssertEqual(slice.matrix.shape,[2,2])
        XCTAssertEqual(slice.matrix.values,[16,17,18,19])
        XCTAssertNil(tensor.matrixSlice(at:6));XCTAssertNil(tensor.matrixSlice(at:-1))
        XCTAssertEqual((0..<6).flatMap {tensor.matrixSlice(at:$0)!.matrix.values},tensor.values)
    }
    func testVectorsMatricesAndMalformedTensors() {
        let vector=LearningTensor(shape:[3],values:[1,2,3])
        XCTAssertEqual(vector.matrixSlice(at:0)?.matrix.shape,[1,3])
        XCTAssertEqual(vector.matrixSlice(at:0)?.leadingCoordinates,[])
        let matrix=LearningTensor(shape:[2,2],values:[1,2,3,4])
        XCTAssertEqual(matrix.matrixSlice(at:0)?.matrix,matrix)
        let malformed=LearningTensor(shape:[2,2,2],values:[1])
        XCTAssertEqual(malformed.matrixSliceCount,0);XCTAssertNil(malformed.matrixSlice(at:0))
    }
}
