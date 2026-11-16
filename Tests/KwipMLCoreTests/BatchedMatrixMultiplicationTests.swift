import XCTest
@testable import KwipMLCore

final class BatchedMatrixMultiplicationTests:XCTestCase {
    func testMatrixProductAndAttentionHeadBroadcasting()throws {
        func product(_ a:[Int],_ b:[Int])throws->[Int] {
            try BatchedTensorLayout(sample:a).matrixMultiplied(by:BatchedTensorLayout(sample:b)).sample.dimensions
        }
        XCTAssertEqual(try product([3,4],[4,5]),[3,5])
        XCTAssertEqual(try product([8,3,4],[1,4,5]),[8,3,5])
        XCTAssertEqual(try product([8,3,4],[4,5]),[8,3,5])
        XCTAssertEqual(try product([3,4],[8,4,5]),[8,3,5])
        XCTAssertEqual(try product([2,1,3,4],[7,4,5]),[2,7,3,5])
        XCTAssertEqual(try product([1,1,1,1,1,3,4],[4,5]),[1,1,1,1,1,3,5])
    }
    func testRejectsContractionMismatchHeadMismatchAndOutputOverflow()throws {
        for (a,b) in [([3,4],[3,5]),([2,3,4],[7,4,5]),([100_000,1],[1,100_000]),([4],[4,5]),([3,4],[4])] {
            XCTAssertThrowsError(try BatchedTensorLayout(sample:a).matrixMultiplied(by:BatchedTensorLayout(sample:b)),"\(a) @ \(b)")
        }
    }
    func testTypedProductRequiresMatchingFloatingPointOperands()throws {
        for dtype in [ContractDType.float32,.float64] {
            let result=try TypedTensorContract(sample:[3,4],dtype:dtype).matrixMultiplied(by:TypedTensorContract(sample:[4,5],dtype:dtype))
            XCTAssertEqual(result.dtype,dtype)
            XCTAssertEqual(result.layout.displayDimensions,["N","3","5"])
        }
        for dtype in [ContractDType.int32,.int64,.bool,.string] {
            XCTAssertThrowsError(try TypedTensorContract(sample:[3,4],dtype:dtype).matrixMultiplied(by:TypedTensorContract(sample:[4,5],dtype:dtype)))
        }
        XCTAssertThrowsError(try TypedTensorContract(sample:[3,4],dtype:.float32).matrixMultiplied(by:TypedTensorContract(sample:[4,5],dtype:.float64)))
    }
}
