import XCTest
@testable import KwipMLCore

final class TensorLayoutTests:XCTestCase {
    func testReshapePreservesElementCountAndInfersExactlyOneDimension()throws {
        let input=try TensorLayout([2,3,4])
        XCTAssertEqual(try input.reshaped([2,-1]).dimensions,[2,12])
        XCTAssertEqual(try input.reshaped([-1,2,2]).dimensions,[6,2,2])
        XCTAssertThrowsError(try input.reshaped([5,-1]))
        XCTAssertThrowsError(try input.reshaped([-1,-1]))
        XCTAssertThrowsError(try input.reshaped([2,0,12]))
        XCTAssertThrowsError(try input.reshaped([2,3]))
    }
    func testTransposePermutationAndFlattenRespectAxisOrder()throws {
        let input=try TensorLayout([2,3,4])
        XCTAssertEqual(try input.transposed(0,-1).dimensions,[4,3,2])
        XCTAssertEqual(try input.permuted([2,0,1]).dimensions,[4,2,3])
        XCTAssertEqual(try input.flattened(from:1,through:-1).dimensions,[2,12])
        XCTAssertEqual(try input.flattened(from:0,through:1).dimensions,[6,4])
        XCTAssertEqual(try input.transposed(1,1),input)
        XCTAssertThrowsError(try input.permuted([0,0,2]))
        XCTAssertThrowsError(try input.permuted([0,1]))
        XCTAssertThrowsError(try input.transposed(Int.min,0))
        XCTAssertThrowsError(try input.flattened(from:2,through:0))
    }
    func testBoundsPreventOverflowAndDoNotInventUnknownDimensions()throws {
        for shape in [[],[0,2],[-1,2],[Int.max,2],Array(repeating:1,count:9),[100_000_001]] {
            XCTAssertThrowsError(try TensorLayout(shape))
        }
        XCTAssertEqual(try TensorLayout([100_000_000]).elementCount,100_000_000)
        XCTAssertThrowsError(try TensorLayout([2,3]).reshaped([Int.max,2,-1]))
    }
}
