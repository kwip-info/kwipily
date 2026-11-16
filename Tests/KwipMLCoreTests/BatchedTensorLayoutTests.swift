import XCTest
@testable import KwipMLCore

final class BatchedTensorLayoutTests:XCTestCase {
    func testLayoutTransformsPreserveIndependentBatchAxis()throws {
        let shape=try BatchedTensorLayout(sample:[3,4])
        XCTAssertEqual(shape.displayDimensions,["N","3","4"])
        XCTAssertEqual(try shape.reshapedSample([2,-1]).sample.dimensions,[2,6])
        XCTAssertEqual(try shape.transposed(1,-1).sample.dimensions,[4,3])
        XCTAssertEqual(try shape.flattened(from:1,through:-1).sample.dimensions,[12])
        XCTAssertThrowsError(try shape.transposed(0,1))
        XCTAssertThrowsError(try shape.transposed(-3,1))
        XCTAssertThrowsError(try shape.flattened(from:0,through:2))
    }
    func testLinearAndMergePreserveLeadingSampleAxes()throws {
        let a=try BatchedTensorLayout(sample:[3,4])
        XCTAssertEqual(try a.linear(inputFeatures:4,outputFeatures:5).sample.dimensions,[3,5])
        XCTAssertThrowsError(try a.linear(inputFeatures:3,outputFeatures:5))
        XCTAssertEqual(try a.concatenated(with:BatchedTensorLayout(sample:[3,2]),axis:-1).sample.dimensions,[3,6])
        XCTAssertEqual(try a.concatenated(with:BatchedTensorLayout(sample:[7,4]),axis:1).sample.dimensions,[10,4])
        XCTAssertThrowsError(try a.concatenated(with:BatchedTensorLayout(sample:[7,2]),axis:1))
        XCTAssertThrowsError(try a.concatenated(with:a,axis:0))
        XCTAssertEqual(try a.adding(a),a)
        XCTAssertThrowsError(try a.adding(BatchedTensorLayout(sample:[1,4])))
    }
}
