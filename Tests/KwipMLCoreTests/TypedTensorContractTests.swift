import XCTest
@testable import KwipMLCore

final class TypedTensorContractTests:XCTestCase {
    func testLayoutOperationsPreserveIndexDType() throws {
        let input=try TypedTensorContract(sample:[2,3],dtype:.int64)
        let reshaped=try input.reshapedSample([3,2])
        XCTAssertEqual(reshaped.dtype,.int64)
        XCTAssertEqual(reshaped.layout.sample.dimensions,[3,2])
        XCTAssertEqual(try reshaped.transposed(1,2),input)
        XCTAssertEqual(try input.flattened(from:1,through:2),try TypedTensorContract(sample:[6],dtype:.int64))
        XCTAssertThrowsError(try input.flattened(from:0,through:2))
    }
    func testEmbeddingRequiresIntegerIndicesAndReturnsWeightType() throws {
        let config=try EmbeddingConfiguration(values:["num_embeddings":5,"embedding_dim":3,"padding_idx":.null,"max_norm":.null,"norm_type":2,"scale_grad_by_freq":false,"sparse":false,"freeze":false])
        for dtype in [ContractDType.int32,.int64] {
            let output=try TypedTensorContract(sample:[4],dtype:dtype).embedding(config,weightType:.float32)
            XCTAssertEqual(output,try TypedTensorContract(sample:[4,3],dtype:.float32))
        }
        for dtype in [ContractDType.float32,.float64,.bool,.string] {
            XCTAssertThrowsError(try TypedTensorContract(sample:[4],dtype:dtype).embedding(config,weightType:.float32))
        }
        XCTAssertThrowsError(try TypedTensorContract(sample:[4],dtype:.int64).embedding(config,weightType:.int64))
    }
    func testMergeRejectsImplicitTypePromotion() throws {
        let indices=try TypedTensorContract(sample:[2],dtype:.int64)
        let features=try TypedTensorContract(sample:[2],dtype:.float32)
        XCTAssertThrowsError(try indices.adding(features))
        XCTAssertThrowsError(try indices.concatenated(with:features,axis:1))
        XCTAssertEqual(try indices.adding(indices),indices)
        XCTAssertEqual(try indices.concatenated(with:indices,axis:1),try TypedTensorContract(sample:[4],dtype:.int64))
    }
}
