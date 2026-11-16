import XCTest
@testable import KwipMLCore

final class EmbeddingConfigurationTests:XCTestCase {
    var values:[String:JSONValue] {[
        "num_embeddings":5,"embedding_dim":3,"padding_idx":.null,"max_norm":.null,
        "norm_type":2,"scale_grad_by_freq":false,"sparse":false,"freeze":false
    ]}
    func testLookupAppendsFeatureAxisAndBoundsAllocation() throws {
        let config=try EmbeddingConfiguration(values:values)
        XCTAssertEqual(config.parameterCount,15)
        XCTAssertEqual(try config.outputLayout(for:BatchedTensorLayout(sample:[2,4])).sample.dimensions,[2,4,3])
        XCTAssertThrowsError(try config.outputLayout(for:BatchedTensorLayout(sample:Array(repeating:1,count:7))))
        XCTAssertThrowsError(try config.outputLayout(for:BatchedTensorLayout(sample:[100_000_000])))
    }
    func testPaddingCanonicalizationAndExplicitOptions() throws {
        var source=values;source["padding_idx"] = -1
        source["num_embeddings"] = .number(5)
        source["max_norm"] = .number(1.5);source["norm_type"] = .number(1)
        source["scale_grad_by_freq"]=true;source["sparse"]=true;source["freeze"]=true
        let config=try EmbeddingConfiguration(values:source)
        XCTAssertEqual(config.paddingIndex,4)
        XCTAssertEqual(config.values["padding_idx"],4)
        XCTAssertEqual(config.values["num_embeddings"],.integer(5))
        XCTAssertEqual(config.maximumNorm,1.5)
        XCTAssertTrue(config.sparse);XCTAssertTrue(config.frozen);XCTAssertTrue(config.scaleGradientByFrequency)
        XCTAssertEqual(try EmbeddingConfiguration(values:config.values),config)
        source["padding_idx"] = -5
        XCTAssertEqual(try EmbeddingConfiguration(values:source).paddingIndex,0)
    }
    func testRejectMalformedAndOverBudgetConfiguration() {
        for (key,value) in [("num_embeddings",JSONValue.bool(true)),("num_embeddings",0),("embedding_dim",-1),
                            ("embedding_dim",.number(1.5)),("num_embeddings",.integer(Int64.max)),
                            ("padding_idx",5),("padding_idx",-6),("padding_idx",true),
                            ("max_norm",0),("max_norm",.number(.infinity)),("max_norm",true),
                            ("norm_type",0),("norm_type",.number(.nan)),("scale_grad_by_freq",1),
                            ("sparse",.null),("freeze",1),("extra",1)] {
            var source=values;source[key]=value
            XCTAssertThrowsError(try EmbeddingConfiguration(values:source),key)
        }
        for key in values.keys {
            var source=values;source.removeValue(forKey:key)
            XCTAssertThrowsError(try EmbeddingConfiguration(values:source),key)
        }
        var source=values;source["num_embeddings"]=100_000_000
        XCTAssertThrowsError(try EmbeddingConfiguration(values:source))
    }
}
