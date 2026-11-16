import XCTest
@testable import KwipMLCore

final class LayerNormConfigurationTests:XCTestCase {
    var values:[String:JSONValue] {["normalized_shape":.array([2,3]),"eps":.number(1e-5),"elementwise_affine":true,"bias":true]}
    func testCanonicalShapeParameterCountAndBatchIsolation() throws {
        var source=values;source["normalized_shape"] = .array([.number(2),.number(3)])
        let config=try LayerNormConfiguration(values:source)
        XCTAssertEqual(config.values["normalized_shape"],.array([.integer(2),.integer(3)]))
        XCTAssertEqual(config.parameterCount,12)
        let input=try BatchedTensorLayout(sample:[4,2,3])
        XCTAssertEqual(try config.outputLayout(for:input),input)
        XCTAssertThrowsError(try config.outputLayout(for:BatchedTensorLayout(sample:[3,2])))
        XCTAssertThrowsError(try config.outputLayout(for:BatchedTensorLayout(sample:[3])))
        source["bias"]=false
        XCTAssertEqual(try LayerNormConfiguration(values:source).parameterCount,6)
        source["elementwise_affine"]=false
        XCTAssertEqual(try LayerNormConfiguration(values:source).parameterCount,0)
    }
    func testMalformedFieldsAndUnsupportedEpsilonRejected() {
        for (key,value) in [("normalized_shape",JSONValue.array([])),("normalized_shape",.array([true])),("normalized_shape",.array([-1,3])),("normalized_shape",.array([.number(2.5)])),("normalized_shape",.array([.integer(Int64.max)])),("eps",.number(1e-300)),("eps",.number(1e300)),("eps",.number(.nan)),("eps",true),("elementwise_affine",1),("bias",.null),("extra",1)] {
            var malformed=values;malformed[key]=value
            XCTAssertThrowsError(try LayerNormConfiguration(values:malformed),key)
        }
        for key in values.keys {
            var malformed=values;malformed.removeValue(forKey:key)
            XCTAssertThrowsError(try LayerNormConfiguration(values:malformed))
        }
    }
}
