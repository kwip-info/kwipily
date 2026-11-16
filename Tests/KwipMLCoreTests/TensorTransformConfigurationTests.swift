import XCTest
@testable import KwipMLCore

final class TensorTransformConfigurationTests:XCTestCase {
    func testCanonicalIntegersAndDeferredInputChecks()throws {
        let normalized=try TensorTransformConfiguration(operation:"reshape",values:["shape":.array([2.0,-1.0])])
        XCTAssertEqual(normalized.values,["shape":.array([.integer(2),.integer(-1)])])
        XCTAssertThrowsError(try normalized.outputLayout(for:BatchedTensorLayout(sample:[3])))
        for shape:[JSONValue] in [[.integer(Int64.max),2],[100_000_000,2],Array(repeating:1,count:8)] {
            XCTAssertThrowsError(try TensorTransformConfiguration(operation:"reshape",values:["shape":.array(shape)]))
        }
        let reversed=try TensorTransformConfiguration(operation:"flatten",values:["start_dim":-1,"end_dim":1])
        XCTAssertThrowsError(try reversed.outputLayout(for:BatchedTensorLayout(sample:[3,4])))
        XCTAssertThrowsError(try TensorTransformConfiguration(operation:"flatten",values:["start_dim":true,"end_dim":-1]))
    }
    func testConfigurationRoundTripAndShapeApplication()throws {
        let reshape=try TensorTransformConfiguration(operation:"reshape",values:["shape":.array([2,-1])])
        XCTAssertEqual(reshape.values,["shape":.array([2,-1])])
        XCTAssertEqual(try reshape.outputLayout(for:BatchedTensorLayout(sample:[3,4])).sample.dimensions,[2,6])
        let transpose=try TensorTransformConfiguration(operation:"transpose",values:["dim0":1,"dim1":-1])
        XCTAssertEqual(try transpose.outputLayout(for:BatchedTensorLayout(sample:[3,4])).sample.dimensions,[4,3])
        let flatten=try TensorTransformConfiguration(operation:"flatten",values:["start_dim":1,"end_dim":-1])
        XCTAssertEqual(try flatten.outputLayout(for:BatchedTensorLayout(sample:[3,4])).sample.dimensions,[12])
    }
    func testMalformedOrAmbiguousConfigurationIsRejected()throws {
        for values:[String:JSONValue] in [[:],["shape":.array([])],["shape":.array([-1,-1])],["shape":.array([0,2])],
            ["shape":.array([true,2])],["shape":.array([2,3]),"extra":1],["shape":.string("2,3")]] {
            XCTAssertThrowsError(try TensorTransformConfiguration(operation:"reshape",values:values))
        }
        XCTAssertThrowsError(try TensorTransformConfiguration(operation:"transpose",values:["dim0":0,"dim1":1]))
        XCTAssertThrowsError(try TensorTransformConfiguration(operation:"transpose",values:["dim0":1,"dim1":8]))
        XCTAssertThrowsError(try TensorTransformConfiguration(operation:"transpose",values:["dim0":1.5,"dim1":1]))
        XCTAssertThrowsError(try TensorTransformConfiguration(operation:"unknown",values:[:]))
        let config=try TensorTransformConfiguration(operation:"transpose",values:["dim0":-3,"dim1":1])
        XCTAssertThrowsError(try config.outputLayout(for:BatchedTensorLayout(sample:[3,4])))
    }
}
