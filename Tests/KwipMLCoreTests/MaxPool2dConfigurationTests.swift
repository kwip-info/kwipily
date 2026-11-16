import XCTest
@testable import KwipMLCore

final class MaxPool2dConfigurationTests: XCTestCase {
    private func values(_ changes:[String:JSONValue] = [:])->[String:JSONValue] {
        ["kernel_size":.array([2,3]),"stride":.null,"padding":.array([0,0]),"dilation":.array([1,1]),"ceil_mode":false,"return_indices":false].merging(changes){_,v in v}
    }
    func testPairedGeometryNullableStrideAndIndices() throws {
        let c=try MaxPool2dConfiguration(values:values(["return_indices":true]))
        XCTAssertEqual(c.effectiveStride,[2,3])
        XCTAssertEqual(c.values,values(["return_indices":true]))
        let result=try c.resolved(input:.init(sample:[3,5,7]))
        XCTAssertEqual(result.outputLayout.sample.dimensions,[3,2,2])
        XCTAssertEqual(result.indicesLayout,result.outputLayout)
    }
    func testCeilingCorrectionOnIndependentAxes() throws {
        let c=try MaxPool2dConfiguration(values:values(["kernel_size":.array([2,2]),"stride":.array([2,2]),"padding":.array([1,1]),"ceil_mode":true]))
        XCTAssertEqual(try c.resolved(input:.init(sample:[2,1,4])).outputLayout.sample.dimensions,[2,1,3])
        let floor=try MaxPool2dConfiguration(values:values())
        XCTAssertThrowsError(try floor.resolved(input:.init(sample:[2,1,7])))
    }
    func testStrictPairsAndBounds() throws {
        for changes:[String:JSONValue] in [["kernel_size":2],["kernel_size":.array([2])],["kernel_size":.array([2,3,4])],["stride":.array([1,0])],["padding":.array([2,0])],["dilation":.array([1,4097])],["ceil_mode":1],["extra":0]] {
            XCTAssertThrowsError(try MaxPool2dConfiguration(values:values(changes)))
        }
        let c=try MaxPool2dConfiguration(values:values())
        for shape in [[2,4],[2,3,4,5],[513,2,3],[2,4097,3],[2,2,4097]] {
            XCTAssertThrowsError(try c.resolved(input:.init(sample:shape)))
        }
    }
}
