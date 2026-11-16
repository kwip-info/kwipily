import XCTest
@testable import KwipMLCore

final class MaxPool3dConfigurationTests:XCTestCase {
    private func values(_ changes:[String:JSONValue]=[:])->[String:JSONValue] {
        ["kernel_size":.array([2,2,3]),"stride":.null,"padding":.array([0,0,0]),"dilation":.array([1,1,1]),"ceil_mode":false,"return_indices":true].merging(changes){_,v in v}
    }
    func testVolumeAxesNullStrideAndIndices() throws {
        let c=try MaxPool3dConfiguration(values:values())
        XCTAssertEqual(c.effectiveStride,[2,2,3]);XCTAssertEqual(c.values,values())
        let r=try c.resolved(input:.init(sample:[2,5,7,8]))
        XCTAssertEqual(r.outputLayout.sample.dimensions,[2,2,3,2]);XCTAssertEqual(r.indicesLayout,r.outputLayout)
    }
    func testCeilingAndIndependentDepthValidation() throws {
        let c=try MaxPool3dConfiguration(values:values(["kernel_size":.array([2,2,2]),"padding":.array([1,1,1]),"ceil_mode":true]))
        XCTAssertEqual(try c.resolved(input:.init(sample:[2,1,4,1])).outputLayout.sample.dimensions,[2,1,3,1])
        for shape in [[2,3,4],[2,3,4,5,6],[513,2,3,4],[2,4097,3,4],[2,2,4097,4],[2,2,3,4097]] {
            XCTAssertThrowsError(try c.resolved(input:.init(sample:shape)))
        }
        for change:[String:JSONValue] in [["kernel_size":.array([2,3])],["stride":.array([1,0,1])],["padding":.array([2,0,0])],["dilation":.array([1,1,true])],["extra":0]] {
            XCTAssertThrowsError(try MaxPool3dConfiguration(values:values(change)))
        }
    }
}
