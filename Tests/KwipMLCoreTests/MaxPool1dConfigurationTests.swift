import XCTest
@testable import KwipMLCore

final class MaxPool1dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["kernel_size":3,"stride":.null,"padding":0,"dilation":1,"ceil_mode":false,"return_indices":false].merging(changes){_,v in v}
    }
    func testStrideDefaultAndCanonicalSettings() throws {
        let c = try MaxPool1dConfiguration(values:values())
        XCTAssertEqual(c.effectiveStride,3)
        XCTAssertEqual(c.values,values())
        XCTAssertEqual(try c.resolved(input:.init(sample:[2,7])).outputLayout.sample.dimensions,[2,2])
        XCTAssertNil(try c.resolved(input:.init(sample:[2,7])).indicesLayout)
        let indexed = try MaxPool1dConfiguration(values:values(["return_indices":true]))
        let r = try indexed.resolved(input:.init(sample:[2,7]))
        XCTAssertEqual(r.indicesLayout,r.outputLayout)
    }
    func testCeilingDropsWindowsStartingInRightPaddingAndUsesSignedFloor() throws {
        let c = try MaxPool1dConfiguration(values:values(["stride":2,"ceil_mode":true]))
        XCTAssertEqual(try c.resolved(input:.init(sample:[1,4])).outputLayout.sample.dimensions,[1,2])
        let edge = try MaxPool1dConfiguration(values:values(["kernel_size":2,"stride":2,"padding":1,"ceil_mode":true]))
        XCTAssertEqual(try edge.resolved(input:.init(sample:[1,1])).outputLayout.sample.dimensions,[1,1])
        let empty = try MaxPool1dConfiguration(values:values(["kernel_size":2,"stride":2]))
        XCTAssertThrowsError(try empty.resolved(input:.init(sample:[1,1])))
        let partial = try MaxPool1dConfiguration(values:values(["kernel_size":2,"stride":2,"ceil_mode":true]))
        XCTAssertEqual(try partial.resolved(input:.init(sample:[1,1])).outputLayout.sample.dimensions,[1,1])
    }
    func testDilationPreservesChannelsAndRejectsInvalidGeometry() throws {
        let c = try MaxPool1dConfiguration(values:values(["kernel_size":2,"stride":1,"padding":1,"dilation":2]))
        XCTAssertEqual(try c.resolved(input:.init(sample:[3,2])).outputLayout.sample.dimensions,[3,2])
        XCTAssertThrowsError(try c.resolved(input:.init(sample:[2])))
        XCTAssertThrowsError(try c.resolved(input:.init(sample:[1,2,3])))
        XCTAssertThrowsError(try c.resolved(input:.init(sample:[513,2])))
        XCTAssertThrowsError(try c.resolved(input:.init(sample:[2,4097])))
        let tooWide = try MaxPool1dConfiguration(values:values(["dilation":4096]))
        XCTAssertThrowsError(try tooWide.resolved(input:.init(sample:[2,7])))
    }
    func testStrictKeysTypesAndPaddingLimit() throws {
        for changes: [String:JSONValue] in [["kernel_size":0],["kernel_size":true],["stride":0],["stride":.number(1.5)],["padding":2],["padding":-1],["dilation":0],["ceil_mode":1],["return_indices":1],["unknown":0]] {
            XCTAssertThrowsError(try MaxPool1dConfiguration(values:values(changes)),"\(changes)")
        }
        var missing = values();missing.removeValue(forKey:"stride")
        XCTAssertThrowsError(try MaxPool1dConfiguration(values:missing))
    }
}
