import XCTest
@testable import KwipMLCore

final class SpatialPatchConfigurationTests: XCTestCase {
    func testDefaultsAndAsymmetricWindowOrderShape() throws {
        let configuration = try SpatialPatchConfiguration(componentID: "unfold", values: [:])
        XCTAssertEqual(configuration.values["kernel_size"], .array([2, 2]))
        XCTAssertEqual(try configuration.resolved(layout: BatchedTensorLayout(sample: [2, 3, 4])).sample.dimensions, [8, 6])
        let spaced = try SpatialPatchConfiguration(componentID: "unfold", values: ["stride": .array([2,2]), "padding": .array([1,1]), "dilation": .array([2,1])])
        XCTAssertEqual(try spaced.resolved(layout: BatchedTensorLayout(sample: [2,3,4])).sample.dimensions, [8,6])
    }
    func testFoldOutputSizeAmbiguityAndOverlapGeometry() throws {
        for extent in [4,5] {
            let configuration = try SpatialPatchConfiguration(componentID: "fold", values: ["output_size": .array([.integer(Int64(extent)),.integer(Int64(extent))]), "stride": .array([2,2])])
            XCTAssertEqual(try configuration.resolved(layout: BatchedTensorLayout(sample: [8,4])).sample.dimensions, [2,extent,extent])
        }
        let overlap = try SpatialPatchConfiguration(componentID: "fold", values: ["output_size": .array([3,4]), "padding": .array([1,1])])
        XCTAssertEqual(try overlap.resolved(layout: BatchedTensorLayout(sample: [8,20])).sample.dimensions, [2,3,4])
    }
    func testRejectsMalformedConfigurationAndShapeBeforeAllocation() throws {
        for bad: [String:JSONValue] in [["kernel_size":2],["stride":.array([1])],["padding":.array([-1,0])],["dilation":.array([true,1])],["kernel_size":.array([.number(1.5),2])],["output_size":.array([2,2])],["unknown":1]] {
            XCTAssertThrowsError(try SpatialPatchConfiguration(componentID:"unfold",values:bad))
        }
        let unfold = try SpatialPatchConfiguration(componentID:"unfold",values:[:])
        for shape in [[3,4],[513,3,4],[1,1,1],[1,3,4,2]] { XCTAssertThrowsError(try unfold.resolved(layout:BatchedTensorLayout(sample:shape))) }
        let fold = try SpatialPatchConfiguration(componentID:"fold",values:["output_size":.array([3,4])])
        for shape in [[7,6],[8,5],[8,6,1]] { XCTAssertThrowsError(try fold.resolved(layout:BatchedTensorLayout(sample:shape))) }
    }
    func testExpandedPatchBudgetRejectsSmallSourceWithHugePadding() throws {
        let config = try SpatialPatchConfiguration(componentID:"unfold",values:["kernel_size":.array([512,512]),"padding":.array([512,512])])
        XCTAssertThrowsError(try config.resolved(layout:BatchedTensorLayout(sample:[1,1,1])))
    }
}
