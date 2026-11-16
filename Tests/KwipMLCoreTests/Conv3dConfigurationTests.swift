import XCTest
@testable import KwipMLCore

final class Conv3dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 4, "out_channels": 6, "kernel_size": .array([2, 3, 2]), "stride": .array([1, 1, 1]),
         "dilation": .array([1, 1, 1]), "groups": 2, "bias": true, "padding_policy": .string("explicit"),
         "padding": .array([0, 0, 0]), "padding_mode": .string("zeros")].merging(changes) { _, value in value }
    }
    func testGroupedRectangularVolumeAndDepthwiseArithmetic() throws {
        let config = try Conv3dConfiguration(values: values(["stride": .array([2, 1, 1]), "dilation": .array([2, 1, 1]), "padding": .array([1, 0, 0])]))
        XCTAssertEqual(config.parameterCount, 150)
        XCTAssertEqual(config.kernelSize, [2, 3, 2])
        let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 7, 5, 6]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 4, 3, 5])
        XCTAssertEqual([result.paddingFront, result.paddingBack, result.paddingTop, result.paddingBottom, result.paddingLeft, result.paddingRight], [1, 1, 0, 0, 0, 0])
        XCTAssertEqual(result.multiplyAdds, 8640)
        let depthwise = try Conv3dConfiguration(values: values(["out_channels": 8, "groups": 4, "bias": false]))
        XCTAssertEqual(depthwise.parameterCount, 96)
    }
    func testSameAsymmetryAndValidAcrossModes() throws {
        for mode in ["zeros", "reflect", "replicate", "circular"] {
            let config = try Conv3dConfiguration(values: values(["padding_policy": .string("same"), "padding_mode": .string(mode)]))
            let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 5, 7, 9]))
            XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 5, 7, 9])
            XCTAssertEqual([result.paddingFront, result.paddingBack, result.paddingTop, result.paddingBottom, result.paddingLeft, result.paddingRight], [0, 1, 1, 1, 0, 1])
            let valid = try Conv3dConfiguration(values: values(["padding_policy": .string("valid"), "padding_mode": .string(mode)]))
            XCTAssertEqual(try valid.resolved(input: BatchedTensorLayout(sample: [4, 5, 7, 9])).outputLayout.sample.dimensions, [6, 4, 5, 8])
        }
    }
    func testStrictTriplesGroupsTypesAndAdversarialParameterProduct() throws {
        let invalid: [[String: JSONValue]] = [
            ["kernel_size": 3], ["stride": .array([1, 1])], ["padding": .array([0, 0, 0, 0])],
            ["dilation": .array([1, true, 1])], ["kernel_size": .array([1, 1.5, 2])],
            ["in_channels": true], ["out_channels": 513], ["groups": 3], ["bias": 1],
            ["padding_policy": .string("same"), "stride": .array([2, 1, 1])],
            ["padding_policy": .string("valid"), "padding": .array([0, 0, 1])],
            ["padding_mode": .string("other")], ["padding": .array([0, -1, 0])],
            ["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": .array([4096, 4096, 4096])]
        ]
        for change in invalid { XCTAssertThrowsError(try Conv3dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "bias")
        XCTAssertThrowsError(try Conv3dConfiguration(values: missing))
        XCTAssertThrowsError(try Conv3dConfiguration(values: values(["extra": 1])))
        XCTAssertEqual(try Conv3dConfiguration(values: values(["stride": .array([.number(2), 1, 1])])).values["stride"], .array([2, 1, 1]))
    }
    func testEachAxisPaddingConstraintsAndNegativeOutput() throws {
        for axis in 0..<3 {
            var pads: [JSONValue] = [0, 0, 0]; pads[axis] = 4
            XCTAssertThrowsError(try Conv3dConfiguration(values: values(["padding_mode": .string("reflect"), "padding": .array(pads)])).resolved(input: BatchedTensorLayout(sample: [4, 4, 4, 4])))
            pads[axis] = 5
            XCTAssertThrowsError(try Conv3dConfiguration(values: values(["padding_mode": .string("circular"), "padding": .array(pads)])).resolved(input: BatchedTensorLayout(sample: [4, 4, 4, 4])))
            var shape = [4, 5, 5, 5]; shape[axis+1] = 1
            XCTAssertThrowsError(try Conv3dConfiguration(values: values(["stride": .array([3, 3, 3])])).resolved(input: BatchedTensorLayout(sample: shape)))
        }
        let ordinary = try Conv3dConfiguration(values: values())
        for shape in [[4, 5, 5], [3, 5, 5, 5], [4, 4097, 2, 2]] {
            XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: shape)))
        }
    }
    func testPaddedVolumeAndWorkAreBoundedBeforeAllocationOrOverflow() throws {
        let padding = try Conv3dConfiguration(values: values(["kernel_size": .array([1, 1, 1]), "padding": .array([4096, 4096, 4096]), "stride": .array([4096, 4096, 4096])]))
        XCTAssertThrowsError(try padding.resolved(input: BatchedTensorLayout(sample: [4, 1, 1, 1])))
        let heavy = try Conv3dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": .array([1, 1, 1])]))
        XCTAssertThrowsError(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 8, 8, 8])))
        XCTAssertNoThrow(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 4, 4, 4])))
        let giantOutput = try Conv3dConfiguration(values: values(["in_channels": 1, "out_channels": 512, "groups": 1, "kernel_size": .array([1, 1, 1]), "padding": .array([2047, 2047, 2047])]))
        XCTAssertThrowsError(try giantOutput.resolved(input: BatchedTensorLayout(sample: [1, 1, 1, 1])))
    }
}
