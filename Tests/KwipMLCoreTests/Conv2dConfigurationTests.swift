import XCTest
@testable import KwipMLCore

final class Conv2dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 4, "out_channels": 6, "kernel_size": .array([3, 2]), "stride": .array([1, 1]),
         "dilation": .array([1, 1]), "groups": 2, "bias": true, "padding_policy": .string("explicit"),
         "padding": .array([0, 0]), "padding_mode": .string("zeros")].merging(changes) { _, value in value }
    }
    func testGroupedRectangularKernelStrideDilationAndDepthwise() throws {
        let config = try Conv2dConfiguration(values: values(["stride": .array([2, 1]), "dilation": .array([2, 1]), "padding": .array([1, 0])]))
        XCTAssertEqual(config.parameterCount, 78)
        XCTAssertEqual(config.kernelSize, [3, 2])
        let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 11, 7]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 5, 6])
        XCTAssertEqual([result.paddingTop, result.paddingBottom, result.paddingLeft, result.paddingRight], [1, 1, 0, 0])
        XCTAssertEqual(result.multiplyAdds, 2160)
        let depthwise = try Conv2dConfiguration(values: values(["out_channels": 8, "groups": 4, "bias": false]))
        XCTAssertEqual(depthwise.parameterCount, 48)
        XCTAssertEqual(try depthwise.resolved(input: BatchedTensorLayout(sample: [4, 5, 7])).multiplyAdds, 864)
    }
    func testSameAsymmetryOnBothAxesAndValidAcrossAllModes() throws {
        for mode in ["zeros", "reflect", "replicate", "circular"] {
            let config = try Conv2dConfiguration(values: values(["kernel_size": .array([4, 2]), "padding_policy": .string("same"), "padding_mode": .string(mode)]))
            let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 7, 9]))
            XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 7, 9])
            XCTAssertEqual([result.paddingTop, result.paddingBottom, result.paddingLeft, result.paddingRight], [1, 2, 0, 1])
            let valid = try Conv2dConfiguration(values: values(["padding_policy": .string("valid"), "padding_mode": .string(mode)]))
            XCTAssertEqual(try valid.resolved(input: BatchedTensorLayout(sample: [4, 7, 9])).outputLayout.sample.dimensions, [6, 5, 8])
        }
    }
    func testExactPairsStrictFieldsTypesAndParameterCap() throws {
        let invalid: [[String: JSONValue]] = [
            ["kernel_size": 3], ["stride": .array([1])], ["padding": .array([0, 0, 0])],
            ["dilation": .array([true, 1])], ["kernel_size": .array([1.5, 2])],
            ["in_channels": true], ["out_channels": 513], ["groups": 3], ["bias": 1],
            ["padding": .array([-1, 0])], ["padding_policy": .string("other")], ["padding_mode": .string("other")],
            ["padding_policy": .string("valid"), "padding": .array([0, 1])],
            ["padding_policy": .string("same"), "stride": .array([1, 2])],
            ["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": .array([3, 3])]
        ]
        for change in invalid { XCTAssertThrowsError(try Conv2dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "bias")
        XCTAssertThrowsError(try Conv2dConfiguration(values: missing))
        XCTAssertThrowsError(try Conv2dConfiguration(values: values(["extra": 1])))
        XCTAssertEqual(try Conv2dConfiguration(values: values(["stride": .array([.number(2), .number(1)])])).values["stride"], .array([2, 1]))
    }
    func testShapePaddingAndPreallocationBudgets() throws {
        let ordinary = try Conv2dConfiguration(values: values())
        for shape in [[4, 9], [4, 2, 9, 1], [3, 9, 9], [4, 4097, 3], [4, 2, 7], [4, 9, 1]] {
            XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: shape)), "\(shape)")
        }
        for mode in ["reflect", "circular"] {
            let config = try Conv2dConfiguration(values: values(["padding_mode": .string(mode), "padding": .array([0, 5])]))
            XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [4, 9, 4])))
        }
        let hugePad = try Conv2dConfiguration(values: values(["in_channels": 2, "out_channels": 2, "groups": 2,
            "kernel_size": .array([1, 1]), "stride": .array([4096, 4096]), "padding": .array([4096, 4096])]))
        XCTAssertThrowsError(try hugePad.resolved(input: BatchedTensorLayout(sample: [2, 1, 1])))
        let heavy = try Conv2dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": .array([1, 1])]))
        XCTAssertThrowsError(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 32, 16])))
        XCTAssertNoThrow(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 16, 16])))
        let negative = try Conv2dConfiguration(values: values(["stride": .array([3, 3])]))
        XCTAssertThrowsError(try negative.resolved(input: BatchedTensorLayout(sample: [4, 2, 7])))
    }
}
