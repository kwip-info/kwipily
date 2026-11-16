import XCTest
@testable import KwipMLCore

final class DepthwiseConv2dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 3, "channel_multiplier": 2, "kernel_size": .array([2, 3]), "stride": .array([1, 1]),
         "dilation": .array([1, 1]), "bias": true, "padding_policy": .string("same"), "padding": .array([0, 0]),
         "padding_mode": .string("zeros")].merging(changes) { _, value in value }
    }
    func testDerivesIndependentChannelGroupsAndCanonicalAuthoredSettings() throws {
        let config = try DepthwiseConv2dConfiguration(values: values(["channel_multiplier": .number(2)]))
        XCTAssertEqual(config.convolution.inputChannels, 3)
        XCTAssertEqual(config.convolution.outputChannels, 6)
        XCTAssertEqual(config.convolution.groups, 3)
        XCTAssertEqual(config.parameterCount, 42)
        XCTAssertEqual(config.values, values())
        XCTAssertNil(config.values["groups"]); XCTAssertNil(config.values["out_channels"])
        let input = try BatchedTensorLayout(sample: [3, 7, 9])
        let result = try config.resolved(input: input)
        XCTAssertEqual(result, try config.convolution.resolved(input: input))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 7, 9])
        XCTAssertEqual([result.paddingTop, result.paddingBottom, result.paddingLeft, result.paddingRight], [0, 1, 1, 1])
    }
    func testStrictAuthoredKeysAndDerivedOutputChannelLimit() throws {
        let invalid: [[String: JSONValue]] = [
            ["in_channels": true], ["channel_multiplier": true], ["channel_multiplier": .number(1.5)],
            ["channel_multiplier": 0], ["channel_multiplier": 513], ["in_channels": 513],
            ["in_channels": 257, "channel_multiplier": 2], ["out_channels": 6], ["groups": 3],
            ["kernel_size": 3], ["padding_policy": .string("same"), "stride": .array([1, 2])], ["bias": 1]
        ]
        for change in invalid { XCTAssertThrowsError(try DepthwiseConv2dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "channel_multiplier")
        XCTAssertThrowsError(try DepthwiseConv2dConfiguration(values: missing))
        let limit = try DepthwiseConv2dConfiguration(values: values(["in_channels": 256, "channel_multiplier": 2]))
        XCTAssertEqual(limit.convolution.outputChannels, 512)
    }
    func testDelegatesSpatialPaddingAndAllocationRefusals() throws {
        let config = try DepthwiseConv2dConfiguration(values: values())
        XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [2, 7, 9])))
        XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [3, 7])))
        let reflect = try DepthwiseConv2dConfiguration(values: values(["padding_mode": .string("reflect")]))
        XCTAssertThrowsError(try reflect.resolved(input: BatchedTensorLayout(sample: [3, 1, 9])))
        XCTAssertThrowsError(try DepthwiseConv2dConfiguration(values: values(["in_channels": 512, "channel_multiplier": 1, "kernel_size": .array([64, 64])])) )
        let padding = try DepthwiseConv2dConfiguration(values: values(["in_channels": 2, "channel_multiplier": 1, "kernel_size": .array([1, 1]),
            "padding_policy": .string("explicit"), "padding": .array([4096, 4096]), "stride": .array([4096, 4096])]))
        XCTAssertThrowsError(try padding.resolved(input: BatchedTensorLayout(sample: [2, 1, 1])))
    }
    func testExactResourceBoundariesUseTheSharedConvolutionRules() throws {
        var parameters = values(["in_channels": 500, "channel_multiplier": 1, "kernel_size": .array([20, 200]), "bias": false])
        XCTAssertEqual(try DepthwiseConv2dConfiguration(values: parameters).parameterCount, 2_000_000)
        parameters["bias"] = true
        XCTAssertThrowsError(try DepthwiseConv2dConfiguration(values: parameters))
        let work = try DepthwiseConv2dConfiguration(values: values(["in_channels": 512, "channel_multiplier": 1,
            "kernel_size": .array([1, 250]), "padding_policy": .string("valid")]))
        XCTAssertEqual(try work.resolved(input: BatchedTensorLayout(sample: [512, 10, 349])).multiplyAdds, 128_000_000)
        XCTAssertThrowsError(try work.resolved(input: BatchedTensorLayout(sample: [512, 10, 350])))
        let output = try DepthwiseConv2dConfiguration(values: values(["in_channels": 1, "channel_multiplier": 100, "kernel_size": .array([1, 1])]))
        XCTAssertEqual(try output.resolved(input: BatchedTensorLayout(sample: [1, 1000, 1000])).outputLayout.sample.elementCount, 100_000_000)
        XCTAssertThrowsError(try output.resolved(input: BatchedTensorLayout(sample: [1, 1001, 1000])))
        var padded = values(["in_channels": 100, "channel_multiplier": 1, "kernel_size": .array([1, 1]),
            "padding_policy": .string("explicit"), "padding": .array([499, 499]), "stride": .array([1000, 1000])])
        XCTAssertNoThrow(try DepthwiseConv2dConfiguration(values: padded).resolved(input: BatchedTensorLayout(sample: [100, 2, 2])))
        padded["padding"] = .array([500, 499])
        XCTAssertThrowsError(try DepthwiseConv2dConfiguration(values: padded).resolved(input: BatchedTensorLayout(sample: [100, 2, 2])))
    }

}
