import XCTest
@testable import KwipMLCore

final class SeparableConv2dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 2, "channel_multiplier": 2, "out_channels": 3,
         "kernel_size": .array([2, 3]), "stride": .array([1, 1]), "dilation": .array([1, 1]),
         "depthwise_bias": true, "pointwise_bias": false, "padding_policy": .string("same"),
         "padding": .array([0, 0]), "padding_mode": .string("zeros")].merging(changes) { _, value in value }
    }
    func testCompositionDerivesIntermediateWidthWithoutDuplicatingSpatialRules() throws {
        let config = try SeparableConv2dConfiguration(values: values())
        XCTAssertEqual(config.values, values())
        XCTAssertEqual(config.depthwise.convolution.outputChannels, 4)
        XCTAssertEqual(config.pointwise.inputChannels, 4)
        XCTAssertEqual(config.pointwise.outputChannels, 3)
        XCTAssertEqual(config.pointwise.groups, 1)
        XCTAssertEqual(config.pointwise.kernelSize, [1, 1])
        XCTAssertEqual(config.pointwise.stride, [1, 1])
        XCTAssertEqual(config.pointwise.padding, [0, 0])
        XCTAssertEqual(config.parameterCount, 40)
        let result = try config.resolved(input: BatchedTensorLayout(sample: [2, 4, 5]))
        XCTAssertEqual(result.intermediateLayout.sample.dimensions, [4, 4, 5])
        XCTAssertEqual(result.outputLayout.sample.dimensions, [3, 4, 5])
        XCTAssertEqual(result.multiplyAdds, 720)
    }
    func testIndependentBiasAndStrictAuthoredContract() throws {
        for depthwise in [false, true] { for pointwise in [false, true] {
            let config = try SeparableConv2dConfiguration(values: values(["depthwise_bias": .bool(depthwise), "pointwise_bias": .bool(pointwise)]))
            XCTAssertEqual(config.parameterCount, 36 + (depthwise ? 4 : 0) + (pointwise ? 3 : 0))
        }}
        for change: [String: JSONValue] in [["out_channels": true], ["out_channels": 0], ["out_channels": 513],
            ["groups": 1], ["bias": true], ["pointwise_in_channels": 4], ["pointwise_bias": 1],
            ["channel_multiplier": .number(1.5)], ["in_channels": 257, "channel_multiplier": 2]] {
            XCTAssertThrowsError(try SeparableConv2dConfiguration(values: values(change)))
        }
        var missing = values(); missing.removeValue(forKey: "pointwise_bias")
        XCTAssertThrowsError(try SeparableConv2dConfiguration(values: missing))
    }
    func testAggregateParametersAndSharedPerStageResourceBounds() throws {
        // Each layer is individually below two million; their sum is above it.
        XCTAssertThrowsError(try SeparableConv2dConfiguration(values: values([
            "in_channels": 500, "channel_multiplier": 1, "out_channels": 512,
            "kernel_size": .array([19, 200]), "depthwise_bias": false])))
        // Report combined work without inventing a tighter limit than the expanded primitives.
        let config = try SeparableConv2dConfiguration(values: values([
            "in_channels": 1, "channel_multiplier": 4, "out_channels": 1,
            "kernel_size": .array([1, 1]), "depthwise_bias": false]))
        XCTAssertEqual(try config.resolved(input: BatchedTensorLayout(sample: [1, 4000, 4000])).multiplyAdds, 128_000_000)
        XCTAssertEqual(try config.resolved(input: BatchedTensorLayout(sample: [1, 4000, 4001])).multiplyAdds, 128_032_000)
        XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [1, 4097, 4000])))
        XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [2, 4, 5])))
    }
}
