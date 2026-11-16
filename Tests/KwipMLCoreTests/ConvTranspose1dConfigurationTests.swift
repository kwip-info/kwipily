import XCTest
@testable import KwipMLCore

final class ConvTranspose1dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 4, "out_channels": 6, "kernel_size": 3, "stride": 2, "padding": 1,
         "output_padding": 1, "dilation": 1, "groups": 2, "bias": true].merging(changes) { _, value in value }
    }
    func testGroupedExpansionUsesInputLengthWorkAndTransposedWeightShape() throws {
        let config = try ConvTranspose1dConfiguration(values: values())
        XCTAssertEqual(config.parameterCount, 42)
        let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 5]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 10])
        XCTAssertEqual(result.uncroppedOutputLength, 12)
        XCTAssertEqual(result.multiplyAdds, 180)
    }
    func testDilationAllowsOutputPaddingBeyondStrideAndLargeCropRemainsValid() throws {
        let dilated = try ConvTranspose1dConfiguration(values: values(["stride": 1, "dilation": 3, "output_padding": 2]))
        XCTAssertEqual(try dilated.resolved(input: BatchedTensorLayout(sample: [4, 5])).outputLayout.sample.dimensions, [6, 11])
        let cropped = try ConvTranspose1dConfiguration(values: values(["kernel_size": 1, "stride": 4, "padding": 5, "output_padding": 0]))
        XCTAssertEqual(try cropped.resolved(input: BatchedTensorLayout(sample: [4, 5])).outputLayout.sample.dimensions, [6, 7])
    }
    func testStrictKeysTypesBoundsAndParameterBudget() throws {
        let invalid: [[String: JSONValue]] = [
            ["in_channels": true], ["out_channels": 513], ["groups": 3], ["kernel_size": .array([3])],
            ["stride": 0], ["dilation": .number(1.5)], ["padding": -1], ["output_padding": 4097], ["bias": 1],
            ["stride": 2, "dilation": 3, "output_padding": 3],
            ["padding_mode": .string("zeros")], ["padding_policy": .string("explicit")],
            ["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": 8]
        ]
        for change in invalid { XCTAssertThrowsError(try ConvTranspose1dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "bias")
        XCTAssertThrowsError(try ConvTranspose1dConfiguration(values: missing))
        XCTAssertEqual(try ConvTranspose1dConfiguration(values: values(["stride": .number(2)])).values["stride"], .integer(2))
    }
    func testShapesOutputScratchAndWorkRefuseBeforeAllocation() throws {
        let ordinary = try ConvTranspose1dConfiguration(values: values())
        for shape in [[4], [4, 2, 3], [3, 5], [4, 4097]] {
            XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: shape)))
        }
        XCTAssertThrowsError(try ConvTranspose1dConfiguration(values: values(["padding": 20])).resolved(input: BatchedTensorLayout(sample: [4, 5])))
        XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: [4, 4096])))
        // Extreme dilation remains bounded even with valid parameter count.
        let scratch = try ConvTranspose1dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 512,
            "kernel_size": 3900, "dilation": 4096, "stride": 1, "padding": 4096, "output_padding": 0]))
        XCTAssertThrowsError(try scratch.resolved(input: BatchedTensorLayout(sample: [512, 1])))
        let heavy = try ConvTranspose1dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 1,
            "kernel_size": 1, "stride": 1, "padding": 0, "output_padding": 0]))
        XCTAssertThrowsError(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 512])))
        XCTAssertNoThrow(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 488])))
    }
}
