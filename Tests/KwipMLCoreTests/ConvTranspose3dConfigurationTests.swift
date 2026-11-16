import XCTest
@testable import KwipMLCore

final class ConvTranspose3dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 4, "out_channels": 6, "kernel_size": .array([2, 3, 2]), "stride": .array([2, 2, 3]),
         "padding": .array([0, 1, 0]), "output_padding": .array([1, 1, 2]), "dilation": .array([1, 1, 1]),
         "groups": 2, "bias": true].merging(changes) { _, value in value }
    }
    func testGroupedRectangularExpansionAndInputBasedWork() throws {
        let config = try ConvTranspose3dConfiguration(values: values())
        XCTAssertEqual(config.parameterCount, 150)
        let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 3, 5, 4]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 7, 10, 13])
        XCTAssertEqual(result.uncroppedOutputShape, [7, 12, 13])
        XCTAssertEqual(result.multiplyAdds, 8640)
    }
    func testDilationAlternativeAndCropLargerThanEffectiveKernel() throws {
        let config = try ConvTranspose3dConfiguration(values: values(["stride": .array([1, 1, 1]), "dilation": .array([2, 3, 2]), "output_padding": .array([1, 2, 1])]))
        XCTAssertEqual(try config.resolved(input: BatchedTensorLayout(sample: [4, 3, 5, 4])).outputLayout.sample.dimensions, [6, 6, 11, 7])
        let crop = try ConvTranspose3dConfiguration(values: values(["kernel_size": .array([1, 1, 1]), "stride": .array([4, 4, 4]), "padding": .array([5, 5, 5]), "output_padding": .array([0, 0, 0])]))
        XCTAssertEqual(try crop.resolved(input: BatchedTensorLayout(sample: [4, 5, 5, 5])).outputLayout.sample.dimensions, [6, 7, 7, 7])
    }
    func testStrictTriplesChannelsAndParameterCap() throws {
        let invalid: [[String: JSONValue]] = [
            ["kernel_size": 3], ["stride": .array([1])], ["padding": .array([0, 0, 0, 0])],
            ["output_padding": .array([true, 0, 0])], ["dilation": .array([1, 1.5, 2])],
            ["groups": 3], ["in_channels": true], ["out_channels": 513], ["bias": 1],
            ["output_padding": .array([2, 1, 2])], ["padding": .array([-1, 0, 0])],
            ["padding_mode": .string("zeros")], ["padding_policy": .string("explicit")],
            ["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": .array([4096, 4096, 4096])]
        ]
        for change in invalid { XCTAssertThrowsError(try ConvTranspose3dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "bias")
        XCTAssertThrowsError(try ConvTranspose3dConfiguration(values: missing))
        XCTAssertEqual(try ConvTranspose3dConfiguration(values: values(["stride": .array([.number(2), .number(2), .number(3)])])).values["stride"], .array([2, 2, 3]))
    }
    func testShapeOutputScratchAndWorkBudgetsAreIndependent() throws {
        let ordinary = try ConvTranspose3dConfiguration(values: values())
        for shape in [[4, 5, 4], [4, 2, 3, 4, 1], [3, 3, 5, 4], [4, 4097, 1, 1]] {
            XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: shape)))
        }
        XCTAssertThrowsError(try ConvTranspose3dConfiguration(values: values(["padding": .array([20, 0, 0])])).resolved(input: BatchedTensorLayout(sample: [4, 3, 5, 4])))
        // Valid one-pixel output and tiny work, but an excessive uncropped accumulation volume.
        let scratch = try ConvTranspose3dConfiguration(values: values(["in_channels": 2, "out_channels": 2, "groups": 2,
            "kernel_size": .array([1, 1, 1]), "stride": .array([1, 4096, 4096]), "padding": .array([0, 4096, 4096]), "output_padding": .array([0, 0, 0])]))
        XCTAssertThrowsError(try scratch.resolved(input: BatchedTensorLayout(sample: [2, 1, 3, 3])))
        // Output exceeds 100M (uncropped also exceeds it); input and work remain small.
        let output = try ConvTranspose3dConfiguration(values: values(["in_channels": 1, "out_channels": 8, "groups": 1,
            "kernel_size": .array([1, 1, 1]), "stride": .array([1, 4095, 4095]), "padding": .array([0, 0, 0]), "output_padding": .array([0, 0, 0])]))
        XCTAssertThrowsError(try output.resolved(input: BatchedTensorLayout(sample: [1, 1, 2, 2])))
        let work = try ConvTranspose3dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 1,
            "kernel_size": .array([1, 1, 1]), "stride": .array([1, 1, 1]), "padding": .array([0, 0, 0]), "output_padding": .array([0, 0, 0])]))
        XCTAssertThrowsError(try work.resolved(input: BatchedTensorLayout(sample: [512, 1, 32, 16])))
        XCTAssertNoThrow(try work.resolved(input: BatchedTensorLayout(sample: [512, 1, 16, 16])))
    }
    func testExactWorkLimitAcceptsDespiteLargerExpandedOutput() throws {
        let config = try ConvTranspose3dConfiguration(values: values(["in_channels": 500, "out_channels": 512, "groups": 1,
            "kernel_size": .array([1, 1, 1]), "stride": .array([1, 2, 2]), "padding": .array([0, 0, 0]), "output_padding": .array([0, 0, 0])]))
        XCTAssertEqual(try config.resolved(input: BatchedTensorLayout(sample: [500, 1, 20, 25])).multiplyAdds, 128_000_000)
        XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [500, 1, 20, 26])))
    }

    func testExactUncroppedVolumeLimit() throws {
        let base = values(["in_channels": 1, "out_channels": 100, "groups": 1, "bias": false,
            "kernel_size": .array([1, 1, 1]), "stride": .array([100, 100, 100]),
            "output_padding": .array([99, 99, 99]), "padding": .array([49, 49, 49])])
        let exact = try ConvTranspose3dConfiguration(values: base).resolved(input: BatchedTensorLayout(sample: [1, 1, 1, 1]))
        XCTAssertEqual(exact.uncroppedOutputShape, [100, 100, 100])
        XCTAssertEqual(exact.outputLayout.sample.dimensions, [100, 2, 2, 2])
        var over = base; over["stride"] = .array([101, 100, 100]); over["output_padding"] = .array([100, 99, 99])
        XCTAssertThrowsError(try ConvTranspose3dConfiguration(values: over).resolved(input: BatchedTensorLayout(sample: [1, 1, 1, 1])))
    }
    func testInputVolumeAndParameterBoundaries() throws {
        let inputConfig = try ConvTranspose3dConfiguration(values: values(["in_channels": 100, "out_channels": 1, "groups": 1,
            "kernel_size": .array([1, 1, 1]), "stride": .array([1, 1, 1]), "padding": .array([0, 0, 0]), "output_padding": .array([0, 0, 0])]))
        XCTAssertEqual(try inputConfig.resolved(input: BatchedTensorLayout(sample: [100, 100, 100, 100])).multiplyAdds, 100_000_000)
        // Layout admission itself owns input sample size; no oversized layout reaches resolution.
        XCTAssertThrowsError(try BatchedTensorLayout(sample: [100, 101, 100, 100]))
        let exact = values(["in_channels": 500, "out_channels": 500, "groups": 1, "bias": false, "kernel_size": .array([2, 2, 2])])
        XCTAssertEqual(try ConvTranspose3dConfiguration(values: exact).parameterCount, 2_000_000)
        var over = exact; over["bias"] = true
        XCTAssertThrowsError(try ConvTranspose3dConfiguration(values: over))
    }
    func testOutputPaddingAlternativeMustHoldOnEveryAxis() throws {
        let base = values(["stride": .array([1, 3, 1]), "dilation": .array([2, 1, 3]), "output_padding": .array([1, 2, 2])])
        XCTAssertNoThrow(try ConvTranspose3dConfiguration(values: base))
        for axis in 0..<3 {
            var invalid = base
            var outputPadding: [JSONValue] = [1, 2, 2]
            outputPadding[axis] = .integer(Int64([2, 3, 3][axis]))
            invalid["output_padding"] = .array(outputPadding)
            XCTAssertThrowsError(try ConvTranspose3dConfiguration(values: invalid))
        }
    }
}
