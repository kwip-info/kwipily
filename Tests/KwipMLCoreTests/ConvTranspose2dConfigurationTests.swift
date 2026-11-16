import XCTest
@testable import KwipMLCore

final class ConvTranspose2dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 4, "out_channels": 6, "kernel_size": .array([3, 2]), "stride": .array([2, 3]),
         "padding": .array([1, 0]), "output_padding": .array([1, 2]), "dilation": .array([1, 1]),
         "groups": 2, "bias": true].merging(changes) { _, value in value }
    }
    func testGroupedRectangularExpansionAndInputBasedWork() throws {
        let config = try ConvTranspose2dConfiguration(values: values())
        XCTAssertEqual(config.parameterCount, 78)
        let result = try config.resolved(input: BatchedTensorLayout(sample: [4, 5, 4]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 10, 13])
        XCTAssertEqual(result.uncroppedOutputShape, [12, 13])
        XCTAssertEqual(result.multiplyAdds, 1440)
    }
    func testDilationAlternativeAndCropLargerThanEffectiveKernel() throws {
        let config = try ConvTranspose2dConfiguration(values: values(["stride": .array([1, 1]), "dilation": .array([3, 2]), "output_padding": .array([2, 1])]))
        XCTAssertEqual(try config.resolved(input: BatchedTensorLayout(sample: [4, 5, 4])).outputLayout.sample.dimensions, [6, 11, 7])
        let crop = try ConvTranspose2dConfiguration(values: values(["kernel_size": .array([1, 1]), "stride": .array([4, 4]), "padding": .array([5, 5]), "output_padding": .array([0, 0])]))
        XCTAssertEqual(try crop.resolved(input: BatchedTensorLayout(sample: [4, 5, 5])).outputLayout.sample.dimensions, [6, 7, 7])
    }
    func testStrictPairsChannelsAndParameterCap() throws {
        let invalid: [[String: JSONValue]] = [
            ["kernel_size": 3], ["stride": .array([1])], ["padding": .array([0, 0, 0])],
            ["output_padding": .array([true, 0])], ["dilation": .array([1.5, 2])],
            ["groups": 3], ["in_channels": true], ["out_channels": 513], ["bias": 1],
            ["output_padding": .array([2, 2])], ["padding": .array([-1, 0])],
            ["padding_mode": .string("zeros")], ["padding_policy": .string("explicit")],
            ["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": .array([4096, 4096])]
        ]
        for change in invalid { XCTAssertThrowsError(try ConvTranspose2dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "bias")
        XCTAssertThrowsError(try ConvTranspose2dConfiguration(values: missing))
        XCTAssertEqual(try ConvTranspose2dConfiguration(values: values(["stride": .array([.number(2), .number(3)])])).values["stride"], .array([2, 3]))
    }
    func testShapeOutputScratchAndWorkBudgetsAreIndependent() throws {
        let ordinary = try ConvTranspose2dConfiguration(values: values())
        for shape in [[4, 5], [4, 2, 3, 4], [3, 5, 4], [4, 4097, 1]] {
            XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: shape)))
        }
        XCTAssertThrowsError(try ConvTranspose2dConfiguration(values: values(["padding": .array([20, 0])])).resolved(input: BatchedTensorLayout(sample: [4, 5, 4])))
        // Valid one-pixel output and tiny work, but an excessive uncropped accumulation volume.
        let scratch = try ConvTranspose2dConfiguration(values: values(["in_channels": 2, "out_channels": 2, "groups": 2,
            "kernel_size": .array([1, 1]), "stride": .array([4096, 4096]), "padding": .array([4096, 4096]), "output_padding": .array([0, 0])]))
        XCTAssertThrowsError(try scratch.resolved(input: BatchedTensorLayout(sample: [2, 3, 3])))
        // Output exceeds 100M (uncropped also exceeds it); input and work remain small.
        let output = try ConvTranspose2dConfiguration(values: values(["in_channels": 1, "out_channels": 8, "groups": 1,
            "kernel_size": .array([1, 1]), "stride": .array([4095, 4095]), "padding": .array([0, 0]), "output_padding": .array([0, 0])]))
        XCTAssertThrowsError(try output.resolved(input: BatchedTensorLayout(sample: [1, 2, 2])))
        let work = try ConvTranspose2dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 1,
            "kernel_size": .array([1, 1]), "stride": .array([1, 1]), "padding": .array([0, 0]), "output_padding": .array([0, 0])]))
        XCTAssertThrowsError(try work.resolved(input: BatchedTensorLayout(sample: [512, 32, 16])))
        XCTAssertNoThrow(try work.resolved(input: BatchedTensorLayout(sample: [512, 16, 16])))
    }
    func testExactWorkLimitAcceptsDespiteLargerExpandedOutput() throws {
        let config = try ConvTranspose2dConfiguration(values: values(["in_channels": 500, "out_channels": 512, "groups": 1,
            "kernel_size": .array([1, 1]), "stride": .array([2, 2]), "padding": .array([0, 0]), "output_padding": .array([0, 0])]))
        XCTAssertEqual(try config.resolved(input: BatchedTensorLayout(sample: [500, 20, 25])).multiplyAdds, 128_000_000)
        XCTAssertThrowsError(try config.resolved(input: BatchedTensorLayout(sample: [500, 20, 26])))
    }

    func testExactUncroppedVolumeLimit() throws {
        let base = values(["in_channels": 1, "out_channels": 100, "groups": 1, "kernel_size": .array([1, 1]),
            "stride": .array([1000, 1000]), "output_padding": .array([999, 999]), "padding": .array([499, 499])])
        let exact = try ConvTranspose2dConfiguration(values: base).resolved(input: BatchedTensorLayout(sample: [1, 1, 1]))
        XCTAssertEqual(exact.uncroppedOutputShape, [1000, 1000])
        XCTAssertEqual(exact.outputLayout.sample.dimensions, [100, 2, 2])
        var over = base; over["stride"] = .array([1001, 1000]); over["output_padding"] = .array([1000, 999])
        XCTAssertThrowsError(try ConvTranspose2dConfiguration(values: over).resolved(input: BatchedTensorLayout(sample: [1, 1, 1])))
    }

}
