import XCTest
@testable import KwipMLCore

final class Conv1dConfigurationTests: XCTestCase {
    private func values(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        ["in_channels": 4, "out_channels": 6, "kernel_size": 3, "stride": 1, "dilation": 1,
         "groups": 2, "bias": true, "padding_policy": .string("explicit"), "padding": 0, "padding_mode": .string("zeros")]
            .merging(changes) { _, new in new }
    }
    func testGroupedStrideDilationAndDepthwiseAnalyticalShapes() throws {
        let grouped = try Conv1dConfiguration(values: values(["stride": 2, "dilation": 2, "padding": 1]))
        XCTAssertEqual(grouped.parameterCount, 42)
        let result = try grouped.resolved(input: BatchedTensorLayout(sample: [4, 11]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 5])
        XCTAssertEqual(result.paddingLeft, 1); XCTAssertEqual(result.paddingRight, 1)
        XCTAssertEqual(result.multiplyAdds, 180)
        let depthwise = try Conv1dConfiguration(values: values(["out_channels": 8, "groups": 4, "bias": false]))
        XCTAssertEqual(depthwise.parameterCount, 24)
        XCTAssertEqual(try depthwise.resolved(input: BatchedTensorLayout(sample: [4, 9])).multiplyAdds, 168)
    }
    func testAsymmetricSameAndValidAcrossPaddingModes() throws {
        for mode in ["zeros", "reflect", "replicate", "circular"] {
            let same = try Conv1dConfiguration(values: values(["kernel_size": 4, "padding_policy": .string("same"), "padding_mode": .string(mode)]))
            let result = try same.resolved(input: BatchedTensorLayout(sample: [4, 9]))
            XCTAssertEqual(result.outputLayout.sample.dimensions, [6, 9])
            XCTAssertEqual(result.paddingLeft, 1); XCTAssertEqual(result.paddingRight, 2)
            let valid = try Conv1dConfiguration(values: values(["padding_policy": .string("valid"), "padding_mode": .string(mode)]))
            let noPad = try valid.resolved(input: BatchedTensorLayout(sample: [4, 9]))
            XCTAssertEqual(noPad.paddingLeft, 0); XCTAssertEqual(noPad.paddingRight, 0)
            XCTAssertEqual(noPad.outputLayout.sample.dimensions, [6, 7])
        }
    }
    func testStrictConfigurationAndCrossFieldBounds() throws {
        let invalid: [[String: JSONValue]] = [
            ["in_channels": true], ["out_channels": 513], ["kernel_size": 0], ["stride": .number(1.5)],
            ["dilation": .string("2")], ["groups": 3], ["bias": 1], ["padding": -1],
            ["padding_policy": .string("unknown")], ["padding_mode": .string("unknown")],
            ["padding_policy": .string("valid"), "padding": 1], ["padding_policy": .string("same"), "stride": 2],
            ["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": 8]
        ]
        for change in invalid { XCTAssertThrowsError(try Conv1dConfiguration(values: values(change)), "\(change)") }
        var missing = values(); missing.removeValue(forKey: "bias")
        XCTAssertThrowsError(try Conv1dConfiguration(values: missing))
        XCTAssertThrowsError(try Conv1dConfiguration(values: values(["extra": 1])))
        XCTAssertEqual(try Conv1dConfiguration(values: values(["stride": .number(2)])).values["stride"], .integer(2))
    }
    func testResolutionRejectsRankChannelsLengthPaddingOutputAndWorkOverflow() throws {
        let ordinary = try Conv1dConfiguration(values: values())
        for shape in [[4], [4, 2, 9], [3, 9], [4, 4097], [4, 2]] {
            XCTAssertThrowsError(try ordinary.resolved(input: BatchedTensorLayout(sample: shape)), "\(shape)")
        }
        // Negative numerator must refuse, rather than Swift truncation accidentally yielding one.
        XCTAssertThrowsError(try Conv1dConfiguration(values: values(["stride": 3])).resolved(input: BatchedTensorLayout(sample: [4, 2])))
        for (mode, pad) in [("reflect", 4), ("circular", 5)] {
            XCTAssertThrowsError(try Conv1dConfiguration(values: values(["padding_mode": .string(mode), "padding": .integer(Int64(pad))])).resolved(input: BatchedTensorLayout(sample: [4, 4])))
        }
        XCTAssertNoThrow(try Conv1dConfiguration(values: values(["padding_mode": .string("circular"), "padding": 4])).resolved(input: BatchedTensorLayout(sample: [4, 4])))
        XCTAssertThrowsError(try Conv1dConfiguration(values: values(["kernel_size": 4, "dilation": 4096, "padding_policy": .string("same")])).resolved(input: BatchedTensorLayout(sample: [4, 10])))
        XCTAssertThrowsError(try Conv1dConfiguration(values: values(["padding": 4096])).resolved(input: BatchedTensorLayout(sample: [4, 4096])))
        let heavy = try Conv1dConfiguration(values: values(["in_channels": 512, "out_channels": 512, "groups": 1, "kernel_size": 1]))
        XCTAssertThrowsError(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 512])))
        XCTAssertNoThrow(try heavy.resolved(input: BatchedTensorLayout(sample: [512, 488])))
    }
}
