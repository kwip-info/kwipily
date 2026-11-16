import XCTest
@testable import KwipMLCore

final class LazyLinearConfigurationTests: XCTestCase {
    func testResolutionInfersTrailingWidthWithoutChangingSavedConfiguration() throws {
        let values: [String: JSONValue] = ["out_features": 7, "bias": true]
        let configuration = try LazyLinearConfiguration(values: values)
        let resolved = try configuration.resolved(input: BatchedTensorLayout(sample: [2, 3, 5]))
        XCTAssertEqual(configuration.values, values)
        XCTAssertEqual(configuration.outputFeatures, 7)
        XCTAssertTrue(configuration.bias)
        XCTAssertEqual(resolved.inputFeatures, 5)
        XCTAssertEqual(resolved.outputFeatures, 7)
        XCTAssertTrue(resolved.bias)
        XCTAssertEqual(resolved.parameterCount, 42)
        XCTAssertEqual(resolved.outputLayout, try BatchedTensorLayout(sample: [2, 3, 7]))
        let different = try configuration.resolved(input: BatchedTensorLayout(sample: [11]))
        XCTAssertEqual(different.inputFeatures, 11)
        XCTAssertEqual(configuration.values, values)
        XCTAssertNil(configuration.values["in_features"])
    }

    func testParameterLimitAndInferredWidthBoundBeforeAllocation() throws {
        let large = try BatchedTensorLayout(sample: [4096])
        let accepted = try LazyLinearConfiguration(values: ["out_features": 488, "bias": true]).resolved(input: large)
        XCTAssertEqual(accepted.parameterCount, 1_999_336)
        XCTAssertThrowsError(try LazyLinearConfiguration(values: ["out_features": 489, "bias": true]).resolved(input: large))
        XCTAssertThrowsError(try LazyLinearConfiguration(values: ["out_features": 1, "bias": false]).resolved(input: BatchedTensorLayout(sample: [4097])))
        let minimum = try LazyLinearConfiguration(values: ["out_features": 1, "bias": false]).resolved(input: BatchedTensorLayout(sample: [1]))
        XCTAssertEqual(minimum.parameterCount, 1)
        XCTAssertFalse(minimum.bias)
    }

    func testStrictFieldsIntegralBoundsAndTrueBooleanBias() throws {
        let invalid: [[String: JSONValue]] = [
            [:], ["out_features": 2], ["out_features": 2, "bias": true, "in_features": 3],
            ["out_features": 0, "bias": true], ["out_features": 513, "bias": true],
            ["out_features": .number(2.5), "bias": true], ["out_features": true, "bias": true],
            ["out_features": .string("2"), "bias": true], ["out_features": 2, "bias": 1],
            ["out_features": 2, "bias": .null]
        ]
        for values in invalid {
            XCTAssertThrowsError(try LazyLinearConfiguration(values: values), "\(values)")
        }
        XCTAssertEqual(try LazyLinearConfiguration(values: ["out_features": .number(512), "bias": false]).values,
                       ["out_features": .integer(512), "bias": false])
    }
}
