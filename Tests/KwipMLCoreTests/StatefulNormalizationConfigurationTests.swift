import XCTest
@testable import KwipMLCore

final class StatefulNormalizationConfigurationTests: XCTestCase {
    func testDefaultsAndInactiveControlsRemainCanonical() throws {
        let batch = try StatefulNormalizationConfiguration(componentID: "batch_norm1d", values: [:])
        XCTAssertEqual(batch.parameterCount, 4)
        XCTAssertTrue(batch.tracksRunningStatistics)
        XCTAssertEqual(batch.momentum, 0.1)
        let instance = try StatefulNormalizationConfiguration(componentID: "instance_norm2d", values: [:])
        XCTAssertEqual(instance.parameterCount, 0)
        XCTAssertFalse(instance.tracksRunningStatistics)
        let cumulative = try StatefulNormalizationConfiguration(componentID: "batch_norm1d", values: ["averaging": .string("cumulative"), "momentum": 0.7, "affine": false, "bias": true])
        XCTAssertEqual(cumulative.values["momentum"], .number(0.7))
        XCTAssertEqual(cumulative.values["bias"], .bool(true))
        XCTAssertEqual(cumulative.parameterCount, 0)
        XCTAssertEqual(try StatefulNormalizationConfiguration(componentID: "batch_norm1d", values: cumulative.values), cumulative)
    }
    func testShapeIdentityRanksChannelsAndIndependentGroups() throws {
        for (id, shape) in [("batch_norm1d", [2]), ("batch_norm1d", [2, 4]), ("batch_norm2d", [2, 3, 4]), ("instance_norm1d", [2, 4]), ("instance_norm2d", [2, 3, 4]), ("group_norm", [2])] {
            let input = try BatchedTensorLayout(sample: shape)
            XCTAssertEqual(try StatefulNormalizationConfiguration(componentID: id, values: [:]).resolved(layout: input), input)
        }
        XCTAssertThrowsError(try StatefulNormalizationConfiguration(componentID: "instance_norm1d", values: [:]).resolved(layout: BatchedTensorLayout(sample: [3, 4])))
        XCTAssertThrowsError(try StatefulNormalizationConfiguration(componentID: "batch_norm2d", values: [:]).resolved(layout: BatchedTensorLayout(sample: [2, 4])))
        XCTAssertThrowsError(try StatefulNormalizationConfiguration(componentID: "group_norm", values: ["num_groups": 2]).resolved(layout: BatchedTensorLayout(sample: [2])))
        // A singleton current-stat batch is a runtime schedule check, not a guessed static batch.
        XCTAssertNoThrow(try StatefulNormalizationConfiguration(componentID: "batch_norm1d", values: [:]).resolved(layout: BatchedTensorLayout(sample: [2])))
    }
    func testStrictControlsAndNumericBoundaries() throws {
        let invalid: [[String: JSONValue]] = [["num_features": true], ["num_features": 2.5], ["num_features": 513], ["eps": 0], ["eps": .number(.infinity)], ["eps": .number(1e-40)], ["momentum": .null], ["momentum": 1.1], ["affine": 1], ["averaging": .string("other")], ["training": true]]
        for values in invalid {
            XCTAssertThrowsError(try StatefulNormalizationConfiguration(componentID: "batch_norm1d", values: values))
        }
        XCTAssertThrowsError(try StatefulNormalizationConfiguration(componentID: "group_norm", values: ["num_channels": 3, "num_groups": 2]))
        XCTAssertThrowsError(try StatefulNormalizationConfiguration(componentID: "instance_norm1d", values: ["averaging": .string("cumulative")]))
        for epsilon in [LayerNormConfiguration.epsilonRange.lowerBound, LayerNormConfiguration.epsilonRange.upperBound] {
            XCTAssertNoThrow(try StatefulNormalizationConfiguration(componentID: "group_norm", values: ["eps": .number(epsilon)]))
        }
    }
    func testBuffersAreASeparateRoundTrippableStateKind() throws {
        let state = try JSONDecoder().decode(StateKind.self, from: Data("\"buffers\"".utf8))
        XCTAssertEqual(String(data: try JSONEncoder().encode(state), encoding: .utf8), "\"buffers\"")
        XCTAssertNotEqual(state, .parameters)
        XCTAssertNotEqual(state, .fittedTransform)
    }
}
