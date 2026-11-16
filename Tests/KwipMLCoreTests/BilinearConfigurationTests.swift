import XCTest
@testable import KwipMLCore

final class BilinearConfigurationTests: XCTestCase {
    private var values: [String: JSONValue] {
        ["in1_features": 3, "in2_features": 5, "out_features": 2, "bias": true]
    }
    func testIndependentWidthsCanonicalConfigurationAndParameterCount() throws {
        var raw = values; raw["in1_features"] = .number(3)
        let config = try BilinearConfiguration(values: raw)
        XCTAssertEqual(config.input1Features, 3); XCTAssertEqual(config.input2Features, 5)
        XCTAssertEqual(config.outputFeatures, 2); XCTAssertTrue(config.bias)
        XCTAssertEqual(config.parameterCount, 32)
        XCTAssertEqual(config.values, values)
        raw["bias"] = false
        XCTAssertEqual(try BilinearConfiguration(values: raw).parameterCount, 30)
    }
    func testShapesPreserveAllSamplePrefixesAndNeverBroadcast() throws {
        let config = try BilinearConfiguration(values: values)
        for prefix in [[], [4], [2, 4], [1, 2, 1, 2, 1, 2]] {
            let left = try BatchedTensorLayout(sample: prefix + [3])
            let right = try BatchedTensorLayout(sample: prefix + [5])
            XCTAssertEqual(try config.outputLayout(first: left, second: right), try BatchedTensorLayout(sample: prefix + [2]))
            XCTAssertEqual(try left.bilinear(with: right, input1Features: 3, input2Features: 5, outputFeatures: 2).displayDimensions, ["N"] + (prefix + [2]).map(String.init))
        }
        for (a, b) in [([1, 3], [4, 5]), ([4, 3], [1, 5]), ([3], [1, 5]), ([2, 2, 3], [4, 5]), ([4], [5]), ([3], [4])] {
            XCTAssertThrowsError(try config.outputLayout(first: BatchedTensorLayout(sample: a), second: BatchedTensorLayout(sample: b)))
        }
    }
    func testMissingUnknownNonintegerAndNonbooleanFieldsRefuse() {
        for key in values.keys {
            var raw = values; raw.removeValue(forKey: key)
            XCTAssertThrowsError(try BilinearConfiguration(values: raw))
        }
        let mutations: [(String, JSONValue)] = [
            ("in1_features", 0), ("in2_features", -1), ("out_features", 513),
            ("in1_features", .number(2.5)), ("in2_features", true), ("out_features", .integer(Int64.max)),
            ("out_features", .number(.infinity)), ("in1_features", .string("3")),
            ("bias", 1), ("bias", .null), ("extra", false)
        ]
        for (key, value) in mutations {
            var raw = values; raw[key] = value
            XCTAssertThrowsError(try BilinearConfiguration(values: raw), key)
        }
    }
    func testParameterAndOutputAllocationBoundsRefuseBeforeConstruction() throws {
        let boundary: [String: JSONValue] = ["in1_features": 500, "in2_features": 400, "out_features": 10, "bias": false]
        XCTAssertEqual(try BilinearConfiguration(values: boundary).parameterCount, 2_000_000)
        var tooMany = boundary; tooMany["bias"] = true
        XCTAssertThrowsError(try BilinearConfiguration(values: tooMany))
        let config = try BilinearConfiguration(values: ["in1_features": 1, "in2_features": 1, "out_features": 512, "bias": true])
        XCTAssertThrowsError(try config.outputLayout(first: BatchedTensorLayout(sample: [1_000_000, 1]), second: BatchedTensorLayout(sample: [1_000_000, 1])))
    }
}
