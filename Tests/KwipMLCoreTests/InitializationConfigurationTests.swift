import Foundation
import XCTest
@testable import KwipMLCore

final class InitializationConfigurationTests: XCTestCase {
    private func rule(_ algorithm: String, node: String = "dense", parameter: String = "weight", settings: [String: JSONValue] = [:]) -> JSONValue {
        .object(["owner": .object(["node": .string(node), "parameter": .string(parameter)]), "algorithm": .string(algorithm), "settings": .object(settings)])
    }
    private func config(_ rules: [JSONValue], seed: JSONValue = 42) -> [String: JSONValue] {
        ["seed": seed, "rules": .array(rules)]
    }
    private func envelope(_ config: [String: JSONValue]) -> JSONValue {
        .object(["schemaVersion": 1, "configurationVersion": 1, "componentID": .string("initialization"), "config": .object(config)])
    }
    func testAllAlgorithmsCanonicalControlsAndExactOwnerOrder() throws {
        let c = try InitializationConfiguration(values: config([
            rule("zeros", node: "z", parameter: "bias"),
            rule("normal", node: "a", parameter: "nested.bias", settings: ["mean": -2, "std": 0.5]),
            rule("kaiming_uniform", node: "a", settings: ["a": 0.2, "mode": .string("fan_out"), "nonlinearity": .string("leaky_relu")]),
            rule("xavier_uniform", node: "b", settings: ["gain": 2]),
        ], seed: .integer(Int64.max)))
        XCTAssertEqual(c.componentID, "initialization")
        XCTAssertEqual(c.seed, Int64.max)
        XCTAssertEqual(c.rules.map { $0.owner.node + "/" + $0.owner.parameter }, ["a/nested.bias", "a/weight", "b/weight", "z/bias"])
        XCTAssertEqual(c.rules[0].settings, ["mean": .number(-2), "std": .number(0.5)])
        XCTAssertEqual(c.rules[1].settings["mode"], .string("fan_out"))
        XCTAssertEqual(c.rules[2].settings["gain"], .number(2))
        XCTAssertEqual(c.rules[3].settings, [:])
        let reordered = try InitializationConfiguration(values: config([
            rule("xavier_uniform", node: "b", settings: ["gain": 2]),
            rule("zeros", node: "z", parameter: "bias"),
            rule("kaiming_uniform", node: "a", settings: ["a": 0.2, "mode": .string("fan_out"), "nonlinearity": .string("leaky_relu")]),
            rule("normal", node: "a", parameter: "nested.bias", settings: ["mean": -2, "std": 0.5]),
        ], seed: .integer(Int64.max)))
        XCTAssertEqual(c, reordered)
    }
    func testAuthoringDefaultsAreAlgorithmSpecificAndZeroIsDeliberate() throws {
        let c = try InitializationConfiguration(values: config([
            rule("xavier_uniform", node: "a"), rule("kaiming_uniform", node: "b"), rule("normal", node: "c"), rule("zeros", node: "d"),
        ], seed: 0))
        XCTAssertEqual(c.rules[0].settings, ["gain": .number(1)])
        XCTAssertEqual(c.rules[1].settings, ["a": .number(0), "mode": .string("fan_in"), "nonlinearity": .string("leaky_relu")])
        XCTAssertEqual(c.rules[2].settings, ["mean": .number(0), "std": .number(1)])
        let zeros = try InitializationConfiguration(values: config([rule("xavier_uniform", settings: ["gain": 0]), rule("normal", node: "other", settings: ["mean": -1_000_000, "std": 0])]))
        XCTAssertEqual(zeros.rules[0].settings["gain"], .number(0))
        XCTAssertEqual(zeros.rules[1].settings["std"], .number(0))
    }
    func testUnknownKeysTypesOwnersAndDuplicateSelectionsRefuse() throws {
        _ = try InitializationConfiguration(values: config([rule("zeros")]))
        for seed in [JSONValue.bool(true), .number(42), .number(0.5), .integer(-1), .null, .string("42")] {
            XCTAssertThrowsError(try InitializationConfiguration(values: config([rule("zeros")], seed: seed)))
        }
        XCTAssertThrowsError(try InitializationConfiguration(componentID: "linear", values: config([rule("zeros")])))
        var unknown = config([rule("zeros")]); unknown["callback"] = .string("code")
        XCTAssertThrowsError(try InitializationConfiguration(values: unknown))
        for bad in [rule("ones"), rule("zeros", settings: ["gain": 1]), rule("normal", settings: ["mode": .string("fan_in")]), rule("kaiming_uniform", settings: ["mode": .string("automatic")]), rule("kaiming_uniform", settings: ["nonlinearity": .string("tanh")]), rule("zeros", node: ""), rule("zeros", parameter: ""), rule("zeros", node: "bad\0node"), rule("zeros", parameter: "bad\0name")] {
            XCTAssertThrowsError(try InitializationConfiguration(values: config([bad])))
        }
        XCTAssertThrowsError(try InitializationConfiguration(values: config([rule("zeros"), rule("normal")])))
        var badOwner: [String: JSONValue] = ["owner": JSONValue.object(["node": .string("dense"), "parameter": .string("weight"), "domain": .string("outside")]), "algorithm": .string("zeros"), "settings": .object([:])]
        XCTAssertThrowsError(try InitializationConfiguration(values: config([.object(badOwner)])))
        badOwner["owner"] = .object(["node": .string("dense"), "parameter": .string("weight")]); badOwner["extra"] = true
        XCTAssertThrowsError(try InitializationConfiguration(values: config([.object(badOwner)])))
    }
    func testFiniteInclusiveBoundsAndFloat32Representability() throws {
        _ = try InitializationConfiguration(values: config([rule("normal", settings: ["mean": 1_000_000, "std": 1_000_000]), rule("xavier_uniform", node: "a", settings: ["gain": 100]), rule("kaiming_uniform", node: "b", settings: ["a": 100, "nonlinearity": .string("relu")])]))
        let invalid: [(String, String, JSONValue)] = [
            ("xavier_uniform", "gain", -0.01), ("xavier_uniform", "gain", 100.01),
            ("kaiming_uniform", "a", -1), ("kaiming_uniform", "a", 101),
            ("normal", "mean", -1_000_001), ("normal", "mean", 1_000_001), ("normal", "std", -1), ("normal", "std", 1_000_001),
            ("normal", "std", .number(1e-50)), ("normal", "mean", .number(-1e-50)),
            ("xavier_uniform", "gain", .number(.infinity)), ("normal", "mean", .number(.nan)), ("normal", "std", true),
        ]
        for (algorithm, key, value) in invalid {
            XCTAssertThrowsError(try InitializationConfiguration(values: config([rule(algorithm, settings: [key: value])])), "\(algorithm).\(key)")
        }
        _ = try InitializationConfiguration(values: config([rule("normal", settings: ["std": .number(Double(Float.leastNonzeroMagnitude))])]))
    }
    func testExactRecipeDecoderAndCanonicalRoundTrip() throws {
        let c = try InitializationConfiguration(values: config([rule("normal", settings: ["mean": 2, "std": 0]), rule("zeros", node: "b")]))
        let data = try JSONEncoder().encode(c.recipeValue)
        XCTAssertEqual(try InitializationConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self, from: data)), c)
        var value: [String: JSONValue] = ["schemaVersion": 1, "configurationVersion": 1, "componentID": .string("initialization"), "config": .object(c.values)]
        for key in value.keys {
            var missing = value; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try InitializationConfiguration.decodeRecipe(.object(missing)))
        }
        value["extra"] = false
        XCTAssertThrowsError(try InitializationConfiguration.decodeRecipe(.object(value)))
        XCTAssertThrowsError(try InitializationConfiguration.decodeRecipe(envelope(config([rule("normal")]))), "Wire settings must be complete, not filled silently.")
        for key in ["schemaVersion", "configurationVersion"] {
            var bad: [String: JSONValue] = ["schemaVersion": 1, "configurationVersion": 1, "componentID": .string("initialization"), "config": .object(c.values)]
            bad[key] = .number(1)
            XCTAssertThrowsError(try InitializationConfiguration.decodeRecipe(.object(bad)))
            bad[key] = 2
            XCTAssertThrowsError(try InitializationConfiguration.decodeRecipe(.object(bad)))
        }
    }
    func testRuleCountAndCanonicalWireByteBudget() throws {
        let maximum = (0..<128).map { rule("zeros", node: "node-\($0)") }
        XCTAssertEqual(try InitializationConfiguration(values: config(maximum)).rules.count, 128)
        XCTAssertThrowsError(try InitializationConfiguration(values: config([])))
        XCTAssertThrowsError(try InitializationConfiguration(values: config(maximum + [rule("zeros", node: "extra")])))
        // Owner identity remains exact; no truncation to fit the wire budget.
        XCTAssertThrowsError(try InitializationConfiguration(values: config([rule("zeros", node: String(repeating: "x", count: 32_768))])))
        XCTAssertEqual(InitializationConfiguration.maximumRecipeBytes, 32_768)
    }
    func testPortableIntegralRealBudgetAndUnicodeOwnerBytes() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let small = try InitializationConfiguration(values: config([rule("xavier_uniform", node: "x")]))
        let fixedBytes = try encoder.encode(small.recipeValue).count - 1
        // One integral real gain reserves Python's two-byte .0 suffix.
        let exactNode = String(repeating: "x", count: 32_768 - fixedBytes - 2)
        let edge = try InitializationConfiguration(values: config([rule("xavier_uniform", node: exactNode)]))
        XCTAssertEqual(try encoder.encode(edge.recipeValue).count + 2, 32_768)
        XCTAssertThrowsError(try InitializationConfiguration(values: config([rule("xavier_uniform", node: exactNode + "x")])))
        XCTAssertThrowsError(try InitializationConfiguration.decodeRecipe(envelope(config([rule("xavier_uniform", node: exactNode + "x", settings: ["gain": 1])]))))
        let unicode = try InitializationConfiguration(values: config([rule("zeros", node: "Block/é/模型", parameter: "nested.weight")]))
        XCTAssertEqual(unicode.rules[0].owner.node, "Block/é/模型")
        XCTAssertEqual(try InitializationConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self, from: encoder.encode(unicode.recipeValue))), unicode)
        XCTAssertThrowsError(try InitializationConfiguration(values: config([rule("zeros", node: String(repeating: "é", count: 16_384))])))
    }

}
