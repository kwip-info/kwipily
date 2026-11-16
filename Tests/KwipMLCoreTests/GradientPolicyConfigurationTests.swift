import Foundation
import XCTest
@testable import KwipMLCore

final class GradientPolicyConfigurationTests: XCTestCase {
    private func input(_ node: String = "input") -> JSONValue { .object(["kind": .string("input"), "node": .string(node), "port": .string("out")]) }
    private func parameter(_ node: String = "first", _ name: String = "weight") -> JSONValue { .object(["kind": .string("parameter"), "node": .string(node), "parameter": .string(name)]) }
    private func request(_ id: String = "effect", targets: [JSONValue]? = nil, reduction: String = "sum", upstream: JSONValue = .null) -> JSONValue {
        .object(["id": .string(id), "output": .object(["node": .string("second"), "port": .string("hidden")]), "partition": .string("validation"), "row_start": 0, "row_count": 1, "reduction": .string(reduction), "upstream": upstream, "targets": .array(targets ?? [input(), parameter()]), "unavailable": .string("error")])
    }
    private func replacing(_ request: JSONValue, _ key: String, _ value: JSONValue?) -> JSONValue {
        guard case .object(var object) = request else { fatalError("test request") }
        object[key] = value; return .object(object)
    }
    private func config(_ requests: [JSONValue] = [], steps: JSONValue = 1, scale: JSONValue = .string("mean")) -> [String: JSONValue] {
        ["accumulation_steps": steps, "gradient_scale": scale, "requests": .array(requests)]
    }
    func testDefaultsNondefaultsAndOrderedRequestsRemainDistinct() throws {
        let defaults = try GradientPolicyConfiguration()
        XCTAssertEqual(defaults.values, config())
        XCTAssertEqual(defaults.accumulationSteps, 1)
        let r = [request("z-effect"), request("_first", targets: [parameter("other", "nested.weight")])]
        let chosen = try GradientPolicyConfiguration(values: config(r, steps: 32, scale: .string("sum")))
        XCTAssertEqual(chosen.values, config(r, steps: 32, scale: .string("sum")))
        XCTAssertEqual(chosen.requests, r)
        XCTAssertEqual(chosen.accumulationSteps, 32)
        XCTAssertEqual(chosen.gradientScale, "sum")
        XCTAssertNotEqual(chosen, defaults)
        XCTAssertEqual(try GradientPolicyConfiguration(values: ["accumulation_steps": 2]).values, config([], steps: 2))
    }
    func testTopLevelExactTypesAndRangesRefuse() throws {
        for value in [JSONValue.bool(true), .number(1), .integer(0), .integer(33), .null, .string("2")] {
            XCTAssertThrowsError(try GradientPolicyConfiguration(values: config(steps: value)))
        }
        for scale in [JSONValue.null, .bool(false), .integer(1), .string("Mean"), .string("native")] {
            XCTAssertThrowsError(try GradientPolicyConfiguration(values: config(scale: scale)))
        }
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: ["callback": .string("code")]))
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: ["requests": .null]))
        XCTAssertThrowsError(try GradientPolicyConfiguration(componentID: "linear"))
    }
    func testRequestSelectorIdentityAndConditionalKeysAreStrict() throws {
        let base = request()
        guard case .object(let fields) = base else { return XCTFail("fixture") }
        for key in fields.keys { XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([replacing(base, key, nil)])), key) }
        let invalid: [(String, JSONValue)] = [
            ("id", .string("")), ("id", .string("path/id")), ("id", .string("a.b")), ("id", .string("é")), ("id", .string(String(repeating: "a", count: 65))),
            ("output", .object(["node": .string(""), "port": .string("out")])), ("output", .object(["node": .string("x"), "port": .string("out"), "shape": .array([2])])),
            ("partition", .string("test")), ("row_start", -1), ("row_start", .number(0)), ("row_count", 0), ("row_count", 33), ("row_count", true),
            ("reduction", .string("none")), ("upstream", .object([:])), ("targets", .array([])), ("unavailable", .string("zero")), ("extra", false),
        ]
        for (key, value) in invalid { XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([replacing(base, key, value)])), key) }
        for target in [input(""), parameter("first", ""), parameter("bad\0node"), .object(["kind": .string("intermediate"), "node": .string("hidden"), "port": .string("out")]), .object(["kind": .string("input"), "node": .string("input"), "parameter": .string("weight")])] {
            XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request(targets: [target])])))
        }
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request(), request()])), "duplicate request identity")
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request(targets: [input(), input()])])), "duplicate target")
        _ = try GradientPolicyConfiguration(values: config([request("-"), request("_", targets: [input()])]))
        _ = try GradientPolicyConfiguration(values: config([replacing(base, "row_start", .integer(Int64.max - 1))]))
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([replacing(base, "row_start", .integer(Int64.max))])), "checked row range overflow")
    }
    func testGlobalRequestTargetLimitsAndLengthQualifiedIdentity() throws {
        XCTAssertEqual(try GradientPolicyConfiguration(values: config((0..<32).map { request("r\($0)", targets: [input()]) })).requests.count, 32)
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config((0..<33).map { request("r\($0)", targets: [input()]) })))
        let targets = (0..<32).map { parameter("node\($0)") }
        XCTAssertEqual(try GradientPolicyConfiguration(values: config([request(targets: targets)])).requests.count, 1)
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request(targets: targets + [parameter("extra")])])))
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request("a", targets: targets), request("b", targets: [parameter("extra")])])), "global unique count")
        _ = try GradientPolicyConfiguration(values: config([request(targets: [parameter("a:b", "c"), parameter("a", "b:c")])]))
    }
    func testUpstreamExactGeometryFiniteFloat32AndPortableBudget() throws {
        let upstream: JSONValue = .object(["dtype": .string("float32"), "shape": .array([1, 2]), "values": .array([.integer(0), .number(-0.5)])])
        let good = request(reduction: "upstream", upstream: upstream)
        let chosen = try GradientPolicyConfiguration(values: config([good]))
        guard case .object(let saved)? = chosen.requests.first, case .object(let vector) = saved["upstream"] else { return XCTFail("upstream retained") }
        XCTAssertEqual(vector["values"], .array([.number(0), .number(-0.5)]))
        _ = try GradientPolicyConfiguration(values: config([request(reduction: "upstream", upstream: .object(["dtype": .string("float32"), "shape": .array([1]), "values": .array([.number(Double(Float.leastNonzeroMagnitude))])]))]))
        let bad: [JSONValue] = [.null, .object(["dtype": .string("float64"), "shape": .array([1]), "values": .array([1])]), .object(["dtype": .string("float32"), "shape": .array([1, 2]), "values": .array([1])]), .object(["dtype": .string("float32"), "shape": .array([0]), "values": .array([])]), .object(["dtype": .string("float32"), "shape": .array([.number(1)]), "values": .array([1])])]
        for value in bad { XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request(reduction: "upstream", upstream: value)]))) }
        for scalar in [JSONValue.bool(false), .number(.nan), .number(.infinity), .number(1e39), .number(1e-50)] {
            XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([request(reduction: "upstream", upstream: .object(["dtype": .string("float32"), "shape": .array([1]), "values": .array([scalar])]))])))
        }
        let huge = request(targets: [parameter(String(repeating: "界", count: 12_000))])
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([huge])))
        let tooMany: JSONValue = .object(["dtype": .string("float32"), "shape": .array([32, 4096]), "values": .array(Array(repeating: 0, count: 131_072))])
        XCTAssertThrowsError(try GradientPolicyConfiguration(values: config([replacing(request(reduction: "upstream", upstream: tooMany), "row_count", 32)])))
        XCTAssertEqual(GradientPolicyConfiguration.maximumDerivativeElements, 65_536)
    }
    func testSelectedObjectiveCompatibilityPreservesReductionMeaning() throws {
        let mean = try GradientPolicyConfiguration()
        for reduction in ["mean", "batchmean"] { XCTAssertNoThrow(try mean.validateObjectiveCompatibility(reduction: reduction)) }
        XCTAssertNoThrow(try mean.validateObjectiveCompatibility(reduction: "none", scalarization: "mean"))
        XCTAssertThrowsError(try mean.validateObjectiveCompatibility(reduction: "sum"))
        XCTAssertThrowsError(try mean.validateObjectiveCompatibility(reduction: "none", scalarization: "sum"))
        let sum = try GradientPolicyConfiguration(values: ["gradient_scale": .string("sum")])
        XCTAssertNoThrow(try sum.validateObjectiveCompatibility(reduction: "sum"))
        XCTAssertNoThrow(try sum.validateObjectiveCompatibility(reduction: "none", scalarization: "sum"))
        for reduction in ["mean", "batchmean", "invalid"] { XCTAssertThrowsError(try sum.validateObjectiveCompatibility(reduction: reduction)) }
        XCTAssertThrowsError(try sum.validateObjectiveCompatibility(reduction: "none", scalarization: "mean"))
    }
    func testStrictCompleteRecipeAndCanonicalRoundTrip() throws {
        let c = try GradientPolicyConfiguration(values: config([request()], steps: 3))
        let data = try JSONEncoder().encode(c.recipeValue)
        XCTAssertEqual(try GradientPolicyConfiguration.decodeRecipe(JSONDecoder().decode(JSONValue.self, from: data)), c)
        let base: [String: JSONValue] = ["schemaVersion": 1, "configurationVersion": 1, "componentID": .string("gradient_policy"), "config": .object(c.values)]
        for key in base.keys { var bad = base; bad.removeValue(forKey: key); XCTAssertThrowsError(try GradientPolicyConfiguration.decodeRecipe(.object(bad))) }
        for key in ["schemaVersion", "configurationVersion"] { for value in [JSONValue.number(1), .bool(true), .integer(2)] { var bad = base; bad[key] = value; XCTAssertThrowsError(try GradientPolicyConfiguration.decodeRecipe(.object(bad))) } }
        var incomplete = base; incomplete["config"] = .object([:]); XCTAssertThrowsError(try GradientPolicyConfiguration.decodeRecipe(.object(incomplete)))
        var extra = base; extra["source"] = .object([:]); XCTAssertThrowsError(try GradientPolicyConfiguration.decodeRecipe(.object(extra)))
    }
}
