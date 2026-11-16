import Foundation
import XCTest
@testable import KwipMLCore

final class DerivativeCapturePolicyTests: XCTestCase {
    private func data(_ text: String) -> Data { Data(text.utf8) }
    private func selected() throws -> GradientPolicyConfiguration {
        try GradientPolicyConfiguration(values: ["requests": .array([.object([
            "id": .string("influence"), "output": .object(["node": .string("result"), "port": .string("out")]),
            "partition": .string("validation"), "row_start": 0, "row_count": 1,
            "reduction": .string("sum"), "upstream": .null,
            "targets": .array([.object(["kind": .string("input"), "node": .string("input"), "port": .string("out")])]),
            "unavailable": .string("error")])])])
    }
    func testDefaultFalseExplicitValuesAndCanonicalRoundTrips() throws {
        XCTAssertEqual(DerivativeCapturePolicy(), DerivativeCapturePolicy.factory)
        XCTAssertFalse(DerivativeCapturePolicy.factory.values)
        XCTAssertEqual(DerivativeCapturePolicy.maximumWireBytes, 256)
        for allowed in [false, true] {
            let policy = DerivativeCapturePolicy(values: allowed)
            XCTAssertEqual(policy.schemaVersion, 1)
            XCTAssertEqual(policy.jsonValue, .object(["schemaVersion": 1, "values": .bool(allowed)]))
            XCTAssertEqual(try DerivativeCapturePolicy(jsonValue: policy.jsonValue), policy)
            let wire = try policy.wireData()
            XCTAssertLessThanOrEqual(wire.count, 256)
            XCTAssertEqual(try DerivativeCapturePolicy.decode(wire), policy)
            XCTAssertEqual(try JSONDecoder().decode(DerivativeCapturePolicy.self, from: JSONEncoder().encode(policy)), policy)
        }
    }
    func testExactJSONValueTypesKeysAndVersionsRefuse() {
        let bad: [JSONValue] = [
            .null, .array([]), .bool(false), .object([:]), .object(["schemaVersion": 1]),
            .object(["schemaVersion": .number(1), "values": false]),
            .object(["schemaVersion": true, "values": false]),
            .object(["schemaVersion": 2, "values": false]),
            .object(["schemaVersion": 1, "values": 1]),
            .object(["schemaVersion": 1, "values": .string("false")]),
            .object(["schemaVersion": 1, "values": .null]),
            .object(["schemaVersion": 1, "values": false, "activationValues": true]),
            .object(["schemaVersion": 1, "values": false, "predictionRows": true])]
        for value in bad { XCTAssertThrowsError(try DerivativeCapturePolicy(jsonValue: value), "\(value)") }
    }
    func testBoundedWireStrictNumericSpellingDuplicatesAndMalformedValues() throws {
        XCTAssertEqual(try DerivativeCapturePolicy.decode(data(" { \"values\": true, \"schemaVersion\": 1 } \n")), DerivativeCapturePolicy(values: true))
        for text in ["null", "[]", "{}", "{\"schemaVersion\":1.0,\"values\":false}", "{\"schemaVersion\":1e0,\"values\":false}", "{\"schemaVersion\":true,\"values\":false}", "{\"schemaVersion\":2,\"values\":false}", "{\"schemaVersion\":1,\"values\":0}", "{\"schemaVersion\":1,\"values\":false,\"values\":true}", "{\"schemaVersion\":1,\"values\":false,\"extra\":false}", "{\"schemaVersion\":1}", "{\"values\":true}", "{\"schemaVersion\":1,\"values\":false}garbage"] {
            XCTAssertThrowsError(try DerivativeCapturePolicy.decode(data(text)), text)
        }
        let canonical = "{\"schemaVersion\":1,\"values\":false}"
        XCTAssertEqual(try DerivativeCapturePolicy.decode(data(canonical + String(repeating: " ", count: 256 - canonical.utf8.count))), .factory)
        XCTAssertThrowsError(try DerivativeCapturePolicy.decode(data(canonical + String(repeating: " ", count: 257 - canonical.utf8.count))))
        XCTAssertThrowsError(try DerivativeCapturePolicy.decode(Data([0xff, 0xfe])))
        for text in ["{\"schemaVersion\":1,\"values\":false,\"unknown\":true}", "{\"schemaVersion\":true,\"values\":false}", "{\"schemaVersion\":1,\"values\":1}"] {
            XCTAssertThrowsError(try JSONDecoder().decode(DerivativeCapturePolicy.self, from: data(text)))
        }
    }
    func testOnlySelectedRequestsRequireIndependentValuesPermission() throws {
        let request = try selected(), empty = try GradientPolicyConfiguration()
        XCTAssertNoThrow(try DerivativeCapturePolicy(values: false).validate(gradientPolicy: empty))
        XCTAssertNoThrow(try DerivativeCapturePolicy(values: true).validate(gradientPolicy: empty))
        XCTAssertNoThrow(try DerivativeCapturePolicy(values: true).validate(gradientPolicy: request))
        XCTAssertThrowsError(try DerivativeCapturePolicy.factory.validate(gradientPolicy: request)) { error in
            XCTAssertTrue(String(describing: error).localizedCaseInsensitiveContains("enable"))
            XCTAssertTrue(String(describing: error).localizedCaseInsensitiveContains("capture"))
        }
        XCTAssertFalse(DerivativeCapturePolicy.factory.values, "Other sample/activation/gradient summary permissions cannot grant this separate policy.")
    }
}
