import XCTest
@testable import KwipMLCore

final class LazyLinearRuntimeCapabilityTests: XCTestCase {
    private var descriptor: [String: Any] { [
        "version": ["major": 1, "minor": 0, "patch": 0],
        "adapterID": "kwip.runner.graph-v2.lazy-linear", "graphSchemaVersions": [2],
        "maximumRank": 8, "preservesBatch": true, "minimumSampleRank": 1,
        "maximumSampleElements": 100_000_000, "maximumParameters": 2_000_000,
        "inputFeatureMinimum": 1, "inputFeatureMaximum": 4096,
        "outputFeatureMinimum": 1, "outputFeatureMaximum": 512,
        "biasModes": [false, true], "graphDTypes": ["float32"],
        "inputWidthInferred": true, "materializationPolicy": "validatedShapeBeforeOptimizer"
    ] }
    func testReviewedContractAccepted() throws {
        let bytes = try JSONSerialization.data(withJSONObject: descriptor)
        XCTAssertEqual(try LazyLinearRuntimeCapability.decode(bytes), try LazyLinearRuntimeCapability.decode(bytes))
    }
    func testMissingUnknownChangedAndWrongTypeFieldsRejected() throws {
        for key in descriptor.keys {
            var missing = descriptor; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try LazyLinearRuntimeCapability.decode(JSONSerialization.data(withJSONObject: missing)), key)
        }
        let changes: [(String, Any)] = [
            ("extra", true), ("inputWidthInferred", 1), ("preservesBatch", false),
            ("maximumRank", true), ("graphSchemaVersions", [1,2]), ("graphDTypes", ["float64"]),
            ("inputFeatureMaximum", 512), ("outputFeatureMaximum", 4096),
            ("maximumParameters", 3_000_000), ("biasModes", [true]),
            ("materializationPolicy", "firstForward"), ("version", ["major":1,"minor":1,"patch":0]),
            ("version", ["major":1,"minor":0,"patch":0,"extra":0])
        ]
        for (key,value) in changes {
            var changed = descriptor; changed[key] = value
            XCTAssertThrowsError(try LazyLinearRuntimeCapability.decode(JSONSerialization.data(withJSONObject: changed)), key)
        }
    }
    func testOversizedPayloadRefusedEvenIfJSONOtherwiseValid() throws {
        var bytes = try JSONSerialization.data(withJSONObject: descriptor)
        bytes.append(Data(repeating: 32, count: 4096))
        XCTAssertThrowsError(try LazyLinearRuntimeCapability.decode(bytes))
    }
}
