import XCTest
@testable import KwipMLCore

final class Conv1dDiscoveryTests: XCTestCase {
    static var descriptor: [String: Any] {
        ["version": ["major": 1, "minor": 0, "patch": 0], "adapterID": "kwip.runner.graph-v2.conv1d",
         "graphSchemaVersions": [2], "sampleRank": 2, "preservesBatch": true, "graphDTypes": ["float32"],
         "channelMinimum": 1, "channelMaximum": 512, "lengthMinimum": 1, "lengthMaximum": 4096,
         "kernelMinimum": 1, "kernelMaximum": 4096, "strideMinimum": 1, "strideMaximum": 4096,
         "dilationMinimum": 1, "dilationMaximum": 4096, "groupMinimum": 1, "groupMaximum": 512,
         "paddingMinimum": 0, "paddingMaximum": 4096, "maximumParameters": 2_000_000,
         "maximumMultiplyAdds": 128_000_000, "paddingPolicies": ["explicit", "valid", "same"],
         "paddingModes": ["zeros", "reflect", "replicate", "circular"], "biasModes": [false, true]]
    }
    func testExactDescriptorRejectsMissingExtraAndChangedFields() throws {
        XCTAssertNoThrow(try Conv1dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: Self.descriptor)))
        for key in Self.descriptor.keys {
            var missing = Self.descriptor; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try Conv1dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: missing)), key)
        }
        let changes: [(String, Any)] = [("extra", true), ("sampleRank", 3), ("preservesBatch", 1), ("maximumMultiplyAdds", 129_000_000)]
        for (key, value) in changes {
            var changed = Self.descriptor; changed[key] = value
            XCTAssertThrowsError(try Conv1dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: changed)))
        }
    }
    func testSharedRegistryAndBridgeDescribeChannelsFirstLearnedState() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "conv1d"), Conv1dComponent.definition)
        let package = try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1")
        let value = try XCTUnwrap(package.capabilities.first { $0.id == "conv1d" })
        XCTAssertEqual(value.adapterID, "kwip.runner.graph-v2.conv1d")
        XCTAssertEqual(value.ports.map(\.id), ["in", "out"])
        XCTAssertEqual(value.ports[0].schema.tensor?.dimensions, [.parameter("in_channels"), .symbol("inputLength")])
        XCTAssertEqual(value.ports[1].schema.tensor?.dimensions, [.parameter("out_channels"), .symbol("outputLength")])
        for port in value.ports {
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.minimum, 1)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.maximum, 1)
            XCTAssertEqual(port.schema.tensor?.dtype, .float32)
        }
        XCTAssertEqual(value.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        XCTAssertEqual(value.learning.lessonIDs, ["assessment.conv1d-shared-kernel"]); XCTAssertEqual(value.learning.exampleIDs, ["prepared-conv1d-classifier"])
        var mutated = registry
        let index = try XCTUnwrap(mutated.components.firstIndex { $0.id == "conv1d" })
        mutated.components[index].fields[0].max = 1024
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry: mutated, runtimeVersion: "2.14.1"))
    }
}
