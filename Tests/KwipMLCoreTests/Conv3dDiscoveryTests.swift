import XCTest
@testable import KwipMLCore

final class Conv3dDiscoveryTests: XCTestCase {
    static var descriptor: [String: Any] {
        ["version": ["major": 1, "minor": 0, "patch": 0], "adapterID": "kwip.runner.graph-v2.conv3d",
         "graphSchemaVersions": [2], "sampleRank": 4, "preservesBatch": true, "graphDTypes": ["float32"],
         "spatialParameterLength": 3, "maximumSampleElements": 100_000_000, "maximumPaddedSampleElements": 100_000_000,
         "channelMinimum": 1, "channelMaximum": 512, "lengthMinimum": 1, "lengthMaximum": 4096,
         "kernelMinimum": 1, "kernelMaximum": 4096, "strideMinimum": 1, "strideMaximum": 4096,
         "dilationMinimum": 1, "dilationMaximum": 4096, "groupMinimum": 1, "groupMaximum": 512,
         "paddingMinimum": 0, "paddingMaximum": 4096, "maximumParameters": 2_000_000,
         "maximumMultiplyAdds": 128_000_000, "paddingPolicies": ["explicit", "valid", "same"],
         "paddingModes": ["zeros", "reflect", "replicate", "circular"], "biasModes": [false, true]]
    }
    func testExactDescriptorRejectsMissingExtraAndChangedFields() throws {
        XCTAssertNoThrow(try Conv3dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: Self.descriptor)))
        for key in Self.descriptor.keys {
            var missing = Self.descriptor; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try Conv3dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: missing)), key)
        }
        let changes: [(String, Any)] = [("extra", true), ("sampleRank", 3), ("preservesBatch", 1), ("maximumMultiplyAdds", 129_000_000), ("spatialParameterLength", 2), ("maximumSampleElements", 99_999_999), ("maximumPaddedSampleElements", 99_999_999), ("paddingModes", ["zeros"]), ("graphDTypes", ["float32", "float64"])]
        for (key, value) in changes {
            var changed = Self.descriptor; changed[key] = value
            XCTAssertThrowsError(try Conv3dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: changed)))
        }
    }
    func testSharedRegistryAndBridgeDescribeChannelsFirstLearnedState() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "conv3d"), Conv3dComponent.definition)
        let package = try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1")
        let value = try XCTUnwrap(package.capabilities.first { $0.id == "conv3d" })
        XCTAssertEqual(value.adapterID, "kwip.runner.graph-v2.conv3d")
        XCTAssertEqual(value.ports.map(\.id), ["in", "out"])
        XCTAssertEqual(value.ports[0].schema.tensor?.dimensions, [.parameter("in_channels"), .symbol("inputDepth"), .symbol("inputHeight"), .symbol("inputWidth")])
        XCTAssertEqual(value.ports[1].schema.tensor?.dimensions, [.parameter("out_channels"), .symbol("outputDepth"), .symbol("outputHeight"), .symbol("outputWidth")])
        for port in value.ports {
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.minimum, 1)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.maximum, 1)
            XCTAssertEqual(port.schema.tensor?.dtype, .float32)
        }
        XCTAssertEqual(value.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        XCTAssertEqual(value.learning.lessonIDs, ["assessment.conv3d-shared-kernel"]); XCTAssertEqual(value.learning.exampleIDs, ["prepared-conv3d-classifier"])
        var mutated = registry
        let index = try XCTUnwrap(mutated.components.firstIndex { $0.id == "conv3d" })
        mutated.components[index].fields[0].max = 1024
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry: mutated, runtimeVersion: "2.14.1"))
    }
}
