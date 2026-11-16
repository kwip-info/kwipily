import XCTest
@testable import KwipMLCore

final class DepthwiseConv2dDiscoveryTests: XCTestCase {
    static var descriptor: [String: Any] {
        ["version": ["major": 1, "minor": 0, "patch": 0], "adapterID": "kwip.runner.graph-v2.depthwise_conv2d",
         "graphSchemaVersions": [2], "sampleRank": 3, "preservesBatch": true, "graphDTypes": ["float32"],
         "spatialParameterLength": 2, "maximumSampleElements": 100_000_000, "maximumPaddedSampleElements": 100_000_000,
         "channelMinimum": 1, "channelMaximum": 512, "lengthMinimum": 1, "lengthMaximum": 4096,
         "kernelMinimum": 1, "kernelMaximum": 4096, "strideMinimum": 1, "strideMaximum": 4096,
         "dilationMinimum": 1, "dilationMaximum": 4096, "channelMultiplierMinimum": 1, "channelMultiplierMaximum": 512, "outputChannelMaximum": 512,
         "groupsRule": "inputChannels", "outputChannelsRule": "inputChannelsTimesMultiplier",
         "configurationKeys": ["in_channels", "channel_multiplier", "kernel_size", "stride", "dilation", "bias", "padding_policy", "padding", "padding_mode"],
         "paddingMinimum": 0, "paddingMaximum": 4096, "maximumParameters": 2_000_000,
         "maximumMultiplyAdds": 128_000_000, "paddingPolicies": ["explicit", "valid", "same"],
         "paddingModes": ["zeros", "reflect", "replicate", "circular"], "biasModes": [false, true]]
    }
    func testExactDescriptorRejectsMissingExtraAndChangedFields() throws {
        XCTAssertNoThrow(try DepthwiseConv2dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: Self.descriptor)))
        for key in Self.descriptor.keys {
            var missing = Self.descriptor; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try DepthwiseConv2dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: missing)), key)
        }
        let changes: [(String, Any)] = [("groupsRule", "editable"), ("outputChannelMaximum", 1024), ("extra", true), ("sampleRank", 2), ("preservesBatch", 1), ("maximumMultiplyAdds", 129_000_000), ("spatialParameterLength", 1), ("maximumSampleElements", 99_999_999), ("maximumPaddedSampleElements", 99_999_999), ("paddingModes", ["zeros"]), ("graphDTypes", ["float32", "float64"])]
        for (key, value) in changes {
            var changed = Self.descriptor; changed[key] = value
            XCTAssertThrowsError(try DepthwiseConv2dRuntimeCapability.decode(JSONSerialization.data(withJSONObject: changed)))
        }
    }
    func testSharedRegistryAndBridgeDescribeChannelsFirstLearnedState() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "depthwise_conv2d"), DepthwiseConv2dComponent.definition)
        let package = try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1")
        let value = try XCTUnwrap(package.capabilities.first { $0.id == "depthwise_conv2d" })
        XCTAssertEqual(value.adapterID, "kwip.runner.graph-v2.depthwise_conv2d")
        XCTAssertEqual(value.ports.map(\.id), ["in", "out"])
        XCTAssertEqual(value.ports[0].schema.tensor?.dimensions, [.parameter("in_channels"), .symbol("inputHeight"), .symbol("inputWidth")])
        XCTAssertEqual(value.ports[1].schema.tensor?.dimensions, [.symbol("derivedOutputChannels"), .symbol("outputHeight"), .symbol("outputWidth")])
        for port in value.ports {
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.minimum, 1)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.maximum, 1)
            XCTAssertEqual(port.schema.tensor?.dtype, .float32)
        }
        XCTAssertEqual(value.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        XCTAssertTrue(value.learning.lessonIDs.isEmpty); XCTAssertEqual(value.learning.exampleIDs, ["prepared-depthwise-conv2d-classifier"])
        var mutated = registry
        let index = try XCTUnwrap(mutated.components.firstIndex { $0.id == "depthwise_conv2d" })
        mutated.components[index].fields[0].max = 1024
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry: mutated, runtimeVersion: "2.14.1"))
    }
}
