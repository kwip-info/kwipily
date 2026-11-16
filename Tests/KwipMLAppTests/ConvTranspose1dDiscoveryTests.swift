import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConvTranspose1dDiscoveryTests:XCTestCase {
    let descriptor: [String: Any] =
        ["version": ["major": 1, "minor": 0, "patch": 0], "adapterID": "kwip.runner.graph-v2.conv_transpose1d",
         "graphSchemaVersions": [2], "sampleRank": 2, "preservesBatch": true, "graphDTypes": ["float32"],
         "spatialParameterKind": "scalar", "maximumSampleElements": 100_000_000, "maximumUncroppedSampleElements": 100_000_000,
         "channelMinimum": 1, "channelMaximum": 512, "lengthMinimum": 1, "lengthMaximum": 4096,
         "kernelMinimum": 1, "kernelMaximum": 4096, "strideMinimum": 1, "strideMaximum": 4096,
         "dilationMinimum": 1, "dilationMaximum": 4096, "groupMinimum": 1, "groupMaximum": 512,
         "paddingMinimum": 0, "paddingMaximum": 4096, "outputPaddingMinimum": 0, "outputPaddingMaximum": 4096,
         "maximumParameters": 2_000_000, "maximumMultiplyAdds": 128_000_000, "paddingMode": "zeros",
         "configurationKeys": ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "dilation", "groups", "bias"],
         "outputPaddingRule": "lessThanStrideOrDilation", "supportsDynamicOutputSize": false, "biasModes": [false, true]]
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,convTranspose1dDescriptor:optional, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertNotNil(store.convTranspose1dRuntimeCapability)
        let supportedCount=store.verifiedRuntimeComponentCount
        try install(nil)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,supportedCount-1)
        XCTAssertNil(store.convTranspose1dRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix-test",type:"conv_transpose1d"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("ConvTranspose1d requires") == true)
        try install(Data("{}".utf8))
        XCTAssertNil(store.convTranspose1dRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertNil(store.convTranspose1dRuntimeCapability)
    }
    @MainActor func testActualWorkerProbeInstallsConvTranspose1dDescriptor()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repo.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation {continuation in
            store.refreshCapabilities {continuation.resume(returning:$0)}
        }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertEqual(store.registry.component(type:"conv_transpose1d"),ConvTranspose1dComponent.definition)
        XCTAssertNotNil(store.convTranspose1dRuntimeCapability)
        store.convTranspose1dRuntimeCapability=nil
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix",type:"conv_transpose1d"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("ConvTranspose1d requires") == true)
    }

}
