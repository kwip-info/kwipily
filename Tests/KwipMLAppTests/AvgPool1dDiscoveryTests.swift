import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class AvgPool1dDiscoveryTests:XCTestCase {
    let descriptor: [String:Any] = [
        "version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.avg_pool1d",
        "graphSchemaVersions":[2],"sampleRank":2,"preservesBatch":true,"graphDTypes":["float32"],
        "channelMinimum":1,"channelMaximum":512,"lengthMinimum":1,"lengthMaximum":4096,
        "kernelMinimum":1,"kernelMaximum":4096,"strideMinimum":1,"strideMaximum":4096,
        "paddingMinimum":0,"paddingMaximum":2048,
        "paddingKernelDivisor":2,"nullStrideFollowsKernel":true,"ceilModes":[false,true],
        "countIncludePadModes":[false,true],"outputDTypes":["out":"float32"]]
    func testExactDescriptorRejectsChangedOutputAndUnknownKeys() throws {
        XCTAssertNoThrow(try PoolingRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject:descriptor), componentID:"avg_pool1d"))
        for key in descriptor.keys {
            var missing=descriptor;missing.removeValue(forKey:key)
            XCTAssertThrowsError(try PoolingRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject:missing), componentID:"avg_pool1d"))
        }
        var altered=descriptor;altered["outputDTypes"]=["out":"float32","indices":"float32"]
        XCTAssertThrowsError(try PoolingRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject:altered), componentID:"avg_pool1d"))
        var extra=descriptor;extra["unknown"]=true
        XCTAssertThrowsError(try PoolingRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject:extra), componentID:"avg_pool1d"))
    }
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,poolingDescriptors:optional.map { ["avg_pool1d":$0] } ?? [:], tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertTrue(store.poolingRuntimeCapabilities.contains("avg_pool1d"))
        let supportedCount=store.verifiedRuntimeComponentCount
        try install(nil)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,supportedCount-1)
        XCTAssertFalse(store.poolingRuntimeCapabilities.contains("avg_pool1d"))
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix-test",type:"avg_pool1d"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("AvgPool1d requires") == true)
        try install(Data("{}".utf8))
        XCTAssertFalse(store.poolingRuntimeCapabilities.contains("avg_pool1d"))
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertFalse(store.poolingRuntimeCapabilities.contains("avg_pool1d"))
    }
    @MainActor func testActualWorkerProbeInstallsAvgPool1dDescriptor()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repo.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation {continuation in
            store.refreshCapabilities {continuation.resume(returning:$0)}
        }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertEqual(store.registry.component(type:"avg_pool1d"),AvgPool1dComponent.definition)
        XCTAssertTrue(store.poolingRuntimeCapabilities.contains("avg_pool1d"))
        store.poolingRuntimeCapabilities = .init()
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix",type:"avg_pool1d"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("AvgPool1d requires") == true)
    }

}
