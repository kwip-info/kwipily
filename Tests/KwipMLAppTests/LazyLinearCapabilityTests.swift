import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LazyLinearCapabilityTests:XCTestCase {
    let descriptor:[String:Any] = ["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.lazy-linear","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"minimumSampleRank":1,"maximumSampleElements":100_000_000,"maximumParameters":2_000_000,"inputFeatureMinimum":1,"inputFeatureMaximum":4096,"outputFeatureMinimum":1,"outputFeatureMaximum":512,"biasModes":[false,true],"graphDTypes":["float32"],"inputWidthInferred":true,"materializationPolicy":"validatedShapeBeforeOptimizer"]
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,lazyLinearDescriptor:optional, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertNotNil(store.lazyLinearRuntimeCapability)
        let supportedCount=store.verifiedRuntimeComponentCount
        try install(nil)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,supportedCount-1)
        XCTAssertNil(store.lazyLinearRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix-test",type:"lazy_linear"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("LazyLinear requires") == true)
        try install(Data("{}".utf8))
        XCTAssertNil(store.lazyLinearRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertNil(store.lazyLinearRuntimeCapability)
    }
    @MainActor func testActualWorkerProbeInstallsLazyLinearDescriptor()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repo.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation {continuation in
            store.refreshCapabilities {continuation.resume(returning:$0)}
        }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertEqual(store.registry.component(type:"lazy_linear"),LazyLinearComponent.definition)
        XCTAssertNotNil(store.lazyLinearRuntimeCapability)
        store.lazyLinearRuntimeCapability=nil
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix",type:"lazy_linear"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("LazyLinear requires") == true)
    }

}
