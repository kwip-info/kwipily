import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BilinearCapabilityTests:XCTestCase {
    let descriptor:[String:Any] = ["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.bilinear","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"minimumSampleRank":1,"sampleBroadcasting":false,"maximumSampleElements":100_000_000,"maximumParameters":2_000_000,"featureMinimum":1,"featureMaximum":512,"biasModes":[false,true],"graphDTypes":["float32"]]
    func testStrictDescriptor()throws {
        XCTAssertNoThrow(try BilinearRuntimeCapability.decode(JSONSerialization.data(withJSONObject:descriptor)))
        for (key,value) in [("maximumRank",7 as Any),("minimumSampleRank",2 as Any),("sampleBroadcasting",true as Any),("maximumSampleElements",200_000_000 as Any),("maximumParameters",3_000_000 as Any),("featureMinimum",0 as Any),("featureMaximum",1024 as Any),("biasModes",[true] as Any),("graphDTypes",["int64"] as Any),("preservesBatch",1 as Any),("adapterID","other" as Any),("extra",true as Any),("graphSchemaVersions",[1,2] as Any),("version",["major":1,"minor":1,"patch":0] as Any)] {
            var bad=descriptor;bad[key]=value
            XCTAssertThrowsError(try BilinearRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)),key)
        }
        var missing=descriptor;missing.removeValue(forKey:"preservesBatch")
        XCTAssertThrowsError(try BilinearRuntimeCapability.decode(JSONSerialization.data(withJSONObject:missing)))
        XCTAssertThrowsError(try BilinearRuntimeCapability.decode(Data(repeating:32,count:4097)))
    }
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,bilinearDescriptor:optional, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertNotNil(store.bilinearRuntimeCapability)
        try install(nil)
        XCTAssertNil(store.bilinearRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix-test",type:"bilinear"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("Bilinear requires") == true)
        try install(Data("{}".utf8))
        XCTAssertNil(store.bilinearRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertNil(store.bilinearRuntimeCapability)
    }
    @MainActor func testActualWorkerProbeInstallsBilinearDescriptor()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repo.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation {continuation in
            store.refreshCapabilities {continuation.resume(returning:$0)}
        }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertNotNil(store.bilinearRuntimeCapability)
        store.bilinearRuntimeCapability=nil
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix",type:"bilinear"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("Bilinear requires") == true)
    }

}
