import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MatrixMultiplyCapabilityTests:XCTestCase {
    let descriptor:[String:Any] = ["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.matrixMultiply","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"minimumSampleRank":2,"sampleBroadcasting":true,"maximumOutputElements":100_000_000,"graphDTypes":["float32"]]
    func testStrictDescriptor()throws {
        XCTAssertNoThrow(try MatrixMultiplyRuntimeCapability.decode(JSONSerialization.data(withJSONObject:descriptor)))
        for (key,value) in [("maximumRank",7 as Any),("minimumSampleRank",1 as Any),("sampleBroadcasting",false as Any),("maximumOutputElements",200_000_000 as Any),("graphDTypes",["int64"] as Any),("preservesBatch",1 as Any),("adapterID","other" as Any),("extra",true as Any),("graphSchemaVersions",[1,2] as Any),("version",["major":1,"minor":1,"patch":0] as Any)] {
            var bad=descriptor;bad[key]=value
            XCTAssertThrowsError(try MatrixMultiplyRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)),key)
        }
        var missing=descriptor;missing.removeValue(forKey:"preservesBatch")
        XCTAssertThrowsError(try MatrixMultiplyRuntimeCapability.decode(JSONSerialization.data(withJSONObject:missing)))
        XCTAssertThrowsError(try MatrixMultiplyRuntimeCapability.decode(Data(repeating:32,count:4097)))
    }
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,matrixMultiplyDescriptor:optional, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertNotNil(store.matrixMultiplyRuntimeCapability)
        try install(nil)
        XCTAssertNil(store.matrixMultiplyRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix-test",type:"matmul"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("Matrix multiplication requires") == true)
        try install(Data("{}".utf8))
        XCTAssertNil(store.matrixMultiplyRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertNil(store.matrixMultiplyRuntimeCapability)
    }
    @MainActor func testActualWorkerProbeInstallsMatrixDescriptor()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repo.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation {continuation in
            store.refreshCapabilities {continuation.resume(returning:$0)}
        }
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertNotNil(store.matrixMultiplyRuntimeCapability)
        store.matrixMultiplyRuntimeCapability=nil
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"matrix",type:"matmul"))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("Matrix multiplication requires") == true)
    }

}
