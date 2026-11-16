import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LayerNormCapabilityTests:XCTestCase {
    let descriptor:[String:Any] = ["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.layer_norm","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true,"epsilonMinimum":LayerNormConfiguration.epsilonRange.lowerBound,"epsilonMaximum":LayerNormConfiguration.epsilonRange.upperBound,"elementwiseAffineModes":[false,true],"biasModes":[false,true],"varianceCorrection":0,"usesCurrentInputStatistics":true]
    func testStrictDescriptor()throws {
        XCTAssertNoThrow(try LayerNormRuntimeCapability.decode(JSONSerialization.data(withJSONObject:descriptor)))
        for (key,value) in [("epsilonMinimum",0 as Any),("epsilonMaximum",1e300 as Any),("varianceCorrection",1 as Any),("usesCurrentInputStatistics",false as Any),("elementwiseAffineModes",[true] as Any),("biasModes",[true] as Any),("maximumRank",7 as Any),("preservesBatch",1 as Any),("adapterID","other" as Any),("extra",true as Any),("graphSchemaVersions",[1,2] as Any),("version",["major":1,"minor":1,"patch":0] as Any)] {
            var bad=descriptor;bad[key]=value
            XCTAssertThrowsError(try LayerNormRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)),key)
        }
        var missing=descriptor;missing.removeValue(forKey:"preservesBatch")
        XCTAssertThrowsError(try LayerNormRuntimeCapability.decode(JSONSerialization.data(withJSONObject:missing)))
        XCTAssertThrowsError(try LayerNormRuntimeCapability.decode(Data(repeating:32,count:4097)))
    }
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,layerNormDescriptor:optional, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertNotNil(store.layerNormRuntimeCapability)
        try install(nil)
        XCTAssertNil(store.layerNormRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"layerNorm-test",type:"layer_norm",config:["normalized_shape":.array([2]),"eps":.number(1e-5),"elementwise_affine":true,"bias":true]))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("Layer normalization requires") == true)
        try install(Data("{}".utf8))
        XCTAssertNil(store.layerNormRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertNil(store.layerNormRuntimeCapability)
    }
}
