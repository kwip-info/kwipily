import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SoftmaxCapabilityTests:XCTestCase {
    let descriptor:[String:Any] = ["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.softmax","graphSchemaVersions":[2],"maximumRank":8,"preservesBatch":true]
    func testStrictDescriptor()throws {
        XCTAssertNoThrow(try SoftmaxRuntimeCapability.decode(JSONSerialization.data(withJSONObject:descriptor)))
        for (key,value) in [("maximumRank",7 as Any),("preservesBatch",1 as Any),("adapterID","other" as Any),("extra",true as Any),("graphSchemaVersions",[1,2] as Any),("version",["major":1,"minor":1,"patch":0] as Any)] {
            var bad=descriptor;bad[key]=value
            XCTAssertThrowsError(try SoftmaxRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)),key)
        }
        var missing=descriptor;missing.removeValue(forKey:"preservesBatch")
        XCTAssertThrowsError(try SoftmaxRuntimeCapability.decode(JSONSerialization.data(withJSONObject:missing)))
        XCTAssertThrowsError(try SoftmaxRuntimeCapability.decode(Data(repeating:32,count:4097)))
    }
    @MainActor func testOptionalDiscoveryClearsStaleSupportAndPreservesLegacy()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ optional:Data?)throws {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,softmaxDescriptor:optional, tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors())
        }
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertNotNil(store.softmaxRuntimeCapability)
        try install(nil)
        XCTAssertNil(store.softmaxRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        var graph=store.graph
        graph.nodes.append(GraphNode(id:"softmax-test",type:"softmax",config:["dim":-1]))
        XCTAssertTrue(store.capabilityBoundaryError(for:graph)?.contains("Softmax requires") == true)
        try install(Data("{}".utf8))
        XCTAssertNil(store.softmaxRuntimeCapability)
        XCTAssertNil(store.capabilityBoundaryError)
        try install(JSONSerialization.data(withJSONObject:descriptor))
        XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertNil(store.softmaxRuntimeCapability)
    }
}
