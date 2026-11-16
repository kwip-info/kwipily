import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TensorRuntimeCapabilityTests:XCTestCase {
    let descriptor=Data(#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true}"#.utf8)
    func testOldRuntimeCannotExecuteOrCountTransformsAndReinstallClearsSupport()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let contract=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        func install(_ tensor:Data?)throws {try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,tensorLayoutDescriptor:tensor)}
        try install(nil)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
        XCTAssertNotNil(store.capabilityBoundaryError(for:ExampleFlows.tensorLayoutClassifier))
        try install(descriptor)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,8)
        XCTAssertNil(store.capabilityBoundaryError(for:ExampleFlows.tensorLayoutClassifier))
        var wrong=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptor) as? [String:Any]);wrong["preservesBatch"]=false
        try install(JSONSerialization.data(withJSONObject:wrong))
        XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
        XCTAssertNotNil(store.capabilityBoundaryError(for:ExampleFlows.tensorLayoutClassifier))
        try install(descriptor)
        store.refreshCapabilities()
        XCTAssertNil(store.tensorLayoutRuntimeCapability)
    }
    func testDescriptorRejectsUnknownMissingAndDriftedFields()throws {
        XCTAssertNoThrow(try TensorLayoutRuntimeCapability.decode(descriptor))
        let raw=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptor) as? [String:Any])
        for key in raw.keys {
            var changed=raw;changed.removeValue(forKey:key)
            XCTAssertThrowsError(try TensorLayoutRuntimeCapability.decode(JSONSerialization.data(withJSONObject:changed)))
        }
        var changed=raw;changed["unknown"]=true
        XCTAssertThrowsError(try TensorLayoutRuntimeCapability.decode(JSONSerialization.data(withJSONObject:changed)))
        changed=raw;changed["maximumRank"]=9
        XCTAssertThrowsError(try TensorLayoutRuntimeCapability.decode(JSONSerialization.data(withJSONObject:changed)))
    }
}
