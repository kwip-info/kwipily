import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RecurrentDiscoveryTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testUndiscoveredRecurrentCapabilitiesAreNotImplicitlyGranted()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let base=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
        for id in RecurrentConfiguration.componentIDs {
            let graph=try ExampleFlows.recurrentClassifier(componentID:id)
            let message=try XCTUnwrap(store.capabilityBoundaryError(for:graph,context:.preparedWorkflow))
            XCTAssertTrue(message.contains("verified runtime support"),message)
        }
    }
    func testExactOptionalDescriptorsRevokeAndProjectedDeviceRefuses()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let base=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu,.mps]))
        let layout=try JSONSerialization.data(withJSONObject:["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.tensor-layout","graphSchemaVersions":[2],"operations":["reshape","transpose","flatten"],"maximumRank":8,"preservesBatch":true])
        let descriptors=try Dictionary(uniqueKeysWithValues:RecurrentConfiguration.componentIDs.map {id in (id,try JSONEncoder().encode(BundledRecurrentComponents.entries[id]!.runtimeContract))})
        for id in RecurrentConfiguration.componentIDs.sorted() {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base,tensorLayoutDescriptor:layout,recurrentDescriptors:[id:descriptors[id]!])
            XCTAssertEqual(store.recurrentRuntimeCapabilities.componentIDs,[id]);XCTAssertEqual(store.verifiedRuntimeComponentCount,9)
            XCTAssertNil(store.capabilityBoundaryError(for:try ExampleFlows.recurrentClassifier(componentID:id),context:.preparedWorkflow))
            var wrong=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptors[id]!) as? [String:Any]);wrong["statePersistence"]="implicitCarry"
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base,tensorLayoutDescriptor:layout,recurrentDescriptors:[id:JSONSerialization.data(withJSONObject:wrong)])
            XCTAssertTrue(store.recurrentRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,8)
        }
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base,tensorLayoutDescriptor:layout,recurrentDescriptors:descriptors)
        XCTAssertEqual(store.verifiedRuntimeComponentCount,14)
        var graph=try ExampleFlows.recurrentClassifier(componentID:"lstm")
        let i=try XCTUnwrap(graph.nodes.firstIndex {$0.type=="lstm"});graph.nodes[i].config["hidden_size"]=3;graph.nodes[i].config["proj_size"]=2
        graph.execution?.device="cpu";XCTAssertNil(store.capabilityBoundaryError(for:graph,context:.preparedWorkflow))
        for device in ["mps","auto"] {graph.execution?.device=device;let refusal=try XCTUnwrap(store.capabilityBoundaryError(for:graph,context:.preparedWorkflow));XCTAssertTrue(refusal.contains("Choose CPU"),refusal)}
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base,tensorLayoutDescriptor:layout)
        XCTAssertTrue(store.recurrentRuntimeCapabilities.componentIDs.isEmpty)
    }

}
