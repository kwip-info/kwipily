import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class AttentionDiscoveryTests:XCTestCase {
    func testExactOptionalFamilyDiscoveryAndRevocation()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:storage,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:root,storageOverride:storage,detectCapabilities:false)
        let base=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu,.mps]))
        for id in AttentionConfiguration.componentIDs.sorted() {
            let descriptor=try JSONEncoder().encode(BundledAttentionComponents.entries[id]!.runtimeContract)
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base,attentionDescriptors:[id:descriptor])
            XCTAssertEqual(store.attentionRuntimeCapabilities.componentIDs,[id]);XCTAssertEqual(store.verifiedRuntimeComponentCount,6)
            var wrong=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptor) as? [String:Any]);wrong["maskConvention"]="oneAllowed"
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base,attentionDescriptors:[id:JSONSerialization.data(withJSONObject:wrong)])
            XCTAssertTrue(store.attentionRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
        }
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu","mps"],descriptor:base)
        XCTAssertTrue(store.attentionRuntimeCapabilities.componentIDs.isEmpty)
    }
}
