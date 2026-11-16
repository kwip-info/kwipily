import XCTest
@testable import KwipMLCore
final class StatefulNormalizationDeclarationTests:XCTestCase {
    func testExactFiveDeclarationsKeepShapeModesAndParametersSeparateFromBuffers()throws {
        let declarations:[String:(ComponentDefinition)throws->CapabilityDescriptor]=["batch_norm1d":StatefulNormalizationCapabilityBridge.batchNorm1dDeclaration,"batch_norm2d":StatefulNormalizationCapabilityBridge.batchNorm2dDeclaration,"instance_norm1d":StatefulNormalizationCapabilityBridge.instanceNorm1dDeclaration,"instance_norm2d":StatefulNormalizationCapabilityBridge.instanceNorm2dDeclaration,"group_norm":StatefulNormalizationCapabilityBridge.groupNormDeclaration]
        for (id,declare) in declarations {
            let definition=BundledNormalizationComponents.definition(id),descriptor=try declare(definition)
            XCTAssertEqual(descriptor.id,id);XCTAssertEqual(descriptor.fields,definition.fields)
            XCTAssertEqual(descriptor.devices,[.cpu,.mps]);XCTAssertEqual(descriptor.learning.lessonIDs.count,2)
            XCTAssertEqual(descriptor.lifecycle.phases,[.initialize,.forward,.backward,.export])
            XCTAssertEqual(descriptor.lifecycle.state.map(\.kind),id == "group_norm" ? [.parameters]:[.parameters,.buffers])
            var changed=definition;changed.fields=[];XCTAssertThrowsError(try declare(changed))
        }
    }
    func testOptionalRuntimeDiscoveryRequiresExactPerMemberStateAndGeometryPolicy()throws {
        XCTAssertTrue(StatefulNormalizationRuntimeCapabilities().componentIDs.isEmpty)
        for id in StatefulNormalizationRuntimeCapabilities.operations {
            let entry=try XCTUnwrap(BundledNormalizationComponents.entries[id]),data=try JSONEncoder().encode(entry.runtimeContract)
            XCTAssertEqual(StatefulNormalizationRuntimeCapabilities(descriptors:[id:data]).componentIDs,[id])
            XCTAssertThrowsError(try StatefulNormalizationRuntimeCapabilities.verify(Data("{}".utf8),componentID:id))
            var value=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any]);value["graphDTypes"]=["int64"]
            XCTAssertFalse(StatefulNormalizationRuntimeCapabilities(descriptors:[id:try JSONSerialization.data(withJSONObject:value)]).contains(id))
        }
    }
}
