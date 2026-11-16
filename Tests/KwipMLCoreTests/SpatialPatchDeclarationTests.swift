import Foundation
import XCTest
@testable import KwipMLCore

final class SpatialPatchDeclarationTests:XCTestCase {
    func testExactPairedDefinitionsDeclareDifferentInputOutputRanksAndNoOperatorState()throws {
        for id in ["unfold","fold"] {
            let definition=BundledSpatialPatchComponents.definition(id)
            let declaration:(ComponentDefinition)throws->CapabilityDescriptor = id == "unfold" ? SpatialPatchCapabilityBridge.unfoldDeclaration:SpatialPatchCapabilityBridge.foldDeclaration
            let descriptor=try declaration(definition)
            XCTAssertEqual(descriptor.ports.map(\.id),["in","out"])
            XCTAssertEqual(descriptor.fields,definition.fields)
            XCTAssertEqual(descriptor.devices,[.cpu,.mps])
            XCTAssertEqual(descriptor.lifecycle.phases,[.initialize,.forward,.backward,.export])
            XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
            XCTAssertTrue(CapabilityContractValidator.validate(descriptor).isValid)
            let ranks=descriptor.ports.compactMap {$0.schema.tensor?.dimensions.count}
            XCTAssertEqual(ranks,id == "unfold" ? [3,2]:[2,3])
            for port in descriptor.ports {
                XCTAssertEqual(port.schema.tensor?.dtype,.float32)
                XCTAssertEqual(port.schema.tensor?.leadingAxes,.some(.init(minimum:1,maximum:1)))
            }
            var changed=definition;changed.outputs.append("inverse")
            XCTAssertThrowsError(try declaration(changed))
        }
    }
    func testSupportRequiresExactPresentDescriptorForEachMember()throws {
        XCTAssertTrue(SpatialPatchRuntimeCapabilities().componentIDs.isEmpty)
        for id in ["unfold","fold"] {
            let data=try JSONEncoder().encode(try XCTUnwrap(BundledSpatialPatchComponents.entries[id]).runtimeContract)
            XCTAssertEqual(SpatialPatchRuntimeCapabilities(descriptors:[id:data]).componentIDs,[id])
            var object=try XCTUnwrap(JSONSerialization.jsonObject(with:data)as?[String:Any]);object["paddingMode"]="reflect"
            let changed=try JSONSerialization.data(withJSONObject:object)
            XCTAssertThrowsError(try SpatialPatchRuntimeCapabilities.verify(changed,componentID:id))
            XCTAssertTrue(SpatialPatchRuntimeCapabilities(descriptors:[id:changed]).componentIDs.isEmpty)
            XCTAssertThrowsError(try SpatialPatchRuntimeCapabilities.verify(data,componentID:"unknown"))
            XCTAssertThrowsError(try SpatialPatchRuntimeCapabilities.verify(Data(repeating:32,count:4097),componentID:id))
        }
    }
}
