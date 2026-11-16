import XCTest
@testable import KwipMLCore

final class TransposedConvolutionBundleMetadataTests:XCTestCase {
    func testEachPublishedDimensionOwnsItsNineControlsAndAssessedIdentity()throws {
        let registry=try ComponentRegistry.builtIn()
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(registry).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count,83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        let metadata=[ConvTranspose1dComponent.metadata,ConvTranspose2dComponent.metadata,ConvTranspose3dComponent.metadata]
        let keys:Set<String>=["in_channels","out_channels","kernel_size","stride","padding","output_padding","dilation","groups","bias"]
        for (index,value) in metadata.enumerated() {
            let dimension=index+1,id="conv_transpose\(dimension)d",record=try XCTUnwrap(value)
            XCTAssertEqual(record.definition,try XCTUnwrap(registry.components.first {$0.id == id}))
            XCTAssertEqual(record.definition,try XCTUnwrap(BundledConvolutionComponents.metadata(id)).definition)
            XCTAssertTrue(record.published)
            XCTAssertEqual(Set(record.definition.fields.map(\.key)),keys)
            XCTAssertEqual(record.presentation.icon,id+"-v2")
            XCTAssertEqual(record.learning?.exampleID,"prepared-"+id.replacingOccurrences(of:"_",with:"-")+"-classifier")
            let declaration=try BuiltInCapabilityBridge.declarations(registry:registry).first {$0.id == id}
            let descriptor = try XCTUnwrap(declaration)
            XCTAssertEqual(descriptor.learning.lessonIDs,["assessment.conv-transpose\(dimension)d-shared-kernel"])
            XCTAssertEqual(descriptor.lifecycle.state.map(\.kind),[.parameters])
            var altered = registry
            let position = try XCTUnwrap(altered.components.firstIndex { $0.id == id })
            altered.components[position].title = "Unreviewed"
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry: altered))
        }
    }
}
