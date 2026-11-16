import XCTest
@testable import KwipMLCore

final class ConvolutionBundleMetadataTests: XCTestCase {
    func testEachSpatialDimensionOwnsExactBundleAndIndependentAssessedIdentity() throws {
        let registry=try ComponentRegistry.builtIn()
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(registry).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count,83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        let components=[Conv1dComponent.metadata,Conv2dComponent.metadata,Conv3dComponent.metadata]
        for (dimension,optional) in components.enumerated() {
            let id="conv\(dimension+1)d"
            let entry=try XCTUnwrap(optional,id)
            XCTAssertEqual(BundledConvolutionComponents.metadata(id)?.definition,entry.definition)
            XCTAssertEqual(entry.definition,registry.component(type:id))
            XCTAssertEqual(entry.presentation.icon,id+"-v2")
            XCTAssertEqual(entry.definition.fields.map(\.key),["in_channels","out_channels","kernel_size","stride","dilation","groups","bias","padding_policy","padding","padding_mode"])
            XCTAssertEqual(entry.learning?.exampleID,"prepared-"+id+"-classifier")
            let descriptor=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:registry).first {$0.id == id})
            XCTAssertEqual(descriptor.learning.lessonIDs,["assessment."+id+"-shared-kernel"])
            XCTAssertEqual(descriptor.lifecycle.state.map(\.kind),[.parameters])
            for key in ["title","category"] {
                var altered=registry
                let index=try XCTUnwrap(altered.components.firstIndex {$0.id == id})
                if key == "title" {altered.components[index].title="Unreviewed"}
                else {altered.components[index].category="Unreviewed"}
                XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:altered),id+" "+key)
            }
        }
    }
}
