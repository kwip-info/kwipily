import XCTest
@testable import KwipMLCore
final class LazyLinearBundleMetadataTests: XCTestCase {
    func testAuthoredLazyMetadataRetainsInferredInputExampleAndMeasuredLesson() throws {
        let entry = try XCTUnwrap(LazyLinearComponent.metadata)
        let definition = LazyLinearComponent.definition
        XCTAssertEqual(entry.definition, definition)
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(try .builtIn()).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count, 83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        XCTAssertEqual(definition.fields.map(\.key), ["out_features", "bias"])
        XCTAssertEqual(definition.fields.map(\.defaultValue), [2,true])
        XCTAssertEqual(entry.presentation.icon, "lazy_linear-v2")
        XCTAssertEqual(entry.learning?.exampleID, "prepared-lazy-linear-classifier")
        XCTAssertNoThrow(try LazyLinearRuntimeCapability.decode(JSONEncoder().encode(entry.runtimeContract)))
        let descriptor = try BuiltInCapabilityBridge.lazyLinearDeclaration(definition)
        XCTAssertEqual(descriptor.version, ContractVersion())
        XCTAssertEqual(descriptor.adapterID, "kwip.runner.graph-v2.lazy-linear")
        XCTAssertEqual(descriptor.shapeRuleID, "kwip.shape.lazy-linear.v1")
        XCTAssertEqual(descriptor.learning.lessonIDs, ["assessment.lazy-linear-inferred-width"])
        XCTAssertEqual(descriptor.learning.exampleIDs, ["prepared-lazy-linear-classifier"])
        XCTAssertEqual(descriptor.ports.first?.schema.tensor?.dimensions, [.symbol("inferredFeatures")])
        XCTAssertEqual(descriptor.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        var wrong = definition; wrong.id = "not-lazy-linear"
        XCTAssertThrowsError(try BuiltInCapabilityBridge.lazyLinearDeclaration(wrong))
    }
}
