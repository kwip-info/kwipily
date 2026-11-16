import XCTest
@testable import KwipMLCore

final class TensorLayoutBundleMetadataTests: XCTestCase {
    func testAuthoredDefinitionsAreSharedWithLessonCanvasAndRetainLearningAndStateContracts() throws {
        let registry = try ComponentRegistry.builtIn()
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(registry).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count, 83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        for id in ["reshape", "transpose", "flatten"] {
            let definition = TensorLayoutComponent.definition(id)
            let entry = try XCTUnwrap(BundledTensorLayoutComponents.entries[id])
            XCTAssertEqual(definition, registry.component(type: id))
            XCTAssertEqual(definition, TensorLearningCanvas.registry.component(type: id))
            XCTAssertTrue(entry.published)
            XCTAssertEqual(registry.schemaVersion, 2)
            XCTAssertEqual(entry.presentation.icon, id + "-v2")
            XCTAssertEqual(entry.learning?.exampleID, "prepared-tensor-layout-classifier")
            let declaration = try XCTUnwrap(BundledTensorLayoutComponents.declarations[id])(definition)
            XCTAssertEqual(declaration.version, ContractVersion())
            XCTAssertEqual(declaration.adapterID, "kwip.runner.graph-v2." + id)
            XCTAssertEqual(declaration.shapeRuleID, "kwip.shape." + id + ".v1")
            XCTAssertTrue(declaration.lifecycle.state.isEmpty)
            XCTAssertEqual(declaration.learning.lessonIDs, ["assessment.tensor-" + id])
            XCTAssertEqual(declaration.learning.exampleIDs, ["kwip.example.tensor-layout-classifier.v1"])
            XCTAssertEqual(declaration.ports.map { $0.schema.tensor?.leadingAxes }, [.init(minimum: 2, maximum: 8), .init(minimum: 2, maximum: 8)])
            let data = try JSONEncoder().encode(entry.runtimeContract)
            XCTAssertNoThrow(try TensorLayoutRuntimeCapability.decode(data))
            var drift = definition
            drift.fields[0].defaultValue = 7
            XCTAssertThrowsError(try XCTUnwrap(BundledTensorLayoutComponents.declarations[id])(drift))
        }
    }
}
