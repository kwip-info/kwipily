import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LazyLinearBundleDiscoveryTests: XCTestCase {
    func testAuthoredExampleAndAssessedTaskShareOneActualCapabilityIdentity() throws {
        let entry = try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first { $0.componentID == "lazy_linear" })
        XCTAssertEqual(entry.item.category, "Layers & activations")
        XCTAssertTrue(entry.item.matches(section: "Examples", search: "inferred"))
        XCTAssertEqual(entry.learning.exampleID, "prepared-lazy-linear-classifier")
        XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == "lazy_linear" }, "The specialized example must not create a duplicate generic card")
        let example = PreparedWorkflowExample.component("lazy_linear")
        XCTAssertEqual(example.observationsResource, "workflow-observations")
        let document = try example.document(device: "cpu")
        XCTAssertEqual(document.nodes.filter { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "lazy_linear" }; return false }.count, 2)
        let declaration = try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry: .builtIn()).first { $0.id == "lazy_linear" })
        XCTAssertEqual(declaration.learning.lessonIDs, ["assessment.lazy-linear-inferred-width"])
        XCTAssertTrue(TensorLearningTask.operations.contains(LazyLinearAssessment.operation))
        XCTAssertFalse(declaration.fields.contains { $0.key == "in_features" }, "Input width is derived from the connector")
    }
}
