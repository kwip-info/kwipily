import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MatrixBundleDiscoveryTests: XCTestCase {
    func testAuthoredMatrixExampleUsesActualFactoryAndBothAssessedTasksWithoutDuplicateCards() throws {
        let entry = try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first { $0.componentID == "matmul" })
        XCTAssertEqual(entry.item.category, "Tensors & linear algebra")
        XCTAssertEqual(entry.learning.exampleID, "prepared-matrix-classifier")
        XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == "matmul" })
        let example = PreparedWorkflowExample.component("matmul")
        XCTAssertEqual(example.observationsResource, "workflow-observations")
        let document = try example.document(device: "cpu")
        XCTAssertEqual(document.nodes.filter { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "matmul" }; return false }.count, 1)
        let descriptor = try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry: .builtIn()).first { $0.id == "matmul" })
        XCTAssertEqual(descriptor.learning.lessonIDs, ["assessment.matrix-contraction", "assessment.matrix-broadcast-gradients"])
        XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
    }
}
