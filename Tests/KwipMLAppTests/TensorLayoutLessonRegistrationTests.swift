import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorLayoutLessonRegistrationTests: XCTestCase {
    func testEachAuthoredOperationIsFindableWithoutDuplicateExampleCards() throws {
        for id in ["reshape", "transpose", "flatten"] {
            let entry = try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first { $0.componentID == id })
            XCTAssertEqual(entry.item.category, "Tensors & linear algebra")
            XCTAssertTrue(entry.item.matches(section: "Examples", search: id))
            XCTAssertEqual(entry.learning.exampleID, "prepared-tensor-layout-classifier")
            XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == id })
            let document = try PreparedWorkflowExample.component(id).document(device: "cpu")
            XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == id }; return false })
            XCTAssertEqual(PreparedWorkflowExample.component(id).observationsResource, "workflow-observations")
        }
        XCTAssertNotNil(LearnCatalogContent.items.first { $0.id == "tensor-layout" && $0.kind == "Construction" })
        XCTAssertNotNil(LearnCatalogContent.items.first { $0.id == "tensor-example" && $0.kind == "Example" })
    }
}
