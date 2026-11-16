import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LinearLessonRegistrationTests: XCTestCase {
    func testAuthoredLinearExampleIsFindableAndLaunchesActualPreparedFlow() throws {
        let entry = try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first { $0.componentID == "linear" })
        XCTAssertTrue(entry.item.matches(section: "Examples", search: "weights"))
        XCTAssertEqual(entry.learning.exampleID, "prepared-parallel-classifier")
        let document = try PreparedWorkflowExample.component("linear").document(device: "cpu")
        XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "linear" }; return false })
        XCTAssertTrue(document.nodes.contains { if case .modelBlock = $0.payload { return true }; return false })
        XCTAssertEqual(PreparedWorkflowExample.component("linear").observationsResource, "workflow-observations")
    }
}
