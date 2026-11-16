import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BilinearLessonRegistrationTests: XCTestCase {
    func testAuthoredMetadataKeepsSpecializedLabAndActualExampleDiscoverable() throws {
        let entry = try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first { $0.componentID == "bilinear" })
        XCTAssertEqual(entry.item.category, "Layers & activations")
        XCTAssertTrue(entry.item.matches(section: "Examples", search: "bilinear"))
        XCTAssertEqual(entry.learning.exampleID, "prepared-bilinear-classifier")
        XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == "bilinear" })
        XCTAssertNotNil(LearnCatalogContent.items.first { $0.id == "bilinear" && $0.kind == "Construction" })
        XCTAssertNotNil(LearnCatalogContent.items.first { $0.id == "bilinear-example" && $0.kind == "Example" })
        let document = try PreparedWorkflowExample.component("bilinear").document(device: "cpu")
        XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "bilinear" }; return false })
        XCTAssertEqual(PreparedWorkflowExample.component("bilinear").observationsResource, "workflow-observations")
        let declaration = try BuiltInCapabilityBridge.declarations(registry: .builtIn()).first { $0.id == "bilinear" }
        XCTAssertEqual(try XCTUnwrap(declaration).learning.lessonIDs, ["assessment.bilinear-with-bias", "assessment.bilinear-without-bias"])
    }
}
