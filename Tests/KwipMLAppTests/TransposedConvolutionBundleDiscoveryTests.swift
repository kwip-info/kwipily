import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TransposedConvolutionBundleDiscoveryTests: XCTestCase {
    func testAssessedLabIsUniqueAndSearchableForEveryDimension() throws {
        let cards = LearnCatalogContent.items.filter { $0.id == "transposed-convolution" }
        XCTAssertEqual(cards.count, 1)
        let card = try XCTUnwrap(cards.first)
        XCTAssertEqual(card.kind, "Construction")
        for dimension in 1...3 {
            XCTAssertTrue(card.matches(section: "Library", search: "ConvTranspose\(dimension)d"))
        }
    }

    func testEachDimensionKeepsOneExampleAndItsActualAuthoredFactory() throws {
        for dimension in 1...3 {
            let id = "conv_transpose\(dimension)d"
            let entry = try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first { $0.componentID == id })
            XCTAssertEqual(entry.learning.exampleID, "prepared-conv-transpose\(dimension)d-classifier")
            XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == id })
            let example = PreparedWorkflowExample.component(id)
            XCTAssertEqual(example.observationsResource, "workflow-observations")
            let document = try example.document(device: "cpu")
            XCTAssertEqual(document.nodes.filter {
                if case .modelPrimitive(let type, _, _) = $0.payload { return type == id }
                return false
            }.count, 1)
            XCTAssertEqual(LearnCatalogContent.items.filter { $0.id == "transpose\(dimension)d" && $0.kind == "Example" }.count, 1)
        }
    }
}
