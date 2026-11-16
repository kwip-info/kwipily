import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConfigurableActivationDiscoveryTests: XCTestCase {
    func testAssessedLabIsDiscoverableByEachOperatorAndControl() throws {
        let matches = LearnCatalogContent.items.filter { $0.id == "configurable-activations" }
        XCTAssertEqual(matches.count, 1)
        let lab = try XCTUnwrap(matches.first)
        XCTAssertEqual(lab.kind, "Construction")
        for term in ["SiLU", "LeakyReLU", "ELU", "Negative slope", "Alpha"] {
            XCTAssertTrue(lab.matches(section: "Library", search: term), term)
        }
    }
    func testEachAuthoredExampleHasOneFactoryAndThemedArtwork() throws {
        for id in ConfigurableActivationRuntimeCapabilities.operations.sorted() {
            let entries = LearnCatalogContent.componentExampleEntries.filter { $0.componentID == id }
            XCTAssertEqual(entries.count, 1)
            let entry = try XCTUnwrap(entries.first)
            let component = try XCTUnwrap(ComponentRegistry.builtIn().component(type: id))
            XCTAssertTrue(entry.item.matches(section: "Examples", search: component.title))
            XCTAssertEqual(LearnCatalogContent.displayedComponentExampleEntries.filter { $0.componentID == id }.count, 1)
            XCTAssertEqual(ComponentVisualFamily.resolve(type: id), .activations)
            let document = try PreparedWorkflowExample.component(id).document(device: "cpu")
            XCTAssertEqual(document.nodes.filter {
                if case .modelPrimitive(let type, _, _) = $0.payload { return type == id }
                return false
            }.count, 1)
        }
    }
}
