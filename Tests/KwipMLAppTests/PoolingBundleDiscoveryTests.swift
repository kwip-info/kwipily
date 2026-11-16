import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PoolingBundleDiscoveryTests: XCTestCase {
    func testOneAssessedLabIsSearchableByEveryWindowOperation() throws {
        let matches = LearnCatalogContent.items.filter { $0.id == "pooling" }
        XCTAssertEqual(matches.count, 1)
        let lab = try XCTUnwrap(matches.first)
        XCTAssertEqual(lab.kind, "Construction")
        for title in ["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d"] {
            XCTAssertTrue(lab.matches(section: "Library", search: title))
        }
    }
    func testAllFourKeepOneAuthoredExampleAndActualPreparedFactory() throws {
        for id in ["max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d"] {
            let entries = LearnCatalogContent.componentExampleEntries.filter { $0.componentID == id }
            XCTAssertEqual(entries.count, 1)
            let entry = try XCTUnwrap(entries.first)
            let component = try XCTUnwrap(ComponentRegistry.builtIn().component(type: id))
            XCTAssertTrue(entry.item.matches(section: "Examples", search: component.title))
            XCTAssertEqual(entry.learning.exampleID, "prepared-" + id.replacingOccurrences(of: "_", with: "") + "-classifier")
            XCTAssertEqual(LearnCatalogContent.displayedComponentExampleEntries.filter { $0.componentID == id }.count, 1)
            let example = PreparedWorkflowExample.component(id)
            XCTAssertEqual(example.observationsResource, "workflow-observations")
            let document = try example.document(device: "cpu")
            XCTAssertEqual(document.nodes.filter {
                if case .modelPrimitive(let type, _, _) = $0.payload { return type == id }
                return false
            }.count, 1)
        }
    }
}
