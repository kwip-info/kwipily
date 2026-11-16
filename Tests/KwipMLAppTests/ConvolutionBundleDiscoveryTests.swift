import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConvolutionBundleDiscoveryTests:XCTestCase {
    func testAssessedLabIsUniqueAndSearchableForEveryDimension() throws {
        let cards=LearnCatalogContent.items.filter {$0.id == "convolution"}
        XCTAssertEqual(cards.count,1)
        let card=try XCTUnwrap(cards.first)
        XCTAssertEqual(card.kind,"Construction")
        for task in ConvolutionLearningTask.allCases.filter({ !$0.isTransposed }) {
            XCTAssertTrue(card.matches(section:"Library",search:task.title))
        }
    }
    func testEachDimensionKeepsUniqueDiscoveryAndItsActualFactory() throws {
        for dimension in 1...3 {
            let id="conv\(dimension)d"
            let entry=try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first {$0.componentID == id},id)
            XCTAssertEqual(entry.learning.exampleID,"prepared-"+id+"-classifier")
            XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains {$0.componentID == id})
            let example=PreparedWorkflowExample.component(id)
            XCTAssertEqual(example.observationsResource,"workflow-observations")
            let document=try example.document(device:"cpu")
            XCTAssertEqual(document.nodes.filter {if case .modelPrimitive(let type,_,_)=$0.payload {return type == id};return false}.count,1)
            XCTAssertEqual(LearnCatalogContent.items.filter {$0.id == id && $0.kind == "Example"}.count,1)
        }
    }
}
