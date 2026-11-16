import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingBundleDiscoveryTests:XCTestCase {
    func testAuthoredLookupFindsCategoricalFactoryAndAllThreeAssessedTasksWithoutDuplicateCards()throws {
        for id in ["embedding","index_input"] {
            let entry=try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first {$0.componentID == id})
            XCTAssertEqual(entry.learning.exampleID,"prepared-embedding-classifier")
            XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains {$0.componentID == id})
            let example=PreparedWorkflowExample.component(id)
            XCTAssertEqual(example.observationsResource,"embedding-observations")
            let document=try example.document(device:"cpu")
            XCTAssertEqual(document.nodes.filter {if case .modelPrimitive(let type,_,_)=$0.payload {return type == "embedding"};return false}.count,1)
            XCTAssertEqual(document.nodes.filter {if case .modelPrimitive(let type,_,_)=$0.payload {return type == "index_input"};return false}.count,1)
            let descriptor=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:.builtIn()).first {$0.id == id})
            XCTAssertEqual(descriptor.learning.lessonIDs,["assessment.embedding-gather","assessment.embedding-padding-frequency","assessment.embedding-renormalize"])
        }
    }
}
