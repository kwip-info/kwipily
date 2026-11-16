import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientPublicationTests: XCTestCase {
    func testHistoricalTasksRemainReadableAndCurrentDiscoveryTeachesInlineTrain() throws {
        for id in ["gradient-controls","gradient-examples"] {
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:id))
            XCTAssertFalse(LearnCatalogContent.items.contains {$0.id==id})
        }
        let current=try XCTUnwrap(LearnCatalogContent.items.first {$0.id=="train-data-access"})
        XCTAssertTrue(current.matches(section:"Practice",search:"Train gradients trainability"))
        for task in GradientLearningTask.allCases {
            XCTAssertTrue(LearnCatalogContent.historicalItems.contains { $0.category == "Training & evaluation" && $0.kind == "Practice" && $0.matches(section: "Practice", search: task.title) })
            XCTAssertTrue(LearnCatalogContent.historicalItems.contains { $0.category == "Training & evaluation" && $0.kind == "Example" && $0.matches(section: "Examples", search: "prepared-"+task.rawValue) })
            let guide = LessonAudienceGuide.make(operation: task.rawValue)
            XCTAssertEqual(guide.goal, task.title)
            for criterion in try task.measuredShapes(widths: [2,2,1]).keys {
                let feedback = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: criterion))
                XCTAssertFalse(feedback.title == "Construction check"); XCTAssertFalse(feedback.nextStep.isEmpty)
            }
        }
    }
    func testOriginalOliveAssetsAndPublishedControlRegistry() throws {
        for task in GradientLearningTask.allCases {
            let example = PreparedWorkflowExample.component(task.rawValue)
            XCTAssertEqual(example.observationsResource, "gradient-observations")
            XCTAssertEqual(try example.document(device: "cpu").schemaVersion, 15)
        }
        let registry = try ComponentRegistry.builtIn()
        XCTAssertNotNil(registry.component(type: "gradient_policy")); XCTAssertNotNil(registry.component(type: "trainability"))
        XCTAssertEqual(ComponentArtworkCatalog.assets["gradient_policy"], "gradient-policy-v1")
        XCTAssertEqual(ComponentArtworkCatalog.assets["trainability"], "trainability-v1")
    }
}
