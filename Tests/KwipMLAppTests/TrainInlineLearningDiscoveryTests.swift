import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainInlineLearningDiscoveryTests:XCTestCase {
    private let historical:Set<String> = ["data-loading", "loss-results", "optimizer-updates", "scheduler-rates", "gradient-controls", "gradient-examples", "initialization-starting-settings", "initialization-examples"]
    func testConnectedRecipeLabsRemainReadableButAreAbsentFromCurrentDiscoveryAndPaths() {
        XCTAssertTrue(historical.isSubset(of:LearnCatalogContent.historicalEntryIDs))
        XCTAssertTrue(historical.isDisjoint(with:Set(LearningLibraryPresentation.items.map(\.id))))
        XCTAssertTrue(historical.isDisjoint(with:Set(LearningJourney.all.flatMap {$0.steps+$0.optionalSteps}.map(\.entryID))))
        for id in historical {XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:id),id)}
        let path=LearningJourney.all.first {$0.id == "training-generalization"}
        XCTAssertEqual(path?.steps.first?.entryID,"train-data-access")
        XCTAssertEqual(path?.steps.first?.evidence,.unscoredPractice)
    }
    func testPublishedRecipeExamplesRetainIdentityAndExplainInlineTrainConfiguration() {
        let recipeIDs=Set(BundledOptimizerComponents.entries.keys).union(BundledSchedulerComponents.entries.keys).union(BundledLossComponents.entries.keys)
        let displayed=LearnCatalogContent.displayedComponentExampleEntries.filter {recipeIDs.contains($0.componentID)}
        XCTAssertFalse(displayed.isEmpty)
        for entry in displayed {
            let text=entry.learning.paragraphs.joined(separator:" ")
            XCTAssertTrue(text.contains("Train"),entry.componentID)
            XCTAssertFalse(text.contains("Connect this recipe"),entry.componentID)
            XCTAssertFalse(text.contains("Connect the recipe"),entry.componentID)
            XCTAssertFalse(text.contains("connect an explicit optimizer recipe"),entry.componentID)
            XCTAssertFalse(text.contains("build the scoring connections"),entry.componentID)
            XCTAssertFalse(entry.learning.formula.isEmpty)
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:entry.componentID))
        }
    }
    func testComputationalModelLessonsStayDiscoverable() {
        let visible=Set(LearningLibraryPresentation.items.map(\.id))
        for id in ["affine","tensor-layout","matrix","attention-connections","embedding","embedding-bag","configurable-activations","remaining-activations"] {XCTAssertTrue(visible.contains(id),id)}
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"matmul"),"matrix")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"attention_mask"),"attention-connections")
    }
}
