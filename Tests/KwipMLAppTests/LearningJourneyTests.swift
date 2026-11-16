import XCTest
@testable import KwipMLApp

@MainActor final class LearningJourneyTests: XCTestCase {
    func testSixJourneysPreserveReviewedOrderAndExistingEntryRoutes() {
        let expected = [
            ["guided", "parallel", "preprocess"],
            ["data-preparation-course", "columns", "filter", "formula", "preprocess", "table-output", "tensor-layout"],
            ["relu-lesson", "sigmoid-lesson", "tanh-lesson", "configurable-activations", "parallel", "module-composition", "module-composition-examples"],
            ["embedding", "matrix", "softmax", "recurrent-memory", "attention-connections"],
            ["train-data-access", "holdout"],
            ["checkpoint-lifecycle", "checkpoint-examples", "train-data-access", "embedding-bag", "program-export", "iteration-transfer"]
        ]
        XCTAssertEqual(LearningJourney.all.map { $0.steps.map(\.entryID) }, expected)
        XCTAssertEqual(Set(LearningJourney.all.map(\.id)).count, 6)
        let routes = Set(LearnCatalogContent.items.map(\.id)).union(LearnCatalogContent.historicalEntryIDs)
        for journey in LearningJourney.all {
            XCTAssertFalse(journey.objective.isEmpty)
            XCTAssertFalse(journey.prerequisites.isEmpty)
            XCTAssertFalse(journey.runtimeGuidance.isEmpty)
            for step in journey.steps + journey.optionalSteps {
                XCTAssertTrue(routes.contains(step.entryID), "Missing Learn route: \(step.entryID)")
                XCTAssertFalse(step.advice.isEmpty)
            }
        }
    }

    func testGuidanceDoesNotPromoteUnscoredExamplesOrMakeOptionalBagWorkMandatory() throws {
        let data = try XCTUnwrap(LearningJourney.all.first { $0.id == "data-shapes" })
        let currentTablePracticeIDs:Set<String>=["data-preparation-course","columns","filter","formula","preprocess","table-output"]
        XCTAssertEqual(Set(data.steps.filter {$0.evidence == .unscoredPractice}.map(\.entryID)),currentTablePracticeIDs)
        XCTAssertTrue(data.steps.allSatisfy {!LearnCatalogContent.historicalEntryIDs.contains($0.entryID)})
        let transfer = try XCTUnwrap(LearningJourney.all.first { $0.id == "saved-transfer" })
        XCTAssertTrue(try XCTUnwrap(transfer.steps.first { $0.entryID == "embedding-bag" }).isOptional)
        XCTAssertEqual(try XCTUnwrap(transfer.steps.first { $0.entryID == "iteration-transfer" }).evidence, .unscoredPractice)
        XCTAssertTrue(transfer.runtimeGuidance.contains("CPU"))
        XCTAssertEqual(Set(transfer.optionalSteps.map(\.entryID)), ["sklearn", "tensorflow"])
        XCTAssertTrue(transfer.optionalSteps.allSatisfy { $0.evidence == .unscoredPractice })
        XCTAssertTrue(LearningJourney.all.allSatisfy { !$0.requiresCompletionGate })
    }

    func testJourneySearchIncludesRouteAdviceAndOptionalBranches() {
        let columns=LearningJourney.matching("COLUMNS").map(\.id)
        XCTAssertTrue(columns.contains("data-shapes"))
        XCTAssertEqual(columns.count,Set(columns).count,"Search returns each matching journey once")
        XCTAssertEqual(LearningJourney.matching("parse real date/time").map(\.id), ["data-shapes"])
        XCTAssertEqual(LearningJourney.matching("tensorflow").map(\.id), ["saved-transfer"])
        XCTAssertEqual(LearningJourney.matching("scaled attention").map(\.id), ["sequence-attention"])
        XCTAssertTrue(LearningJourney.matching("not-an-existing-topic").isEmpty)
        XCTAssertEqual(LearningJourney.matching("  ").count, 6)
    }
}
