import XCTest
import KwipMLCore
@testable import KwipMLApp

final class IterationTransferLessonContentTests: XCTestCase {
    func testSixTopicsHaveVisualActionsObservationsAndHonestLimits() {
        let topics = IterationTransferLessonContent.topics
        XCTAssertEqual(Set(topics.map(\.id)), ["run", "experiments", "checkpoint", "export", "blocks", "diagnostics"])
        XCTAssertEqual(topics.count, 6)
        for topic in topics {
            XCTAssertFalse(topic.goal.isEmpty, topic.id)
            XCTAssertFalse(topic.steps.isEmpty, topic.id)
            XCTAssertFalse(topic.observation.isEmpty, topic.id)
            XCTAssertFalse(topic.limits.isEmpty, topic.id)
        }
        XCTAssertTrue(IterationTransferLessonContent.assessmentDisclosure.contains("no automatically scored"))
    }
    func testRunGuideNamesActualControls() {
        let run = topic("run")
        let text = run.steps.joined(separator: " ")
        for field in WorkflowControlContracts.runFields {
            XCTAssertTrue(text.contains(field.title), field.key)
        }
        XCTAssertTrue(run.limits.contains("Captured preview limits"))
    }
    func testCheckpointGuideRetainsModeAndCompatibilityDistinctions() {
        let checkpoint = topic("checkpoint")
        let text = checkpoint.steps.joined(separator: " ")
        for mode in WorkflowTrainingMode.allCases { XCTAssertTrue(text.contains(mode.title)) }
        XCTAssertTrue(checkpoint.limits.contains("CPU"))
        XCTAssertTrue(checkpoint.limits.contains("architecture, runtime and data"))
        XCTAssertTrue(text.contains("total"))
    }
    func testExperimentPracticeUsesValidFiniteTwoRunRecipe() throws {
        let recipe = IterationTransferLessonContent.practiceRecipe
        try recipe.validate()
        XCTAssertEqual(recipe.axes.map(\.field), ["learningRate"])
        XCTAssertEqual(recipe.axes[0].values, [.number(0.03), .number(0.01)])
        XCTAssertEqual(recipe.budget.maximumRuns, 2)
        XCTAssertEqual(recipe.budget.maximumTotalEpochs, 6)
        XCTAssertTrue(topic("experiments").limits.contains("fresh training"))
    }
    func testTransferDoesNotHidePreparationOrConfuseStructureWithWeights() {
        XCTAssertTrue(topic("export").limits.contains("does not include fitted preprocessing"))
        XCTAssertTrue(topic("export").limits.contains("Python"))
        XCTAssertTrue(topic("blocks").limits.contains("does not share learned weights"))
        XCTAssertTrue(topic("blocks").limits.contains("pinned revision"))
        XCTAssertTrue(topic("diagnostics").limits.contains("not proof of a cause"))
        XCTAssertTrue(topic("diagnostics").limits.contains("does not run training"))
    }
    private func topic(_ id: String) -> IterationTransferLessonContent.Topic {
        IterationTransferLessonContent.topics.first { $0.id == id }!
    }
}
