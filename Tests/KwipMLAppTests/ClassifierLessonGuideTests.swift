import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ClassifierLessonGuideTests: XCTestCase {
    func testEveryCurrentStageHasVisualGuidanceInsteadOfTechnicalFallback() throws {
        let plan = try WorkflowGuidedBuildPlan.parallelClassifier()
        XCTAssertEqual(plan.steps.count, 15)
        for step in plan.steps {
            let text = ClassifierLessonGuide.explanation(for: step.address.localID, fallback: step.explanation)
            XCTAssertNotEqual(text, step.explanation, "Missing beginner guidance for \(step.address.localID)")
            XCTAssertTrue(text.contains("**Try it**"), "Missing canvas action for \(step.address.localID)")
            XCTAssertFalse(text.contains("Σ"))
            XCTAssertFalse(text.contains("μ"))
        }
    }

    func testLegacyFirstUseKeepsItsReviewedInputAndCheckingLanguage() {
        let source = guidance("observations")
        XCTAssertTrue(source.contains("Features are the input columns"))
        XCTAssertTrue(source.contains("target is the answer"))
        let split = guidance("split")
        XCTAssertTrue(split.contains("Validation rows are reserved for checking"))
        XCTAssertTrue(split.contains("seed is a number"))
    }

    func testRunSettingsAndResultsDefineTheirMeaningWithoutPromisingSuccess() {
        let setup = guidance("setup")
        XCTAssertTrue(setup.contains("epoch is one pass"))
        XCTAssertTrue(setup.contains("batch is a small group"))
        XCTAssertTrue(setup.contains("learning rate controls"))
        let train = ClassifierLessonGuide.explanation(for:"train",fallback:"Missing guidance",version:5)
        XCTAssertTrue(train.lowercased().contains("accuracy is the fraction"))
        XCTAssertTrue(train.contains("Loss measures"))
        XCTAssertTrue(train.contains("does not prove accuracy on new data or award assessment completion"))
    }

    func testSharedInputDoesNotClaimSharedWeightsOrProbabilities() {
        XCTAssertTrue(guidance("branch-b").contains("learns its own weights"))
        XCTAssertTrue(guidance("classifier").contains("not probabilities"))
        XCTAssertTrue(guidance("output").contains("Do not add Softmax"))
    }

    func testUnknownStageRetainsItsAuthoredExplanation() {
        XCTAssertEqual(ClassifierLessonGuide.explanation(for: "future-stage", fallback: "Future authored guidance"), "Future authored guidance")
    }

    func testCurrentCopyIsSharedAndLegacyTitlesArePreserved() throws {
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        for step in plan.steps {
            let copy=try XCTUnwrap(GuidedLessonCopy.stages[step.address.localID])
            XCTAssertEqual(ClassifierLessonGuide.explanation(for:step.address.localID,fallback:step.explanation,version:5),copy.text)
            XCTAssertEqual(ClassifierLessonGuide.title(for:step.address.localID,fallback:step.title,version:5),copy.title)
            XCTAssertEqual(ClassifierLessonGuide.title(for:step.address.localID,fallback:step.title,version:4),step.title)
            XCTAssertNotEqual(ClassifierLessonGuide.explanation(for:step.address.localID,fallback:step.explanation,version:4),copy.text)
        }
    }

    private func guidance(_ id: String) -> String {
        ClassifierLessonGuide.explanation(for: id, fallback: "Missing guidance",version:4)
    }
}
