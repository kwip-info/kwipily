import XCTest
import KwipMLCore
@testable import KwipMLApp

final class StatefulNormalizationLessonAudienceTests: XCTestCase {
    func testEveryActualCriterionHasSpecificTeachingFeedback() throws {
        for task in StatefulNormalizationLearningTask.allCases {
            let plan = try StatefulNormalizationAssessment.plan(task: task, runtimeVersion: "2.14.1", seed: 41)
            var titles = Set<String>()
            for criterion in plan.criteria {
                let feedback = try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: criterion.id), "Missing \(task.rawValue) / \(criterion.id)")
                XCTAssertTrue(titles.insert(feedback.title).inserted, "Checks must have distinct readable titles")
                XCTAssertFalse(feedback.title.contains("Construction check"))
                XCTAssertTrue(feedback.nextStep.contains("Check construction"))
                if criterion.id.contains("Gradient") { XCTAssertTrue(feedback.nextStep.lowercased().contains("influence")) }
                if criterion.id.hasPrefix("postEval") { XCTAssertTrue(feedback.nextStep.lowercased().contains("unchanged")) }
            }
            XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: "unknown"))
            if !task.remembered { XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: "runningMean")) }
            if !task.affine { XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation: task.rawValue, criterionID: "scaleGradient")) }
        }
    }
    func testTenGuidesUseSpecificBuildConfigureCheckAndDefinedTerms() throws {
        for task in StatefulNormalizationLearningTask.allCases {
            let guide = LessonAudienceGuide.make(operation: task.rawValue)
            XCTAssertEqual(guide.goal, task.title)
            XCTAssertEqual(guide.steps.count, 3)
            XCTAssertTrue(guide.steps[0].contains(task.componentTitle))
            XCTAssertTrue(guide.steps[1].contains("Channels"))
            XCTAssertFalse(guide.steps[1].contains("An observation is"))
            XCTAssertTrue(guide.steps[2].contains("Check construction"))
            let terms = Set(guide.terms.map { $0.name.lowercased() })
            XCTAssertTrue(terms.isSuperset(of: ["observation", "batch", "channel", "normalization", "gradient", "training", "evaluation"]))
            if task.remembered { XCTAssertTrue(terms.contains("running statistics")); XCTAssertTrue(terms.contains("calibration")) }
            let requirements = LessonAudienceGuide.requirements(operation: task.rawValue, fallback: "generic fallback")
            for header in ["**Build**", "**Configure**", "**Check**"] { XCTAssertTrue(requirements.contains(header)) }
            XCTAssertTrue(requirements.contains(task.componentTitle))
            XCTAssertFalse(requirements.contains("generic fallback"))
            if task.affine { XCTAssertTrue(requirements.contains("Initialize / reset")); XCTAssertTrue(requirements.contains("learned-adjustment influence")) }
            if task.remembered { XCTAssertTrue(requirements.contains("two supplied")); XCTAssertTrue(requirements.contains("unchanged")) }
        }
    }
}
