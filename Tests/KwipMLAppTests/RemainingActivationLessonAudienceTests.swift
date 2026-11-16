import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RemainingActivationLessonAudienceTests:XCTestCase {
    func testEveryTaskExplainsItsControlsAndNewTermsWithoutCode() {
        for task in RemainingActivationLearningTask.allCases {
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertFalse(guide.goal.isEmpty)
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["array","channel","gradient","observation"]))
            XCTAssertTrue(guide.steps.joined(separator:" ").contains("Check construction"))
            XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"inputGradient"))
            if task.isPReLU {XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"slopeGradient"))}
            if task == .dropoutEvaluation {XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"trainingOutput"))}
        }
    }
}
