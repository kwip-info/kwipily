import XCTest
import KwipMLCore
@testable import KwipMLApp
final class ConfigurableActivationLessonAudienceTests:XCTestCase {
    func testAllThreeAccessibleRequiredControlsAndDefinedTerms() {
        for task in ConfigurableActivationLearningTask.allCases {
            let text=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for term in ["**Configure**","Select","Check construction","automatically"] {XCTAssertTrue(text.contains(term),term)}
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["array","activation","gradient"]))
        }
        XCTAssertTrue(ConfigurableActivationLearningTask.leakyReLU.instructions.contains("0.2"))
        XCTAssertTrue(ConfigurableActivationLearningTask.elu.instructions.contains("2"))
    }
}
