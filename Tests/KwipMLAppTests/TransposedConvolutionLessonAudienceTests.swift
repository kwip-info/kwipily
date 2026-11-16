import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TransposedConvolutionLessonAudienceTests:XCTestCase {
    func testAllThreeExplainOverlapExtentBiasAndActualControls() {
        for task in [ConvolutionLearningTask.transposedOne,.transposedTwo,.transposedThree] {
            let required=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for text in ["**Configure**","Output padding","overlap","bias","automatically","Select"] {XCTAssertTrue(required.contains(text),text)}
            XCTAssertTrue(required.contains("does not append input values or reverse the original calculation"))
            XCTAssertTrue(task.instructions.contains("does not recover the original input"))
            XCTAssertTrue(task.instructions.contains("eight"))
            XCTAssertTrue(ConvolutionLearningTask.allCases.contains(task))
        }
    }
}
