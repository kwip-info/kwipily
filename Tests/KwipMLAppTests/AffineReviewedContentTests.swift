import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class AffineReviewedContentTests:XCTestCase {
    func testReviewedActivationReadingKeepsStableStepsAndTaskSpecificDirections()throws {
        for identity:ModelConstructionLessonIdentity in [.tanh,.sigmoid,.geluExact,.geluTanh] {
            let copy=try XCTUnwrap(AffineRemainingActivationCopy.packet(identity))
            XCTAssertEqual(copy.steps.map(\.id),["goal","try.0","inspect"])
            XCTAssertEqual(Set(copy.reference.map(\.id)).count,7)
            XCTAssertTrue(copy.steps[0].blocks[0].text.contains("Input"))
            XCTAssertTrue(copy.steps[0].blocks[0].text.contains("Result"))
            XCTAssertTrue(copy.steps[1].blocks[0].text.contains("Submit assessment"))
            XCTAssertTrue(copy.steps[2].blocks[0].text.contains("receipt saved"))
            XCTAssertFalse(copy.steps.flatMap(\.blocks).contains {$0.targetType != nil || $0.visual != nil},"Reading instructions must not create an executable action or pretend to be captured output")
        }
        XCTAssertNil(AffineRemainingActivationCopy.packet(.composition))
        let exact=try XCTUnwrap(AffineRemainingActivationCopy.packet(.geluExact))
        let approximate=try XCTUnwrap(AffineRemainingActivationCopy.packet(.geluTanh))
        XCTAssertTrue(exact.steps[1].blocks[0].text.contains("Approximation to Exact"))
        XCTAssertTrue(approximate.steps[1].blocks[0].text.contains("choose Tanh approximation in Approximation"))
        XCTAssertTrue(try XCTUnwrap(exact.reference.first {$0.id=="reference.notation"}).text.contains(LearningExamples.geluReference(approximateTanh:false).formula))
        XCTAssertTrue(try XCTUnwrap(approximate.reference.first {$0.id=="reference.notation"}).text.contains(LearningExamples.geluReference(approximateTanh:true).formula))
    }
    func testPublicIllustrationsAgreeWithIndependentScalarReferences()throws {
        let tanh=try XCTUnwrap(AffineRemainingActivationCopy.packet(.tanh)?.reference.first {$0.id=="reference.examples"})
        let sigmoid=try XCTUnwrap(AffineRemainingActivationCopy.packet(.sigmoid)?.reference.first {$0.id=="reference.examples"})
        for value:Double in [-2,1,2] {
            let formatter=NumberFormatter();formatter.locale=Locale(identifier:"en_US_POSIX");formatter.minimumFractionDigits=6;formatter.maximumFractionDigits=6
            let t=try XCTUnwrap(formatter.string(from:NSNumber(value:try LearningExamples.tanhReference.apply(value)))).replacingOccurrences(of:"-",with:"−")
            let s=try XCTUnwrap(formatter.string(from:NSNumber(value:try LearningExamples.sigmoidReference.apply(value))))
            XCTAssertTrue(tanh.text.contains(t),t)
            XCTAssertTrue(sigmoid.text.contains(s),s)
        }
        XCTAssertTrue(try XCTUnwrap(AffineRemainingActivationCopy.packet(.tanh)?.reference.first {$0.id=="reference.inspect"}).text.contains("Applying Tanh twice"))
        XCTAssertTrue(try XCTUnwrap(AffineRemainingActivationCopy.packet(.sigmoid)?.reference.first {$0.id=="reference.inspect"}).text.contains("not automatically calibrated probabilities"))
        XCTAssertEqual(AffineReLULessonCopy.steps.map(\.id),["goal","try.0","inspect"])
    }
}
