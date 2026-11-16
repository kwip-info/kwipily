import XCTest
@testable import KwipMLApp

final class DropoutInspectorDraftTests:XCTestCase {
    func testDefaultAndInclusiveEndpointsHaveHonestChanceLabels()throws {
        let initial=DropoutInspectorDraft()
        XCTAssertEqual(initial.numeric,"0.5")
        XCTAssertEqual(initial.probability,0.5)
        XCTAssertEqual(initial.percentageLabel,"50% drop chance")
        var draft=DropoutInspectorDraft(probability:0)
        XCTAssertEqual(draft.probability,0)
        XCTAssertEqual(draft.percentageLabel,"0% drop chance")
        draft.numeric="1"
        XCTAssertEqual(draft.probability,1)
        XCTAssertEqual(draft.percentageLabel,"100% drop chance")
    }
    func testInvalidTypedTextStaysVisibleUntilAnExplicitCorrection()throws {
        var draft=DropoutInspectorDraft(probability:0.5)
        for invalid in ["", "-0.001", "1.001", "nan", "NaN", "inf", "-inf", "Infinity", "1e309", "not a probability"] {
            draft.numeric=invalid
            XCTAssertNil(draft.probability,invalid)
            XCTAssertNil(draft.percentageLabel,invalid)
            XCTAssertEqual(draft.numeric,invalid,"Validation must not clamp or silently replace typed text")
        }
        draft.numeric="0.4"
        XCTAssertEqual(draft.probability,0.4)
        XCTAssertEqual(draft.percentageLabel,"40% drop chance")
        XCTAssertEqual(draft.numeric,"0.4")
    }
    func testScientificNotationAndFineProbabilityPrecisionRemainAvailable()throws {
        var draft=DropoutInspectorDraft(probability:0.5)
        draft.numeric="5e-1"
        XCTAssertEqual(draft.probability,0.5)
        XCTAssertEqual(draft.percentageLabel,"50% drop chance")
        XCTAssertEqual(draft.numeric,"5e-1")
        draft.numeric="1e-4"
        XCTAssertEqual(try XCTUnwrap(draft.probability),0.0001,accuracy:1e-16)
        let precise=0.123456789012345
        let initialized=DropoutInspectorDraft(probability:precise)
        XCTAssertEqual(try XCTUnwrap(initialized.probability),precise,accuracy:1e-15)
        draft.setProbability(precise)
        XCTAssertEqual(try XCTUnwrap(draft.probability),precise,accuracy:1e-15,"Moving the slider must not quantize the authored probability to displayed percent precision")
    }
    func testSliderClampsItsOwnRangeAndSynchronizesTheNumericDraft()throws {
        var draft=DropoutInspectorDraft(probability:0.5)
        draft.numeric="unfinished"
        draft.setProbability(0.25)
        XCTAssertEqual(draft.probability,0.25)
        XCTAssertEqual(Double(draft.numeric),0.25)
        XCTAssertEqual(draft.percentageLabel,"25% drop chance")
        draft.setProbability(0.35+0.1)
        XCTAssertEqual(draft.numeric,"0.45","Slider arithmetic should not spill floating-point noise into the editable field")
        draft.setProbability(-0.2)
        XCTAssertEqual(draft.probability,0)
        XCTAssertEqual(Double(draft.numeric),0)
        draft.setProbability(1.2)
        XCTAssertEqual(draft.probability,1)
        XCTAssertEqual(Double(draft.numeric),1)
        XCTAssertEqual(draft.percentageLabel,"100% drop chance")
    }
}
