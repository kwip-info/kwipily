import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentDraftTests:XCTestCase {
    func testTypedValuesOrderAndBudgetRoundTrip()throws {
        var draft=WorkflowExperimentDraft()
        draft.values=["seed":"7, 8", "learningRate":"0.01, 0.001"]
        draft.maximumRuns="4";draft.maximumTotalEpochs="120"
        let spec=try draft.specification()
        XCTAssertEqual(spec.axes.map(\.field),["seed","learningRate"])
        XCTAssertEqual(spec.axes[0].values,[.integer(7),.integer(8)])
        XCTAssertEqual(spec.axes[1].values,[.number(0.01),.number(0.001)])
        XCTAssertEqual(try WorkflowExperimentDraft(specification:spec).specification(),spec)
        XCTAssertEqual(try draft.summary(baseEpochs:30).runs,4)
        XCTAssertEqual(try draft.summary(baseEpochs:30).epochs,120)
        XCTAssertThrowsError(try draft.summary(baseEpochs:31))
    }
    func testMalformedDuplicateUnsupportedAndOversizedValuesReject()throws {
        for value in ["1,,2","1,","1.5","-1","4294967296","true","1,1"] {
            var draft=WorkflowExperimentDraft();draft.values=["seed":value]
            XCTAssertThrowsError(try draft.specification(),value)
        }
        var draft=WorkflowExperimentDraft();draft.values=["learningRate":"1,1.0"]
        XCTAssertThrowsError(try draft.specification())
        draft.values=["device":"cpu,unknown"]
        XCTAssertThrowsError(try draft.specification())
        draft.values=["bogus":"1"]
        XCTAssertThrowsError(try draft.specification())
        draft.values=["seed":String(repeating:"1",count:4097)]
        XCTAssertThrowsError(try draft.specification())
    }
    func testEpochAxisUsesEachVariantAndRejectsExcessCombinations()throws {
        var draft=WorkflowExperimentDraft();draft.values=["epochs":"2,3","seed":"7,8"]
        draft.maximumRuns="4";draft.maximumTotalEpochs="10"
        XCTAssertEqual(try draft.summary(baseEpochs:999).epochs,10)
        draft.maximumRuns="3"
        XCTAssertThrowsError(try draft.specification())
        draft.maximumRuns="4";draft.maximumTotalEpochs="9"
        XCTAssertThrowsError(try draft.summary(baseEpochs:1))
    }
}
