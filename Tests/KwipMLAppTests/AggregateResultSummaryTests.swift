import XCTest
@testable import KwipMLApp

final class AggregateResultSummaryTests:XCTestCase {
    var evidence:[String:Any] {["kind":"aggregate","inputRows":192,"outputRows":64,"droppedKeyRows":3,"reducers":[["output":"Mean signal","op":"mean","nullInputRows":2],["output":"Observations","op":"countRows","nullInputRows":0]]]}
    func testMeasuredCountsAndReducerNullsArePresentedWithoutInferringFromSample()throws {
        let value=try XCTUnwrap(AggregateResultSummary(evidence:evidence))
        XCTAssertEqual(value.inputRows,192);XCTAssertEqual(value.outputRows,64)
        XCTAssertEqual(value.retainedRows,189);XCTAssertEqual(value.droppedKeyRows,3)
        XCTAssertEqual(value.reducers.map(\.output),["Mean signal","Observations"])
        XCTAssertEqual(value.reducers.map(\.nullInputRows),[2,0])
    }
    func testImpossibleOrMalformedMeasuredCountsDoNotBecomeClaims() {
        for (key,value) in [("inputRows",true as Any),("outputRows",193),("outputRows",0),("droppedKeyRows",-1),("droppedKeyRows",192),("inputRows",2.5)] {
            var next=evidence;next[key]=value
            XCTAssertNil(AggregateResultSummary(evidence:next),key)
        }
        var next=evidence;next["reducers"]=[["output":"x","op":"mean","nullInputRows":190]]
        XCTAssertNil(AggregateResultSummary(evidence:next))
        next["reducers"]=[["output":"x","op":"countRows","nullInputRows":1]]
        XCTAssertNil(AggregateResultSummary(evidence:next))
    }
    func testLegacyCountsCanRenderWithoutInventingReducerEvidence()throws {
        var next=evidence;next.removeValue(forKey:"reducers")
        XCTAssertTrue(try XCTUnwrap(AggregateResultSummary(evidence:next)).reducers.isEmpty)
    }
}
