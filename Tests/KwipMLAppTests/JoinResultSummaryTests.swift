import XCTest
@testable import KwipMLApp

final class JoinResultSummaryTests: XCTestCase {
    private func evidence(_ mode:String="inner",output:Int=6)->[String:Any] {
        ["kind":"join","mode":mode,"leftRows":5,"rightRows":4,
         "matchedPairs":6,"leftUnmatched":2,"rightUnmatched":1,"outputRows":output]
    }
    func testModesExplainWhichUnmatchedRowsWereRetained() throws {
        for (mode,output,retained) in [("inner",6,0),("left",8,2),("right",7,1),("full",9,3)] {
            let summary=try XCTUnwrap(JoinResultSummary(evidence:evidence(mode,output:output)))
            XCTAssertEqual(summary.outputRows,output)
            XCTAssertEqual(summary.retainedUnmatchedRows,retained)
            XCTAssertEqual(summary.discardedUnmatchedRows,3-retained)
            XCTAssertEqual(summary.matchedPairs,6)
        }
    }
    func testMalformedOrInconsistentCountsNeverBecomeExplanatoryMetrics() {
        for field in ["leftRows","rightRows","matchedPairs","leftUnmatched","rightUnmatched","outputRows"] {
            for invalid:Any in [-1,1.5,true,"6"] {
                var value=evidence();value[field]=invalid
                XCTAssertNil(JoinResultSummary(evidence:value))
            }
        }
        XCTAssertNil(JoinResultSummary(evidence:evidence("full",output:6)))
        var impossible=evidence();impossible["leftUnmatched"]=6
        XCTAssertNil(JoinResultSummary(evidence:impossible))
        var other=evidence();other["kind"]="encoder"
        XCTAssertNil(JoinResultSummary(evidence:other))
    }
}
