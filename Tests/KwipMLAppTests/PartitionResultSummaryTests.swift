import XCTest
@testable import KwipMLApp

final class PartitionResultSummaryTests:XCTestCase {
    var evidence:[String:Any] {["kind":"partition","algorithm":"kwip.groupSplit.v1","inputRows":100,"excludedRows":0,"partitions":[["role":"training","rows":60,"groups":6],["role":"validation","rows":20,"groups":2],["role":"test","rows":20,"groups":2]]]}
    func testMeasuredGroupCountsRemainDistinctFromRowCounts()throws {
        let summary=try XCTUnwrap(PartitionResultSummary(evidence:evidence))
        XCTAssertEqual(summary.inputRows,100);XCTAssertEqual(summary.excludedRows,0)
        XCTAssertEqual(summary.partitions.map(\.rows),[60,20,20])
        XCTAssertEqual(summary.partitions.map(\.groups),[6,2,2])
    }
    func testExcludedRowsAndChronologicalEvidence()throws {
        var value=evidence;value["algorithm"]="kwip.temporalSplit.v1";value["inputRows"]=103;value["excludedRows"]=3
        value["partitions"]=[["role":"training","rows":60,"minTimeMicroseconds":-1000,"maxTimeMicroseconds":0],["role":"validation","rows":20,"minTimeMicroseconds":1,"maxTimeMicroseconds":2],["role":"test","rows":20,"minTimeMicroseconds":3,"maxTimeMicroseconds":4]]
        let summary=try XCTUnwrap(PartitionResultSummary(evidence:value))
        XCTAssertEqual(summary.excludedRows,3);XCTAssertTrue(summary.partitions.allSatisfy {$0.groups==nil})
        var parts=value["partitions"] as! [[String:Any]];parts[1]["minTimeMicroseconds"]=0;value["partitions"]=parts
        XCTAssertNil(PartitionResultSummary(evidence:value),"Equal instants cannot span partitions.")
    }
    func testMalformedOrUnaccountedCountsDoNotBecomeClaims() {
        for (key,item) in [("inputRows",true as Any),("inputRows",101),("excludedRows",-1),("inputRows",100.5),("algorithm","unknown")] {
            var value=evidence;value[key]=item;XCTAssertNil(PartitionResultSummary(evidence:value),key)
        }
        var value=evidence;var parts=value["partitions"] as! [[String:Any]]
        parts[2]["role"]="validation";value["partitions"]=parts;XCTAssertNil(PartitionResultSummary(evidence:value))
    }
}
