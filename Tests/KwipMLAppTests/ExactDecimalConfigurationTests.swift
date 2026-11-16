import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ExactDecimalConfigurationTests:XCTestCase {
    func testSummaryUsesTextContractAndKeepsMissingPolicy()throws {
        let config:[String:JSONValue] = ["groupBy":.array([]),"nullKeys":.string("group"),"maxRows":.integer(100),
            "aggregations":.array([.object(["op":.string("sumDecimal"),"column":.string("amount"),"output":.string("total"),"nullPolicy":.string("skip")])])]
        let schema=[DataColumnSchema(name:"amount",dtype:"string",nullable:true)]
        XCTAssertEqual(try AggregateTableConfiguration.validate(config,schema:schema),config)
        let output=try AggregateTableConfiguration.outputSchema(config,schema:schema)
        XCTAssertEqual(output.map(\.dtype),["string"])
        XCTAssertEqual(output.map(\.nullable),[true])
        for dtype in ["float64","int64","bool"] {
            XCTAssertThrowsError(try AggregateTableConfiguration.validate(config,schema:[.init(name:"amount",dtype:dtype,nullable:false)]))
        }
        var row=SummaryReducerDraft();row.op="sumDecimal"
        XCTAssertEqual(row.eligible(schema+[.init(name:"rounded",dtype:"float64",nullable:false)]).map(\.name),["amount"])
    }

    func testDecimalAndJoinedTextSummaryEvidenceRemainsVisible()throws {
        for op in ["sumDecimal","joinText"] {
            let evidence:[String:Any] = ["kind":"aggregate","inputRows":3,"outputRows":1,"droppedKeyRows":0,
                "reducers":[["output":"total","op":op,"nullInputRows":0]]]
            XCTAssertEqual(try XCTUnwrap(AggregateResultSummary(evidence:evidence)).reducers.first?.op,op)
        }
    }
}
