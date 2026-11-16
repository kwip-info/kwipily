import XCTest
import KwipMLCore
@testable import KwipMLApp
final class FormulaSampleContractTests:XCTestCase {
    func testSampleRequestRoundTripsAndRefusesExpandedScope()throws {
        var query=TableViewQuery(schemaVersion:4,offset:0,limit:8,search:"",sourceSHA256:String(repeating:"a",count:64),indexSHA256:String(repeating:"b",count:64),formula:FormulaDraft.defaults)
        XCTAssertEqual(try TableViewQuery.decode(JSONEncoder().encode(query)),query)
        query.limit=9;XCTAssertThrowsError(try query.validate());query.limit=8
        query.offset=1;XCTAssertThrowsError(try query.validate());query.offset=0
        query.search="x";XCTAssertThrowsError(try query.validate());query.search=""
        query.sortColumn=0;XCTAssertThrowsError(try query.validate());query.sortColumn=nil
        query.schemaVersion=3;XCTAssertThrowsError(try query.validate())
    }
    func testOnlySelectedFormulaDirectInputCanBeSampled()throws {
        let document=try FormulaPracticeExample.document(),consumer=FormulaPracticeExample.address
        let input=try XCTUnwrap(document.edges.first{$0.to.address == consumer}).from
        let selection=WorkflowSelection(nodeAddresses:[consumer])
        XCTAssertTrue(FormulaSampleScope.permits(consumer:consumer,input:input,document:document,selection:selection))
        XCTAssertFalse(FormulaSampleScope.permits(consumer:consumer,input:.init(address:consumer,port:"table"),document:document,selection:selection))
        XCTAssertFalse(FormulaSampleScope.permits(consumer:consumer,input:input,document:document,selection:.init(nodeAddresses:[input.address])))
        var wrong=document;let i=try XCTUnwrap(wrong.nodes.firstIndex{$0.address==consumer});wrong.nodes[i].payload = .dataOperator(operatorID:"kwip.data.view",config:[:])
        XCTAssertFalse(FormulaSampleScope.permits(consumer:consumer,input:input,document:wrong,selection:selection))
    }
}
