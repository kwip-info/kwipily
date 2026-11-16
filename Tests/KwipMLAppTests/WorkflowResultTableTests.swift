import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class WorkflowResultTableTests:XCTestCase {
    func testNumericOrderMissingAndSelectionIdentityPreserveEvidence() {
        let columns=WorkflowResultTableData.columns(["Name","Elapsed"],numeric:[1])
        let rows=[WorkflowResultTableData.row(0,["Ten","10"]),WorkflowResultTableData.row(1,["Two","2"]),WorkflowResultTableData.row(2,["Missing",nil])]
        XCTAssertEqual(WorkflowResultTableData.sorted(rows,columns:columns,sort:.init(column:1)).map(\.index),[1,0,2])
        XCTAssertEqual(WorkflowResultTableData.sorted(rows,columns:columns,sort:.init(column:1,descending:true)).map(\.index),[0,1,2])
        XCTAssertEqual(rows[0].cells[1].text,"10");XCTAssertTrue(rows[2].cells[1].isAbsent)
        XCTAssertEqual(WorkflowResultTableData.operationName("id",graph:.init(nodes:[.init(id:"id",type:"linear",name:"")])),"Linear")
    }
}
