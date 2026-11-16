import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class Data02PreparationHistoryCatalogTests:XCTestCase {
    func testPreparationHistoryHasOwnSearchableTargetWithoutClaimingModelOrOpeningFiles() {
        let project=UUID().uuidString,operation=UUID().uuidString
        let entry=WorkflowDataPreparationHistory.Entry(id:operation,completedAt:Date(timeIntervalSince1970:100),document:.init(projectID:project,name:"Customer cleanup"),requestIdentity:String(repeating:"a",count:64),artifactDirectory:URL(fileURLWithPath:"/must-not-open"),manifestSHA256:String(repeating:"b",count:64),savedOutputs:[["rowCount":12]],artifacts:[])
        let rows=HistoryTableCatalog.rows(preparations:[entry])
        XCTAssertEqual(rows.count,1)
        XCTAssertEqual(rows[0].target,.preparation(operation))
        XCTAssertEqual(rows[0].kind,.preparation)
        XCTAssertEqual(rows[0].title,"Customer cleanup")
        XCTAssertEqual(rows[0].status,.completed)
        XCTAssertTrue(rows[0].evidenceLabel.contains("recheck"))
        XCTAssertFalse(rows[0].typeLabel.localizedCaseInsensitiveContains("model"))
        XCTAssertEqual(HistoryTableCatalog.filtered(rows,kind:.preparation,query:operation).count,1)
        XCTAssertEqual(HistoryTableCatalog.filtered(rows,query:"Customer cleanup").count,1)
        XCTAssertTrue(HistoryTableCatalog.filtered(rows,kind:.training).isEmpty)
    }
}
