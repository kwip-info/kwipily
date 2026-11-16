import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCanvasProgressTests:XCTestCase {
    private let source=WorkflowNodeAddress(domain:.data,localID:"s")
    private let cast=WorkflowNodeAddress(domain:.data,localID:"c")
    func testOnlyActualStartsActivateTheirIncomingEdgesAndCompletionClears() {
        let token=UUID()
        let edge=WorkflowEdge(id:"input",from:.init(address:source,port:"table"),to:.init(address:cast,port:"table"),kind:.table)
        var feed=WorkflowCanvasProgress(context:token,allowedNodes:[source,cast],edges:[edge])
        XCTAssertTrue(feed.start(cast,stage:.preparing,context:token,sequence:1))
        XCTAssertEqual(feed.activeNodeAddresses,[cast]);XCTAssertEqual(feed.activeEdgeIDs,["input"])
        XCTAssertTrue(feed.complete(cast,disposition:.reused,context:token,sequence:2))
        XCTAssertTrue(feed.activeNodeAddresses.isEmpty);XCTAssertTrue(feed.activeEdgeIDs.isEmpty)
        XCTAssertEqual(feed.completed[cast],.reused)
        feed.clear()
        XCTAssertFalse(feed.start(source,stage:.preparing,context:token,sequence:3))
    }
    func testStaleUnknownDuplicateAndOutOfOrderEventsCannotMoveHighlight() {
        let token=UUID()
        var feed=WorkflowCanvasProgress(context:token,allowedNodes:[source,cast],edges:[])
        XCTAssertTrue(feed.start(source,stage:.capturing,context:token,sequence:1))
        XCTAssertFalse(feed.start(cast,stage:.preparing,context:UUID(),sequence:2))
        XCTAssertFalse(feed.start(cast,stage:.preparing,context:token,sequence:1))
        XCTAssertFalse(feed.start(.init(domain:.model,localID:"unknown"),stage:.preparing,context:token,sequence:2))
        XCTAssertFalse(feed.complete(cast,disposition:.computed,context:token,sequence:2))
        XCTAssertEqual(feed.activeNodeAddresses,[source])
    }
}
