import XCTest
import KwipMLCore
@testable import KwipMLApp
final class WorkflowProgressPresentationTests:XCTestCase {
    func testPresentationRejectsStaleOwnerAndClearedActivity() {
        let node=WorkflowNodeAddress(domain:.data,localID:"source")
        var feed=WorkflowCanvasProgress(context:UUID(),allowedNodes:[node],edges:[])
        feed.start(node,stage:.capturing,context:feed.context,sequence:1)
        XCTAssertTrue(WorkflowProgressPresentation.isActive(node,progress:feed,isCurrent:true))
        XCTAssertFalse(WorkflowProgressPresentation.isActive(node,progress:feed,isCurrent:false))
        XCTAssertFalse(WorkflowProgressPresentation.isActive(.init(domain:.model,localID:"source"),progress:feed,isCurrent:true))
        feed.clear()
        XCTAssertFalse(WorkflowProgressPresentation.isActive(node,progress:feed,isCurrent:true))
    }
}
