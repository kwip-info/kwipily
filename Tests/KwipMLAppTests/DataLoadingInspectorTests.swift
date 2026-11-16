import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DataLoadingInspectorTests:XCTestCase {
    private func fixture()throws->(WorkflowAuthoringStore,WorkflowControlNodeContext) {
        let(owner,authority,_)=try inlineRecipeFixture("optimizer_sgd",role:"optimizer")
        return(owner,try WorkflowControlNodeContext(owner:owner,address:authority.address))
    }
    func testDependentLoaderControlsApplyAsOneUndoableEdit()throws {
        let(owner,context)=try fixture(),initial=owner.document,original=context.configuration
        var draft=try TrainDataAccessDraft(configuration:original)
        draft.batchingEnabled=true;draft.workers="1";draft.timeout="0"
        XCTAssertThrowsError(try draft.configuration(preserving:original))
        XCTAssertEqual(owner.document,initial)
        draft.prefetch="2";draft.timeout="10"
        let next=try draft.configuration(preserving:original)
        let origin=context.numericOrigin(key:"trainDataAccess")
        context.setNumericPending(origin:origin,key:"trainDataAccess",pending:true)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(context.applyTrainDataAccess(next,expected:original))
        context.setNumericPending(origin:origin,key:"trainDataAccess",pending:false)
        guard case .object(let saved)=context.configuration["batching"] else {return XCTFail("Train owns batching")}
        XCTAssertEqual(saved["num_workers"],1);XCTAssertEqual(saved["prefetch_factor"],2)
        XCTAssertEqual(saved["timeout"]?.doubleValue,10)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,initial.nodes)
        XCTAssertFalse(context.applyTrainDataAccess(next,expected:original))
    }
    func testSubsetDraftStaysPendingUntilValidAndRevertPreservesSource()throws {
        let(owner,context)=try fixture(),initial=owner.document,original=context.configuration
        let saved=try TrainDataAccessDraft(configuration:original)
        var draft=saved;draft.accessEnabled=true;draft.kind="subset";draft.indices=""
        XCTAssertThrowsError(try draft.configuration(preserving:original))
        draft.indices="3, 0, 2"
        let next=try draft.configuration(preserving:original)
        guard case .object(let selected)=next["datasetAccess"] else {return XCTFail("Dataset access")}
        XCTAssertEqual(selected["indices"],.array([3,0,2]));XCTAssertEqual(owner.document,initial)
        let origin=context.numericOrigin(key:"trainDataAccess")
        context.setNumericPending(origin:origin,key:"trainDataAccess",pending:true)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        draft=saved;context.setNumericPending(origin:origin,key:"trainDataAccess",pending:false)
        XCTAssertEqual(try draft.configuration(preserving:original),original)
        XCTAssertEqual(owner.document,initial);XCTAssertFalse(owner.hasPendingInspectorEdits)
        context.setNumericPending(origin:"foreign",key:"trainDataAccess",pending:true)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
    }
}
