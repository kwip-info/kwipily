import XCTest
import KwipMLCore
@testable import KwipMLApp

private actor EvaluationHistoryBarrier:WorkflowEvaluationHistoryLoading {
    var listing:CheckedContinuation<WorkflowEvaluationHistoryListing,Error>?
    var selections:[WorkflowEvaluationHistoryReference:CheckedContinuation<WorkflowEvaluationDisclosure.DisclosedEvaluation,Error>]=[:]
    var reached:XCTestExpectation
    init(_ reached:XCTestExpectation) {self.reached=reached}
    func list(root:URL,projectID:String)async throws->WorkflowEvaluationHistoryListing {
        try await withCheckedThrowingContinuation {listing=$0;reached.fulfill()}
    }
    func disclose(_ reference:WorkflowEvaluationHistoryReference,root:URL)async throws->WorkflowEvaluationDisclosure.DisclosedEvaluation {
        try await withCheckedThrowingContinuation {selections[reference]=$0;reached.fulfill()}
    }
    func next(_ expectation:XCTestExpectation) {reached=expectation}
    func releaseList() {listing?.resume(returning:.init(entries:[],issues:["stale listing"],truncated:false));listing=nil}
    func release(_ reference:WorkflowEvaluationHistoryReference,value:WorkflowEvaluationDisclosure.DisclosedEvaluation) {selections.removeValue(forKey:reference)?.resume(returning:value)}
}

@MainActor final class WorkflowEvaluationHistoryStoreTests:XCTestCase {
    private func fixture()throws->(WorkflowEvaluationHistoryReference,WorkflowEvaluationDisclosure.DisclosedEvaluation) {
        let (reference,record)=WorkflowEvaluationDisclosureTests().fixture()
        return(reference,.init(value:try XCTUnwrap(record.result),outcome:record.outcome,priorExposure:.init(priorHistoryUnknown:true,evaluatedCount:1,inspectionCount:0,checkpointInspectionCount:0,sameSourceInspectedOverlap:0,overlapIsComplete:true)))
    }
    func testInvalidationDiscardsSuspendedListing()async throws {
        let reached=expectation(description:"List suspended"),loader=EvaluationHistoryBarrier(reached)
        let store=WorkflowEvaluationHistoryStore(root:URL(fileURLWithPath:"/unused"),projectID:UUID().uuidString,isCurrent:{true},loader:loader)
        store.refresh();await fulfillment(of:[reached],timeout:3)
        store.invalidate();await loader.releaseList()
        for _ in 0..<20 {await Task.yield()}
        XCTAssertFalse(store.isRefreshing);XCTAssertTrue(store.listing.issues.isEmpty)
        XCTAssertNil(store.disclosure)
    }
    func testNewSelectionIgnoresOlderSuccessfulDisclosureAndForeignSelection()async throws {
        let (first,value)=try fixture(),second=WorkflowEvaluationHistoryReference(projectID:first.projectID,evaluationID:UUID())
        let reached=expectation(description:"First selection suspended"),secondReached=expectation(description:"Second selection suspended")
        let loader=EvaluationHistoryBarrier(reached)
        let store=WorkflowEvaluationHistoryStore(root:URL(fileURLWithPath:"/unused"),projectID:first.projectID,isCurrent:{true},loader:loader)
        store.select(first)
        await fulfillment(of:[reached],timeout:3)
        await loader.next(secondReached)
        store.select(second);await fulfillment(of:[secondReached],timeout:3)
        await loader.release(first,value:value)
        for _ in 0..<20 {await Task.yield()}
        XCTAssertEqual(store.selectedReference,second);XCTAssertNil(store.disclosure)
        // Even a loader returning a valid result for the wrong requested ID is rejected.
        await loader.release(second,value:value)
        for _ in 0..<20 {await Task.yield()}
        XCTAssertNil(store.disclosure)
        let foreign=WorkflowEvaluationHistoryReference(projectID:UUID().uuidString,evaluationID:UUID())
        store.select(foreign)
        XCTAssertNotEqual(store.selectedReference,foreign)
    }
    func testScopeChangesPreventLateMetricsAndCancellationClearsSelection()async throws {
        let(reference,value)=try fixture(),reached=expectation(description:"Disclosure suspended"),loader=EvaluationHistoryBarrier(reached)
        var active=true
        let store=WorkflowEvaluationHistoryStore(root:URL(fileURLWithPath:"/unused"),projectID:reference.projectID,isCurrent:{active},loader:loader)
        store.select(reference);await fulfillment(of:[reached],timeout:3)
        active=false;await loader.release(reference,value:value)
        for _ in 0..<20 {await Task.yield()}
        XCTAssertNil(store.disclosure)
        store.cancelLoading();XCTAssertFalse(store.isLoading)
    }
}
