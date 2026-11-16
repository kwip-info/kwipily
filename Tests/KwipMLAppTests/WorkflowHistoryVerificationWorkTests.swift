import XCTest
@testable import KwipMLApp

final class WorkflowHistoryVerificationWorkTests:XCTestCase {
    func testCancelledVerificationCannotPublishCompletedValue()async throws {
        let started=DispatchSemaphore(value:0),release=DispatchSemaphore(value:0),discarded=DispatchSemaphore(value:0)
        let task=Task.detached {try await WorkflowHistoryVerificationWork.run(discard: {_ in discarded.signal()}) {
            started.signal();release.wait();return "verified"
        }}
        XCTAssertEqual(started.wait(timeout:.now()+5),.success)
        task.cancel();release.signal()
        do {_=try await task.value;XCTFail("Cancelled verification published authority")}
        catch is CancellationError {} catch {XCTFail("Unexpected error: \(error)")}
        XCTAssertEqual(discarded.wait(timeout:.now()),.success)
        XCTAssertEqual(discarded.wait(timeout:.now()),.timedOut,"Cancellation must retire staged ownership exactly once")
    }
    func testVerificationFailureIsPropagatedOffMainThread()async throws {
        enum InvalidArchive:Error {case refused}
        do {let _:String=try await WorkflowHistoryVerificationWork.run {
            XCTAssertFalse(Thread.isMainThread)
            throw InvalidArchive.refused
        };XCTFail("Archive failure was suppressed")}
        catch InvalidArchive.refused {} catch {XCTFail("Unexpected error: \(error)")}
    }
    func testCancellableWorkChecksBeforeMutationOnItsBoundedStack()async throws {
        let started=DispatchSemaphore(value:0),release=DispatchSemaphore(value:0),mutation=DispatchSemaphore(value:0)
        let task=Task.detached {try await WorkflowHistoryVerificationWork.runCancellable {check in
            XCTAssertFalse(Thread.isMainThread);XCTAssertGreaterThanOrEqual(Thread.current.stackSize,8*1024*1024)
            started.signal();release.wait();try check();mutation.signal();return "changed"
        }}
        XCTAssertEqual(started.wait(timeout:.now()+5),.success)
        task.cancel();release.signal()
        do {_=try await task.value;XCTFail("Cancelled recovery mutated its ledger")} catch is CancellationError {} catch {XCTFail("Unexpected error: \(error)")}
        XCTAssertEqual(mutation.wait(timeout:.now()),.timedOut)
    }

}
