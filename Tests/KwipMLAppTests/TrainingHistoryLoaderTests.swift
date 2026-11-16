import XCTest
@testable import KwipMLApp

final class TrainingHistoryLoaderTests:XCTestCase {
    func testHistoryLoaderUsesBoundedStackOffMainThread()async throws {
        let loader=WorkflowHistoryLoader {_,_ in
            XCTAssertFalse(Thread.isMainThread)
            XCTAssertGreaterThanOrEqual(Thread.current.stackSize,8*1024*1024)
            return .incomplete("observed on verifier")
        }
        let value=try await loader.load(.init(projectID:UUID().uuidString,directoryID:UUID()),root:FileManager.default.temporaryDirectory)
        guard case .incomplete(let reason)=value else {return XCTFail("Verification result changed")}
        XCTAssertEqual(reason,"observed on verifier")
    }
    func testCancelledQueuedSelectionDoesNotLaunchAnotherVerifier()async throws {
        let entered=DispatchSemaphore(value:0),release=DispatchSemaphore(value:0)
        let state=Calls()
        let loader=WorkflowHistoryLoader {_,_ in
            state.record();entered.signal();release.wait();return .incomplete("finished")
        }
        let reference=WorkflowHistoryReference(projectID:UUID().uuidString,directoryID:UUID())
        let first=Task {try await loader.load(reference,root:FileManager.default.temporaryDirectory)}
        XCTAssertEqual(entered.wait(timeout:.now()+5),.success)
        let second=Task {try await loader.load(reference,root:FileManager.default.temporaryDirectory)}
        second.cancel()
        release.signal();_ = try await first.value
        do {_ = try await second.value;XCTFail("Cancelled queued selection published a result")}
        catch is CancellationError {} catch {XCTFail("Unexpected error: \(error)")}
        XCTAssertEqual(state.count,1)
    }
    private final class Calls:@unchecked Sendable {
        private let lock=NSLock();private var value=0
        func record() {lock.lock();value+=1;lock.unlock()}
        var count:Int {lock.lock();defer {lock.unlock()};return value}
    }
}
