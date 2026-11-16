import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationExportStagingTests: XCTestCase {
    private let pin = String(repeating: "a", count: 64)
    private func event(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }
    func testBudgetTimeoutOverridesProvisionalCompletion() throws {
        let id = UUID().uuidString
        let budget = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 5), projectMaximumRunSeconds: nil)
        let transcript = WorkflowEvaluationExportTranscript(exportID: id, requestIdentity: pin, requestSHA256: pin, ownershipSHA256: pin, includesTestRows: false, executionBudget: budget)
        try transcript.consume(event(["event":"exportReady", "exportID":id, "requestSHA256":pin, "ownershipSHA256":pin]))
        try transcript.consume(event(["event":"exportStarted", "exportID":id, "requestIdentity":pin]))
        try transcript.consume(event(["event":"exportCompleted", "exportID":id, "requestIdentity":pin, "manifestSHA256":pin, "fileCount":12, "totalBytes":500, "includesTestRows":false]))
        XCTAssertThrowsError(try transcript.finish(exitCode: 124, stderr: "")) { error in
            guard case ExecutionBudgetError.wallTimeLimitExceeded(let seconds) = error else { return XCTFail("Missing structured timeout") }
            XCTAssertEqual(seconds, 5)
        }
    }
    func testTranscriptAdmitsOnlyReviewedDependencyCountsBeforeExactPackageVerification()throws {
        for includeRows in [false,true] {
            // Historical worker packages have 11 files plus manifest; current packages add 3 reviewed helpers.
            let baseline=12+(includeRows ? 2:0)
            for dependencyCount in 0...3 {
                let id=UUID().uuidString
                let transcript=WorkflowEvaluationExportTranscript(exportID:id,requestIdentity:pin,requestSHA256:pin,ownershipSHA256:pin,includesTestRows:includeRows)
                try transcript.consume(event(["event":"exportReady","exportID":id,"requestSHA256":pin,"ownershipSHA256":pin]))
                try transcript.consume(event(["event":"exportStarted","exportID":id,"requestIdentity":pin]))
                try transcript.consume(event(["event":"exportCompleted","exportID":id,"requestIdentity":pin,"manifestSHA256":pin,"fileCount":baseline+dependencyCount,"totalBytes":500,"includesTestRows":includeRows]))
                XCTAssertEqual(try transcript.finish(exitCode:0,stderr:"").fileCount,baseline+dependencyCount)
            }
            for count in [baseline-1,baseline+4] {
                let id=UUID().uuidString
                let transcript=WorkflowEvaluationExportTranscript(exportID:id,requestIdentity:pin,requestSHA256:pin,ownershipSHA256:pin,includesTestRows:includeRows)
                try transcript.consume(event(["event":"exportReady","exportID":id,"requestSHA256":pin,"ownershipSHA256":pin]))
                try transcript.consume(event(["event":"exportStarted","exportID":id,"requestIdentity":pin]))
                XCTAssertThrowsError(try transcript.consume(event(["event":"exportCompleted","exportID":id,"requestIdentity":pin,"manifestSHA256":pin,"fileCount":count,"totalBytes":500,"includesTestRows":includeRows])))
            }
        }
    }
    func testResourceFailureOverridesInterruptedWorkerErrorAndStillHonorsUserCancellation() async throws {
        for cancel in [false, true] {
            let relay = WorkflowEvaluationExportProcess(exportID: UUID().uuidString, requestIdentity: pin, requestSHA256: pin, ownershipSHA256: pin, includesTestRows: false)
            do {
                _ = try await relay.run(executable: URL(fileURLWithPath: "/unused"), arguments: [], directory: URL(fileURLWithPath: "/unused"), launch: { _, _, _, line, exit in
                    line("{\"event\":\"error\",\"message\":\"Run cancelled.\"}")
                    if cancel { relay.cancel() }
                    exit(125, "The worker exceeded its sampled resident memory budget. Increase the memory budget.")
                    return .init(cancel: {})
                })
                XCTFail("Resource refusal cannot publish an export")
            } catch {
                if cancel { XCTAssertTrue(error is CancellationError) }
                else { XCTAssertTrue(error.localizedDescription.contains("sampled resident memory")) }
            }
            XCTAssertTrue(relay.hasExited)
        }
    }
    func testStrictTranscriptRequiresActualExitAndMatchingCompletion() throws {
        let id = UUID().uuidString
        let transcript = WorkflowEvaluationExportTranscript(exportID: id, requestIdentity: pin, requestSHA256: pin, ownershipSHA256: pin, includesTestRows: false)
        try transcript.consume(event(["event":"exportReady", "exportID":id, "requestSHA256":pin, "ownershipSHA256":pin]))
        try transcript.consume(event(["event":"exportStarted", "exportID":id, "requestIdentity":pin]))
        try transcript.consume(event(["event":"exportCompleted", "exportID":id, "requestIdentity":pin, "manifestSHA256":pin, "fileCount":12, "totalBytes":500, "includesTestRows":false]))
        XCTAssertThrowsError(try transcript.finish(exitCode: 1, stderr: "worker failed"))
        let result = try transcript.finish(exitCode: 0, stderr: "")
        XCTAssertEqual(result.fileCount, 12)
        XCTAssertEqual(result.manifestSHA256, pin)
        XCTAssertThrowsError(try transcript.consume(event(["event":"exportStarted", "exportID":id, "requestIdentity":pin])))
    }
    func testTranscriptRejectsPrematureForeignRepeatedAndWrongPrivacyEvents() throws {
        let id = UUID().uuidString
        func fresh() -> WorkflowEvaluationExportTranscript { .init(exportID:id,requestIdentity:pin,requestSHA256:pin,ownershipSHA256:pin,includesTestRows:false) }
        let premature = fresh()
        XCTAssertThrowsError(try premature.consume(event(["event":"exportStarted","exportID":id,"requestIdentity":pin])))
        let foreign = fresh()
        XCTAssertThrowsError(try foreign.consume(event(["event":"exportReady","exportID":UUID().uuidString,"requestSHA256":pin,"ownershipSHA256":pin])))
        let repeated = fresh(), ready = try event(["event":"exportReady","exportID":id,"requestSHA256":pin,"ownershipSHA256":pin])
        try repeated.consume(ready)
        XCTAssertThrowsError(try repeated.consume(ready))
        let privacy = fresh()
        try privacy.consume(ready)
        try privacy.consume(event(["event":"exportStarted","exportID":id,"requestIdentity":pin]))
        XCTAssertThrowsError(try privacy.consume(event(["event":"exportCompleted","exportID":id,"requestIdentity":pin,"manifestSHA256":pin,"fileCount":14,"totalBytes":500,"includesTestRows":true])))
        XCTAssertThrowsError(try fresh().finish(exitCode:0,stderr:""))
    }
    func testCancellationRetainsAwaitUntilActualExitEvenBeforeHandleAssignment() async throws {
        let id = UUID().uuidString, launched = expectation(description:"launched"), cancelled = expectation(description:"cancel delivered")
        let relay = WorkflowEvaluationExportProcess(exportID:id,requestIdentity:pin,requestSHA256:pin,ownershipSHA256:pin,includesTestRows:false)
        let callbacks = ExportStagingCallbacks()
        let task = Task {
            try await relay.run(executable:URL(fileURLWithPath:"/unused"),arguments:[],directory:URL(fileURLWithPath:"/unused"),launch:{ _,_,_,line,exit in
                callbacks.set(exit)
                relay.cancel()
                launched.fulfill()
                return .init(cancel:{cancelled.fulfill()})
            })
        }
        await fulfillment(of:[launched,cancelled],timeout:5)
        XCTAssertFalse(relay.hasExited)
        callbacks.exit(0, "")
        do { _ = try await task.value; XCTFail("Cancelled process must not yield a package") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertTrue(relay.hasExited)
    }
    func testInvalidSynchronousEventCancelsOnlyAfterHandleExistsThenWaitsExit() async throws {
        let id = UUID().uuidString, launched = expectation(description:"launched"), cancelled = expectation(description:"protocol rejection")
        let relay = WorkflowEvaluationExportProcess(exportID:id,requestIdentity:pin,requestSHA256:pin,ownershipSHA256:pin,includesTestRows:false)
        let callbacks = ExportStagingCallbacks()
        let task = Task {
            try await relay.run(executable:URL(fileURLWithPath:"/unused"),arguments:[],directory:URL(fileURLWithPath:"/unused"),launch:{ _,_,_,line,exit in
                callbacks.set(exit);line("{\"event\":\"exportStarted\"}")
                launched.fulfill();return .init(cancel:{cancelled.fulfill()})
            })
        }
        await fulfillment(of:[launched,cancelled],timeout:5)
        XCTAssertFalse(relay.hasExited)
        callbacks.exit(0, "")
        do { _ = try await task.value; XCTFail("Malformed transcript accepted") }
        catch { XCTAssertFalse(error is CancellationError) }
    }
}

private final class ExportStagingCallbacks: @unchecked Sendable {
    private let lock = NSLock()
    private var callback: ((Int32,String)->Void)?
    func set(_ value:@escaping (Int32,String)->Void) { lock.lock(); callback=value;lock.unlock() }
    func exit(_ code:Int32,_ error:String) { lock.lock();let value=callback;lock.unlock();value?(code,error) }
}
