import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowEvaluationSessionTests:XCTestCase {
    private let terminal=WorkflowNodeAddress(domain:.control,localID:"evaluate")
    func testBudgetIsPinnedAfterCaptureAndTimeoutCannotPublishCompletion() async throws {
        for (cancel, memory) in [(false, false), (true, false), (false, true), (true, true)] {
            let helper = WorkflowEvaluationRequestTests()
            let (root, owner, bindings, runtime) = try helper.fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 30), projectMaximumRunSeconds: nil)
            let request = try await WorkflowEvaluationRequest.capture(from: owner, bindings: bindings, evaluate: terminal,
                runtime: runtime, root: root, stage: helper.staged, executionBudget: policy)
            let launched = expectation(description: "Budgeted evaluation launch"), finished = expectation(description: "Budgeted evaluation exit")
            var onLine: ((String) -> Void)?, onExit: ((Int32, String) -> Void)?, arguments: [String] = []
            var released = 0, results: [WorkflowEvaluationSessionResult] = []
            let session = WorkflowEvaluationSession(request: request, owner: owner, bindings: bindings, workspace: root,
                reservation: .init(acquire: { UUID() }, release: { _ in released += 1 }),
                launch: { _, args, _, line, exit in
                    arguments = args; onLine = line; onExit = exit; launched.fulfill(); return .init(cancel: {})
                }, onTerminal: { results.append($0); finished.fulfill() })
            XCTAssertTrue(session.start())
            await fulfillment(of: [launched], timeout: 10)
            let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-launch-budget-sha256"))
            let control = try XCTUnwrap(arguments.firstIndex(of: "--kwip-evaluate-supervise"))
            let folder = URL(fileURLWithPath: arguments[control + 1])
            let launchBytes = try Data(contentsOf: folder.appendingPathComponent("launch-budget.json"))
            XCTAssertEqual(WorkflowArchiveCodec.hash(launchBytes), arguments[index + 1])
            let launchValue = try XCTUnwrap(JSONSerialization.jsonObject(with: launchBytes) as? [String: Any])
            XCTAssertEqual(launchValue["requestSHA256"] as? String, arguments[control + 2])
            XCTAssertLessThan(try XCTUnwrap(launchValue["remainingSeconds"] as? Double), 30)
            let input = try XCTUnwrap(session.workerInput)
            XCTAssertEqual(input.request.schemaVersion, 2)
            XCTAssertEqual(input.request.executionBudget, policy)
            func emit(_ object: [String: Any]) throws { onLine?(String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)) }
            try emit(["event": "evaluationReady", "evaluationID": session.evaluationID.uuidString,
                      "requestSHA256": try input.request.requestSHA256(), "ownershipSHA256": try XCTUnwrap(session.ownershipSHA256)])
            try emit(["event": "evaluationStarted", "evaluationID": session.evaluationID.uuidString, "requestIdentity": request.requestIdentity, "totalCount": 3])
            try emit(["event": "evaluationCompleted", "evaluationID": session.evaluationID.uuidString, "requestIdentity": request.requestIdentity,
                      "manifestSHA256": String(repeating: "a", count: 64), "evaluatedCount": 3])
            if cancel { session.cancel() }
            XCTAssertTrue(results.isEmpty); XCTAssertEqual(released, 0)
            onExit?(memory ? 125 : 124, memory ? "The worker exceeded its sampled resident memory budget. Increase the memory budget." : "")
            await fulfillment(of: [finished], timeout: 10)
            XCTAssertEqual(results.first?.outcome.status, cancel ? .cancelled : .failed)
            XCTAssertEqual(results.first?.outcome.failureCode, cancel ? nil : memory ? .resourceBudgetFailed : .wallTimeLimitExceeded)
            if memory && !cancel { XCTAssertTrue(results.first?.outcome.error?.contains("sampled resident memory") == true) }
            XCTAssertNil(results.first?.outcome.manifestSHA256)
            XCTAssertEqual(released, 1)
        }
    }
    func testCompletionBeforeExitIsWithheldAndCancellationRetainsReservation()async throws {
        let helper=WorkflowEvaluationRequestTests()
        let(root,owner,bindings,runtime)=try helper.fixture()
        defer {try? FileManager.default.removeItem(at:root)}
        let request=try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root,stage:helper.staged)
        defer {try? FileManager.default.removeItem(at:request.snapshot.directory)}
        let launched=expectation(description:"Worker launch"),finished=expectation(description:"Actual exit terminal")
        var onLine:((String)->Void)?,onExit:((Int32,String)->Void)?,released=0,cancelled=0,results:[WorkflowEvaluationSessionResult]=[]
        let session=WorkflowEvaluationSession(request:request,owner:owner,bindings:bindings,workspace:root,
            reservation:.init(acquire:{UUID()},release:{_ in released+=1}),launch:{_,_,_,line,exit in
                onLine=line;onExit=exit;launched.fulfill();return .init(cancel:{cancelled+=1})
            },onTerminal:{results.append($0);finished.fulfill()})
        XCTAssertTrue(session.start())
        await fulfillment(of:[launched],timeout:10)
        XCTAssertTrue(owner.isBusy)
        let input=try XCTUnwrap(session.workerInput),ownership=try XCTUnwrap(session.ownershipSHA256)
        func emit(_ object:[String:Any])throws {onLine?(String(decoding:try JSONSerialization.data(withJSONObject:object),as:UTF8.self))}
        try emit(["event":"evaluationReady","evaluationID":session.evaluationID.uuidString,"requestSHA256":try input.request.requestSHA256(),"ownershipSHA256":ownership])
        try emit(["event":"evaluationStarted","evaluationID":session.evaluationID.uuidString,"requestIdentity":request.requestIdentity,"totalCount":3])
        try emit(["event":"evaluationProgress","evaluationID":session.evaluationID.uuidString,"requestIdentity":request.requestIdentity,"completedCount":3,"totalCount":3])
        try emit(["event":"evaluationCompleted","evaluationID":session.evaluationID.uuidString,"requestIdentity":request.requestIdentity,"manifestSHA256":String(repeating:"a",count:64),"evaluatedCount":3])
        XCTAssertTrue(results.isEmpty);XCTAssertEqual(released,0)
        session.cancel();XCTAssertEqual(cancelled,1);XCTAssertEqual(released,0);XCTAssertTrue(owner.isBusy)
        onExit?(0,"")
        await fulfillment(of:[finished],timeout:10)
        XCTAssertEqual(results.count,1);XCTAssertEqual(results.first?.outcome.status,.cancelled)
        XCTAssertNil(results.first?.outcome.manifestSHA256);XCTAssertEqual(released,1);XCTAssertFalse(owner.isBusy)
        onExit?(0,"");XCTAssertEqual(results.count,1)
        let saved=try Data(contentsOf:try XCTUnwrap(results.first).directory.appendingPathComponent("outcome.json"))
        XCTAssertEqual(try JSONDecoder().decode(WorkflowEvaluationOutcome.self,from:saved).status,.cancelled)
    }
    func testPrematureSynchronousOutputCancelsHandleAfterAssignmentAndWaitsExit()async throws {
        let helper=WorkflowEvaluationRequestTests()
        let(root,owner,bindings,runtime)=try helper.fixture()
        defer {try? FileManager.default.removeItem(at:root)}
        let request=try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root,stage:helper.staged)
        defer {try? FileManager.default.removeItem(at:request.snapshot.directory)}
        let launched=expectation(description:"Synchronous invalid event"),finished=expectation(description:"Exit after cancellation")
        var onExit:((Int32,String)->Void)?,cancelled=0,released=0,results:[WorkflowEvaluationSessionResult]=[]
        let session=WorkflowEvaluationSession(request:request,owner:owner,bindings:bindings,workspace:root,
            reservation:.init(acquire:{UUID()},release:{_ in released+=1}),launch:{_,_,_,line,exit in
                onExit=exit;line("{\"event\":\"evaluationProgress\"}");launched.fulfill();return .init(cancel:{cancelled+=1})
            },onTerminal:{results.append($0);finished.fulfill()})
        XCTAssertTrue(session.start());await fulfillment(of:[launched],timeout:10)
        XCTAssertEqual(cancelled,1);XCTAssertEqual(released,0);XCTAssertTrue(owner.isBusy)
        onExit?(1,"");await fulfillment(of:[finished],timeout:10)
        XCTAssertEqual(results.first?.outcome.status,.failed);XCTAssertEqual(released,1);XCTAssertFalse(owner.isBusy)
    }
    func testPublisherRejectsForeignOwnerBeforeCreatingInputs()async throws {
        let helper=WorkflowEvaluationRequestTests()
        let(root,owner,bindings,runtime)=try helper.fixture()
        defer {try? FileManager.default.removeItem(at:root)}
        let request=try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root,stage:helper.staged)
        let id=UUID()
        let foreign=WorkflowEvaluationWorkerRequest(evaluationID:id.uuidString,projectID:request.document.projectID,requestIdentity:String(repeating:"f",count:64),snapshotReceiptSHA256:request.snapshot.receiptSHA256,settings:request.plan.settings)
        let authority=try WorkflowEvaluationOwnership.create(workspace:root,request:foreign)
        defer {authority.close()}
        XCTAssertThrowsError(try WorkflowEvaluationWorkerInput.publish(request:request,evaluationID:id,directory:authority.directory,owner:authority))
        XCTAssertFalse(FileManager.default.fileExists(atPath:authority.directory.appendingPathComponent("input").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:authority.directory.appendingPathComponent("request.json").path))
    }
    func testStaleBindingRejectsBeforeReservationOrAllocation()async throws {
        let helper=WorkflowEvaluationRequestTests()
        let(root,owner,bindings,runtime)=try helper.fixture()
        defer {try? FileManager.default.removeItem(at:root)}
        let request=try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root,stage:helper.staged)
        defer {try? FileManager.default.removeItem(at:request.snapshot.directory)}
        XCTAssertTrue(bindings.setReference(nil,for:request.plan.frozenModel,lease:owner.lease(),expectedRevision:bindings.revision))
        let session=WorkflowEvaluationSession(request:request,owner:owner,bindings:bindings,workspace:root,
            reservation:.init(acquire:{XCTFail("Stale request reserved workspace");return nil},release:{_ in XCTFail("No reservation exists")}),launch:{_,_,_,_,_ in XCTFail("No worker should launch");return .init(cancel:{})},onTerminal:{_ in XCTFail("No run accepted")})
        XCTAssertFalse(session.start());XCTAssertFalse(owner.isBusy)
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("workflow-evaluations").path))
    }
}
