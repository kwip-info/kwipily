import XCTest
import Combine
@testable import KwipMLApp

@MainActor final class WorkflowExperimentRecoveryStoreTests:XCTestCase {
    typealias Store = WorkflowExperimentRecoveryStore
    private func scope(group:String = UUID().uuidString,session:UUID = UUID())->Store.Scope {
        .init(workspace:URL(fileURLWithPath:"/private/tmp/recovery-store"),projectID:UUID().uuidString,sessionID:session,groupID:group)
    }
    private func awaitState(_ store:Store,_ state:Store.State)async {
        if store.state == state {return}
        let done=expectation(description:"Recovery reaches \(state)")
        let observation=store.$state.filter {$0 == state}.prefix(1).sink {_ in done.fulfill()}
        await fulfillment(of:[done],timeout:3);observation.cancel()
    }
    func testLateOldScopeAndSameScopeRefreshCannotPublish()async {
        let gate=RecoveryStoreGate()
        let store=Store(probe:{try await gate.probe($0)},recover:{_ in throw CancellationError()})
        let first=scope(),second=scope()
        store.select(scope:first);await gate.waitForProbes(1)
        store.select(scope:second);await gate.waitForProbes(2)
        await gate.finishProbe(1,.recoverable);await awaitState(store,.recoverable)
        await gate.finishProbe(0,.unavailable("Old group"));await Task.yield()
        XCTAssertEqual(store.scope,second);XCTAssertEqual(store.state,.recoverable)
        store.refresh();await gate.waitForProbes(3)
        store.refresh();await gate.waitForProbes(4)
        await gate.finishProbe(3,.terminal);await awaitState(store,.terminal)
        await gate.finishProbe(2,.inUse);await Task.yield()
        XCTAssertEqual(store.state,.terminal)
    }
    func testDoubleRecoveryAndCancellationWaitForAcknowledgementThenProbeActualState()async {
        let gate=RecoveryStoreGate();var refreshed:[Store.Scope]=[]
        let selected=scope()
        let store=Store(probe:{try await gate.probe($0)},recover:{try await gate.recover($0)},onRefresh:{refreshed.append($0)})
        store.select(scope:selected);await gate.waitForProbes(1)
        await gate.finishProbe(0,.recoverable);await awaitState(store,.recoverable)
        store.recover();store.recover();await gate.waitForRecoveries(1)
        XCTAssertEqual(store.state,.recovering)
        store.cancel();XCTAssertEqual(store.state,.cancelling);XCTAssertTrue(store.isBusy)
        store.recover();store.refresh()
        let recoveryCount=await gate.recoveryCount;XCTAssertEqual(recoveryCount,1)
        XCTAssertTrue(refreshed.isEmpty,"Cancellation is not worker exit or rollback.")
        await gate.failRecovery(0,CancellationError());await gate.waitForProbes(2)
        await gate.finishProbe(1,.terminal);await awaitState(store,.terminal)
        XCTAssertEqual(refreshed,[selected]);XCTAssertFalse(store.isBusy)
    }
    func testOldRecoveryCompletionCannotRefreshNewSelection()async {
        let gate=RecoveryStoreGate();var refreshed:[Store.Scope]=[]
        let store=Store(probe:{try await gate.probe($0)},recover:{try await gate.recover($0)},onRefresh:{refreshed.append($0)})
        let first=scope(),second=scope()
        store.select(scope:first);await gate.waitForProbes(1)
        await gate.finishProbe(0,.recoverable);await awaitState(store,.recoverable)
        store.recover();await gate.waitForRecoveries(1)
        store.select(scope:second);await gate.waitForProbes(2)
        await gate.finishProbe(1,.inUse);await awaitState(store,.inUse)
        await gate.failRecovery(0,CancellationError());await Task.yield()
        XCTAssertEqual(store.scope,second);XCTAssertEqual(store.state,.inUse);XCTAssertTrue(refreshed.isEmpty)
    }
    func testInactiveScopeAndInvalidationRejectLatePublicationAndActions()async {
        let gate=RecoveryStoreGate();var active=true
        let store=Store(probe:{try await gate.probe($0)},recover:{try await gate.recover($0)},isCurrent:{_ in active})
        store.select(scope:scope());await gate.waitForProbes(1)
        active=false
        await gate.finishProbe(0,.recoverable);await Task.yield()
        XCTAssertNotEqual(store.state,.recoverable)
        store.invalidate();store.refresh();store.recover()
        XCTAssertFalse(store.isBusy)
        let count=await gate.recoveryCount;XCTAssertEqual(count,0)
    }
    func testSameScopeReappearanceReactivatesAndDiscardsOldProbe()async {
        let gate=RecoveryStoreGate()
        let store=Store(probe:{try await gate.probe($0)},recover:{try await gate.recover($0)})
        let selected=scope()
        store.select(scope:selected);await gate.waitForProbes(1)
        store.invalidate()
        XCTAssertNil(store.scope);XCTAssertEqual(store.state,.idle)
        store.select(scope:selected)
        // Assert before awaiting a second continuation so the intentionally red
        // implementation fails without leaving an unresolved probe behind.
        guard store.scope == selected,store.state == .checking else {
            await gate.finishProbe(0,.inUse)
            return XCTFail("A current scope must reactivate after its view reappears")
        }
        await gate.waitForProbes(2)
        await gate.finishProbe(1,.recoverable);await awaitState(store,.recoverable)
        await gate.finishProbe(0,.unavailable("Old disappeared view"));await Task.yield()
        XCTAssertEqual(store.scope,selected);XCTAssertEqual(store.state,.recoverable)
    }
    func testReappearanceCannotReactivateRetiredSession()async {
        let gate=RecoveryStoreGate();var active=true
        let store=Store(probe:{try await gate.probe($0)},recover:{try await gate.recover($0)},isCurrent:{_ in active})
        let selected=scope()
        store.select(scope:selected);await gate.waitForProbes(1)
        store.invalidate();active=false;store.select(scope:selected)
        await gate.finishProbe(0,.recoverable);await Task.yield()
        XCTAssertNil(store.scope);XCTAssertEqual(store.state,.idle);XCTAssertFalse(store.isBusy)
    }
    func testUnavailableAndInUseNeverOfferRecovery()async {
        let gate=RecoveryStoreGate()
        let store=Store(probe:{try await gate.probe($0)},recover:{try await gate.recover($0)})
        store.select(scope:scope());await gate.waitForProbes(1)
        await gate.finishProbe(0,.unavailable("Legacy ownership evidence is absent"))
        await awaitState(store,.unavailable("Legacy ownership evidence is absent"));store.recover()
        store.refresh();await gate.waitForProbes(2)
        await gate.finishProbe(1,.inUse);await awaitState(store,.inUse);store.recover()
        let count=await gate.recoveryCount;XCTAssertEqual(count,0)
    }
}

private actor RecoveryStoreGate {
    typealias Scope = WorkflowExperimentRecoveryStore.Scope
    typealias Availability = WorkflowExperimentRecoveryCoordinator.Availability
    private var probes:[CheckedContinuation<Availability,Error>]=[]
    private var recoveries:[CheckedContinuation<WorkflowExperimentGroupRecord,Error>]=[]
    var recoveryCount:Int {recoveries.count}
    func probe(_ scope:Scope)async throws->Availability {
        try await withCheckedThrowingContinuation {probes.append($0)}
    }
    func recover(_ scope:Scope)async throws->WorkflowExperimentGroupRecord {
        try await withCheckedThrowingContinuation {recoveries.append($0)}
    }
    func finishProbe(_ index:Int,_ value:Availability) {probes[index].resume(returning:value)}
    func failRecovery(_ index:Int,_ error:Error) {recoveries[index].resume(throwing:error)}
    func waitForProbes(_ count:Int)async {
        for _ in 0..<10000 {if probes.count >= count {return};await Task.yield()}
        XCTFail("Probe was not started")
    }
    func waitForRecoveries(_ count:Int)async {
        for _ in 0..<10000 {if recoveries.count >= count {return};await Task.yield()}
        XCTFail("Recovery was not started")
    }
}
