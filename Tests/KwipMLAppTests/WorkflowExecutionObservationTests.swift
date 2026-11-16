import XCTest
import Combine
@testable import KwipMLApp

@MainActor final class WorkflowExecutionObservationTests: XCTestCase {
    func testEpochBurstCoalescesBroadInvalidationAndTerminalFlushesWithoutLosingState() async throws {
        let changes = PassthroughSubject<Void, Never>()
        let phase = CurrentValueSubject<WorkflowExecutionPhase, Never>(.training)
        var exactMetrics = 0, renderedMetrics: [Int] = []
        let observation = WorkflowExecutionObservation.connect(changes: changes.eraseToAnyPublisher(), phases: phase.eraseToAnyPublisher(),
            isCurrent: { true }, publish: { renderedMetrics.append(exactMetrics) })
        defer { observation.cancel() }
        for _ in 0..<10_000 { exactMetrics += 1; changes.send() }
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertLessThanOrEqual(renderedMetrics.count, 2)
        XCTAssertEqual(renderedMetrics.last, 10_000)
        exactMetrics += 1; changes.send()
        phase.send(.completed)
        XCTAssertEqual(renderedMetrics.last, 10_001, "Terminal state must publish synchronously without waiting for throttle")
        let terminalCount = renderedMetrics.count
        changes.send()
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertEqual(renderedMetrics.count, terminalCount, "Pending progress cannot publish after terminal state")
        XCTAssertEqual(exactMetrics, 10_001, "Only invalidations are coalesced; evidence is untouched")
    }
    func testCancelledSubscriptionAndReplacedOwnerRejectPendingDelivery() async throws {
        for cancelled in [false, true] {
            let changes = PassthroughSubject<Void, Never>()
            let phase = CurrentValueSubject<WorkflowExecutionPhase, Never>(.training)
            var current = true, updates = 0
            let observation = WorkflowExecutionObservation.connect(changes: changes.eraseToAnyPublisher(), phases: phase.eraseToAnyPublisher(),
                isCurrent: { current }, publish: { updates += 1 })
            changes.send()
            if cancelled { observation.cancel() } else { current = false }
            let count = updates
            phase.send(.cancelled)
            try await Task.sleep(for: .milliseconds(250))
            XCTAssertEqual(updates, count)
            observation.cancel()
        }
    }
}
