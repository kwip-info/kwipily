import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExecutionQueueTests: XCTestCase {
    func testTerminalPersistenceFailureStopsLiveFeedbackAndDoesNotClaimSuccess() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = WorkflowExecutionQueue(directory: root)
        defer { queue.close() }
        let id = try queue.enqueue(capture: capture("Feedback fault"), start: {}, cancel: {})
        await Task.yield(); await Task.yield()
        XCTAssertTrue(RunFeedbackController.shared.activities.contains(where: { $0.id == id }))
        let receipt = root.appendingPathComponent("queue.json")
        try FileManager.default.removeItem(at: receipt)
        try FileManager.default.createDirectory(at: receipt, withIntermediateDirectories: false)
        queue.finish(id, state: .completed, message: "Worker finished")
        XCTAssertNotNil(queue.error)
        XCTAssertFalse(RunFeedbackController.shared.activities.contains(where: { $0.id == id }))
        XCTAssertEqual(RunFeedbackController.shared.latestOutcome, .failed)
    }

    func testOneSlotIndependentCancellationAndTerminalAdvance() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = WorkflowExecutionQueue(directory: root)
        var launched: [String] = [], cancelled = 0
        let first = try queue.enqueue(capture: capture("First"), start: { launched.append("First") }, cancel: { cancelled += 1 })
        let second = try queue.enqueue(capture: capture("Second"), start: { launched.append("Second") }, cancel: { cancelled += 1 })
        await Task.yield(); await Task.yield()
        XCTAssertEqual(launched, ["First"])
        XCTAssertEqual(queue.entries.first(where: {$0.id == second})?.state, .queued)
        queue.cancel(second)
        XCTAssertEqual(cancelled, 1)
        XCTAssertEqual(queue.entries.first(where: {$0.id == second})?.state, .cancelled)
        queue.finish(first, state: .completed, message: "Done")
        await Task.yield(); await Task.yield()
        XCTAssertEqual(launched, ["First"])
        let third = try queue.enqueue(capture: capture("Third"), start: { launched.append("Third") }, cancel: {})
        await Task.yield(); await Task.yield()
        XCTAssertEqual(launched, ["First", "Third"])
        queue.finish(third, state: .completed, message: "Done")
        queue.close()
    }
    func testInterruptedRecordsRestorePausedWithoutLaunchingOrRecapturing() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = WorkflowExecutionQueue(directory: root)
        let captured = try capture("Pinned")
        _ = try queue.enqueue(capture: captured, start: {}, cancel: {})
        _ = try queue.enqueue(capture: capture("Waiting"), start: {}, cancel: {})
        await Task.yield(); await Task.yield()
        queue.close()
        let reopened = WorkflowExecutionQueue(directory: root)
        XCTAssertEqual(reopened.entries.count, 2)
        XCTAssertTrue(reopened.entries.allSatisfy {$0.state == .paused})
        XCTAssertEqual(reopened.entries.first?.capture, captured)
        XCTAssertTrue(reopened.entries.allSatisfy {$0.message.contains("capture")})
        reopened.close()
    }
    func testStaleDispatchFailureCannotBlockNextIndependentRequest() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = WorkflowExecutionQueue(directory: root)
        var nextStarted = false
        let stale = try queue.enqueue(capture: capture("Stale"), start: { throw DataFlowContractError("Source binding changed after capture; submit again.") }, cancel: {})
        _ = try queue.enqueue(capture: capture("Independent"), start: { nextStarted = true }, cancel: {})
        for _ in 0..<5 { await Task.yield() }
        XCTAssertEqual(queue.entries.first(where: {$0.id == stale})?.state, .failed)
        XCTAssertTrue(nextStarted)
        queue.close()
    }
    private func capture(_ title: String) throws -> WorkflowQueueCapture {
        let source = try WorkflowExamples.preparedParallelClassifier()
        return try .init(projectID: source.projectID, title: title, kind: .training,
            requestIdentities: [String(repeating: "a", count: 64)], document: source, sourceHashes: [String(repeating: "b", count: 64)])
    }
}
