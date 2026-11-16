import XCTest
import KwipMLCore
@testable import KwipMLApp

final class HistorySafetyTests: XCTestCase {
    @MainActor private func store() throws -> (WorkspaceStore, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("KwipML-history-test-" + UUID().uuidString)
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        store.beginProject(.init(name: "Untitled flow"))
        return (store, root)
    }
    @MainActor func testCancelledHistoryUsesLatestReceivedMetricNotChartSamples() throws {
        let (store, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        store.graph.execution = ExecutionSettings(epochs: 10000)
        store.consume("{\"event\":\"epoch\",\"epoch\":50,\"trainLoss\":0.7,\"validationLoss\":0.8,\"accuracy\":0.6}\n")
        store.consume("{\"event\":\"epoch\",\"epoch\":51,\"trainLoss\":0.5,\"validationLoss\":0.6,\"accuracy\":0.7}\n")
        XCTAssertEqual(store.metrics.last?.epoch, 50, "Chart remains sampled.")
        store.recordExperiment(status: "cancelled", directory: root.appendingPathComponent("run"), error: nil)
        XCTAssertEqual(store.experiments.first?.epoch, 51)
        XCTAssertEqual(store.experiments.first?.validationLoss, 0.6)
        XCTAssertEqual(store.experiments.first?.accuracy, 0.7)
    }
    @MainActor func testCancellationDoesNotRecordWorkerErrorEvent() throws {
        let (store, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        store.cancelRun()
        store.consume("{\"event\":\"error\",\"message\":\"Run cancelled.\"}\n")
        XCTAssertNil(store.runError)
        XCTAssertFalse(store.eventLog.contains("Worker reported an error"))
    }
    @MainActor func testUnreadableHistoryIsNeverOverwrittenByNextRun() throws {
        let (store, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        let index = try XCTUnwrap(store.experimentIndex)
        try FileManager.default.createDirectory(at: index.deletingLastPathComponent(), withIntermediateDirectories: true)
        for original in [Data("{damaged history".utf8), Data(repeating: 32, count: 8_000_001)] {
            try original.write(to: index)
            store.refreshExperiments()
            XCTAssertTrue(store.experiments.isEmpty)
            store.recordExperiment(status: "failed", directory: root.appendingPathComponent("run"), error: "test")
            XCTAssertEqual(try Data(contentsOf: index), original)
            XCTAssertTrue(store.eventLog.contains { $0.contains("History save failed") })
        }
    }
    @MainActor func testOversizedWritePreservesReadableHistory() throws {
        let (store, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        store.recordExperiment(status: "cancelled", directory: root.appendingPathComponent("first"), error: nil)
        let index = try XCTUnwrap(store.experimentIndex)
        let original = try Data(contentsOf: index)
        let originalID = store.experiments.first?.id
        store.graph.name = String(repeating: "x", count: 8_000_001)
        store.recordExperiment(status: "failed", directory: root.appendingPathComponent("second"), error: "test")
        XCTAssertEqual(try Data(contentsOf: index), original)
        XCTAssertEqual(store.experiments.first?.id, originalID)
        store.refreshExperiments()
        XCTAssertEqual(store.experiments.first?.id, originalID)
    }
    @MainActor func testRepeatedRunLabelsIncludeSplitAndDistinctIdentity() throws {
        let (store, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        store.graph.execution = ExecutionSettings(splitSeed: 17)
        store.recordExperiment(status: "cancelled", directory: root.appendingPathComponent("a"), error: nil)
        store.recordExperiment(status: "cancelled", directory: root.appendingPathComponent("b"), error: nil)
        XCTAssertEqual(store.experiments.count, 2)
        XCTAssertNotEqual(store.experiments[0].label, store.experiments[1].label)
        XCTAssertTrue(store.experiments.allSatisfy { $0.label.contains("split 17") })
    }

}
