import XCTest
import KwipMLCore
@testable import KwipMLApp
final class ExperimentTests: XCTestCase {
    func testHistoryPersistsActualStatusAndDoesNotMixProjects() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            store.beginProject(.init(name: "Untitled flow"))
            let id = try XCTUnwrap(store.activeProjectID)
            store.metrics = [EpochMetric(epoch: 1, trainLoss: 0.5, validationLoss: 0.6, accuracy: 0.7)]
            store.recordExperiment(status: "cancelled", directory: root.appendingPathComponent("test-run"), error: nil)
            XCTAssertEqual(store.experiments.count, 1)
            XCTAssertEqual(store.experiments.first?.status, "cancelled")
            XCTAssertEqual(store.experiments.first?.epoch, 1)
            let restored = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            restored.activeProjectID = id; restored.refreshExperiments()
            XCTAssertEqual(restored.experiments.first?.id, store.experiments.first?.id)
            restored.beginProject(.init(name: "Untitled flow")); XCTAssertTrue(restored.experiments.isEmpty)
        }
    }
}
