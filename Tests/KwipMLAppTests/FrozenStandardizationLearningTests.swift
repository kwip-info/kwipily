import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FrozenStandardizationLearningTests: XCTestCase {
    func testActualAssessmentPassReopensAndWrongOffsetFailsWithoutErasingCompletion() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Enable actual runtime") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .frozenStandardization, mode: .assessment)
        XCTAssertTrue(lesson.isActive)
        XCTAssertEqual(try lesson.draft?.resolvedLessonIdentity(), .frozenStandardization)
        workspace.edit {
            $0 = GraphDocument(nodes: [GraphNode(id: "i", type: "input", config: ["features": 2]),
                GraphNode(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]), GraphNode(id: "o", type: "output")],
                edges: [GraphEdge(source: "i", target: "m"), GraphEdge(source: "m", target: "o")])
        }
        workspace.selectedNodeID = "m"
        lesson.resetParameters(nodeID: "m")
        XCTAssertTrue(lesson.setWeight(nodeID: "m", row: 0, column: 0, value: 0.5))
        XCTAssertTrue(lesson.setWeight(nodeID: "m", row: 1, column: 1, value: 0.25))
        XCTAssertTrue(lesson.setBias(nodeID: "m", index: 0, value: -1))
        XCTAssertTrue(lesson.setBias(nodeID: "m", index: 1, value: -3))
        lesson.submit(); await finish(workspace)
        XCTAssertEqual(lesson.latestGrade?.status, .passed, lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved); XCTAssertTrue(lesson.hasCurrentCompletion)
        let reopened = AffineLearningStore(workspace: workspace)
        reopened.reopenCurrentProject()
        XCTAssertEqual(try reopened.draft?.resolvedLessonIdentity(), .frozenStandardization)
        XCTAssertEqual(reopened.latestGrade?.status, .passed, reopened.statusMessage)
        workspace.learningProjects.attachAffine(reopened)
        XCTAssertTrue(reopened.setBias(nodeID: "m", index: 0, value: -2))
        reopened.submit(); await finish(workspace)
        XCTAssertEqual(reopened.latestGrade?.status, .failed, reopened.statusMessage)
        XCTAssertTrue(reopened.hasCurrentCompletion)
    }
    private func finish(_ workspace: WorkspaceStore) async {
        guard workspace.isAssessingLearning else { return }
        let done = expectation(description: "Measured lesson finishes")
        let observation = workspace.$isAssessingLearning.filter { !$0 }.prefix(1).sink { _ in done.fulfill() }
        await fulfillment(of: [done], timeout: 25); observation.cancel()
    }
}
