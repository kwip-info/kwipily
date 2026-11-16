import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LazyLinearLearningStoreTests: XCTestCase {
    @MainActor func testActualWorkerWrongWidthRepairPrivateAssessmentAndReopenBothBiasModes() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        _ = try XCTUnwrap(workspace.runtime, workspace.runtimeError ?? "Development runtime unavailable")
        let lesson = TensorLearningStore(workspace: workspace)
        await verifyRuntimeForIntegration(workspace)
        let support = try XCTUnwrap(workspace.lazyLinearRuntimeCapability)
        func waitForGrade() async throws {
            let deadline = Date().addingTimeInterval(30)
            while workspace.isAssessingLearning && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
            XCTAssertFalse(workspace.isAssessingLearning, "Worker must finish within its finite lesson budget")
        }
        for bias in [false, true] {
            XCTAssertTrue(lesson.start(operation: LazyLinearAssessment.operation), lesson.statusMessage)
            let graph = GraphDocument(name: "My width experiment", nodes: [
                .init(id: "measurements", type: "input", name: "Two groups", x: 0, y: 0),
                .init(id: "layer", type: "lazy_linear", name: "My inferred layer", x: 200, y: 0, config: ["out_features": 5, "bias": .bool(bias)]),
                .init(id: "result", type: "output", name: "New features", x: 400, y: 0)
            ], edges: [.init(id: "first", source: "measurements", target: "layer"), .init(id: "second", source: "layer", target: "result")])
            XCTAssertTrue(lesson.drafts.update { $0.graph = graph })
            workspace.lazyLinearRuntimeCapability = nil
            XCTAssertFalse(lesson.canSubmit)
            lesson.submit()
            XCTAssertFalse(workspace.isAssessingLearning)
            XCTAssertTrue(lesson.statusMessage.contains("LazyLinear"), lesson.statusMessage)
            workspace.lazyLinearRuntimeCapability = support
            XCTAssertTrue(lesson.canSubmit, lesson.statusMessage)
            lesson.submit(); try await waitForGrade()
            XCTAssertEqual(lesson.grade?.status, .failed, lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved)
            XCTAssertTrue(lesson.drafts.update { $0.graph.nodes[1].config["out_features"] = 4 })
            XCTAssertFalse(lesson.resultIsCurrent, "Edited graph cannot reuse its old grade")
            XCTAssertTrue(lesson.setMode(.assessment))
            XCTAssertNil(lesson.formativeProjection)
            lesson.submit(); try await waitForGrade()
            XCTAssertEqual(lesson.grade?.status, .passed, lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved); XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertNil(lesson.formativeProjection, "Private shape/state fixtures must not become formative view data")
            let id = try XCTUnwrap(workspace.activeProjectID), saved = workspace.graph
            XCTAssertTrue(workspace.beginProject(GraphDocument(name: "Other work")))
            XCTAssertTrue(workspace.beginProject(saved, id: id))
            XCTAssertEqual(lesson.draft?.operation, LazyLinearAssessment.operation)
            XCTAssertEqual(lesson.grade?.status, .passed, lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved); XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertNil(lesson.formativeProjection)
        }
    }
}
