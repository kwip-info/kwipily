import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BilinearLearningStoreTests: XCTestCase {
    @MainActor func testBilinearCapabilityGateMeasurementAndReopen() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: repository.appendingPathComponent(".venv/bin/python").path) else { throw XCTSkip("Development runtime unavailable") }
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        await verifyRuntimeForIntegration(workspace)
        let verifiedCapability = try XCTUnwrap(workspace.bilinearRuntimeCapability)
        for bias in [false, true] {
            let operation = "bilinear-" + (bias ? "with-bias" : "without-bias")
            XCTAssertTrue(lesson.start(operation: operation))
            let roles = ["left", "right", "weight"] + (bias ? ["bias"] : [])
            var nodes = roles.map { GraphNode(id: $0, type: "input", name: $0, config: ["fixture": .string($0)]) }
            nodes += [GraphNode(id: "bilinear", type: "bilinear", config: ["bias": .bool(bias)]), GraphNode(id: "output", type: "output")]
            let ports = ["left": "a", "right": "b", "weight": "weight", "bias": "bias"]
            var edges = roles.map { GraphEdge(id: $0, source: $0, sourcePort: "out", target: "bilinear", targetPort: ports[$0]!) }
            edges.append(GraphEdge(id: "result", source: "bilinear", sourcePort: "out", target: "output", targetPort: "in"))
            XCTAssertTrue(lesson.drafts.update { $0.graph = GraphDocument(name: "Authored Bilinear", nodes: nodes, edges: edges) })
            workspace.bilinearRuntimeCapability = nil
            XCTAssertNotNil(lesson.runtimeReadinessError)
            XCTAssertFalse(lesson.canSubmit, "Bilinear requires its own verified live descriptor")
            lesson.submit()
            XCTAssertFalse(workspace.isAssessingLearning)
            XCTAssertTrue(lesson.statusMessage.contains("Bilinear"), lesson.statusMessage)
            workspace.bilinearRuntimeCapability = verifiedCapability
            XCTAssertTrue(lesson.canSubmit, lesson.statusMessage)
            lesson.submit()
            let deadline = Date().addingTimeInterval(30)
            while workspace.isAssessingLearning && Date() < deadline { try await Task.sleep(nanoseconds: 20_000_000) }
            XCTAssertEqual(lesson.grade?.status, .passed, lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved)
            XCTAssertTrue(lesson.resultIsCurrent)
            let projectID = try XCTUnwrap(workspace.activeProjectID), graph = workspace.graph
            XCTAssertTrue(workspace.beginProject(GraphDocument(name: "Other")))
            XCTAssertTrue(workspace.beginProject(graph, id: projectID))
            XCTAssertEqual(lesson.draft?.operation, operation)
            XCTAssertEqual(lesson.grade?.status, .passed, lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved)
            XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertTrue(lesson.setMode(.assessment))
            XCTAssertNil(lesson.formativeProjection)
            XCTAssertNil(lesson.grade)
        }
    }
}
