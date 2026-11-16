import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparedWorkflowCodeBoundaryTests: XCTestCase {
    @MainActor func testPreparedThreeColumnModelCanBeReviewedInCodeBeforeRunning() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        let document = try PreparedWorkflowExample.preparationPractice.document(device: "cpu")
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let source = try workspace.codeGenerationGraph()
        XCTAssertEqual(source.nodes.first { $0.type == "input" }?.config["features"], .integer(3))
        XCTAssertTrue(workspace.codeValidation.isValid, workspace.codeValidation.issues.map(\.message).joined(separator: "\n"))
    }
    @MainActor func testOrdinaryToyCodeRouteRetainsItsDatasetBoundary() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        let model = try PreparationPracticeExample.document(device: "cpu")
        let graph = try WorkflowCompiler.modelSnapshot(model, definition: WorkflowNodeAddress(domain: .control, localID: "model"),
            modelRegistry: workspace.registry, dataRegistry: workspace.workflowDataRegistry()).graph
        XCTAssertTrue(workspace.beginProject(graph))
        XCTAssertFalse(workspace.codeValidation.isValid)
    }
}
