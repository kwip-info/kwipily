import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SeparableConv2dWorkflowExampleTests: XCTestCase {
    @MainActor func testExampleEntryCreatesPortableProjectAndPreservesSourceOnReopen() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadSeparableConv2dWorkflowExample(device: "cpu"), workspace.statusMessage)
        let session = try XCTUnwrap(workspace.workflowSession)
        let document = session.owner.document
        let path = try XCTUnwrap(session.bindings.path(for: WorkflowExamples.sourceAddress))
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: session.owner.modelRegistry, dataRegistry: session.owner.dataRegistry,contractVersion:15)
        XCTAssertEqual(plan.settings.device, "cpu")
        XCTAssertEqual(plan.model.graph.nodes.filter { $0.type == "depthwise_conv2d" }.count, 1)
        XCTAssertFalse(String(decoding: try WorkflowCodec.encode(document), as: UTF8.self).contains(path))
        XCTAssertTrue(workspace.beginWorkflowProject(id: session.owner.projectID))
        XCTAssertEqual(workspace.workflowSession?.owner.document, document)
        XCTAssertEqual(workspace.workflowSession?.bindings.path(for: WorkflowExamples.sourceAddress), path)
        XCTAssertEqual(document.definitions.count, 1)
        XCTAssertEqual(plan.model.graph.nodes.filter { $0.type == "conv2d" }.count, 1)
        XCTAssertEqual(document.definitions[0].parameters.count, 11)
    }
}
