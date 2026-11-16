import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TableInputViewPracticeTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testEditablePracticeCompilesAnActualSourceToViewWithoutPrivateBindings() throws {
        let example = PreparedWorkflowExample.component(TableInputViewPracticeExample.exampleID)
        let document = try example.document(device: "cpu")
        XCTAssertEqual(document.nodes.count, 2)
        XCTAssertEqual(document.edges.count, 1)
        let registry = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowDataPreparationPlan.compile(document, registry: registry)
        XCTAssertEqual(Set(plan.nodes.map(\.operatorID)), ["kwip.data.sourceCSV", "kwip.data.view"])
        XCTAssertEqual(plan.edges.first?.from.node, WorkflowExamples.sourceAddress.localID)
        XCTAssertEqual(plan.edges.first?.to.node, TableInputViewPracticeExample.viewAddress.localID)
        XCTAssertEqual(example.sourceResources.count, 1)
        XCTAssertEqual(example.sourceResources.first?.resource, "workflow-observations")
        let text = String(decoding: try WorkflowCodec.encode(document), as: UTF8.self)
        XCTAssertFalse(text.contains("/Users/"))
        XCTAssertFalse(text.contains("password"))
        XCTAssertFalse(text.contains("sourceBindings"))
    }

    @MainActor func testPracticeLaunchOwnsDurablePublicObservationsAndRemainsEditable() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(TableInputViewPracticeExample.exampleID, device: "cpu"), workspace.statusMessage)
        let session = try XCTUnwrap(workspace.workflowSession)
        let source = try XCTUnwrap(session.bindings.path(for: WorkflowExamples.sourceAddress))
        XCTAssertTrue(source.hasPrefix(root.appendingPathComponent("example-sources").path + "/"))
        XCTAssertEqual(try String(contentsOfFile: source, encoding: .utf8).split(whereSeparator: \.isNewline).count, 65)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: session.owner.projectID), .workflow(session.owner.document))
        XCTAssertEqual(session.owner.document.nodes.count, 2)
        XCTAssertEqual(workspace.activeSection, "Build")
    }
}
