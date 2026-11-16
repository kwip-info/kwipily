import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowProjectRoutingTests: XCTestCase {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @MainActor func testLibraryOpensCanonicalWorkflowSavesAndExportsAllDomainsThenRetiresOnLegacyTransition() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        let document = WorkflowDocument(projectID: UUID().uuidString, revision: 3, name: "Unified project", nodes: [
            .init(address: .init(domain: .data, localID: "input"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:])),
            .init(address: .init(domain: .control, localID: "train"), payload: .train(config: [:]))
        ])
        let record = try workspace.projectLibrary.saveSource(.workflow(document))
        workspace.refreshProjects()
        XCTAssertTrue(workspace.projects.contains { $0.id == record.id })
        workspace.openManagedProject(record)
        let host = try XCTUnwrap(workspace.workflowSession)
        let history = try XCTUnwrap(workspace.workflowHistory)
        XCTAssertTrue(history.isActive)
        XCTAssertEqual(history.projectID, document.projectID)
        XCTAssertEqual(history.sessionID, host.owner.sessionID)
        XCTAssertEqual(workspace.activeProjectID, document.projectID)
        XCTAssertEqual(host.owner.document, document)
        XCTAssertNil(workspace.modelAuthoring)
        XCTAssertEqual(workspace.modelSourceName, document.name)
        XCTAssertTrue(workspace.graph.nodes.isEmpty, "No expanded editable graph may replace workflow source")
        XCTAssertTrue(workspace.saveManagedProject())
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: record.id), .workflow(document))
        let exported = root.appendingPathComponent("portable.kwipml")
        try workspace.exportModelProject(to: exported)
        XCTAssertEqual(try ProjectSourceCodec.decode(Data(contentsOf: exported)), .workflow(document))
        workspace.graph = ExampleFlows.parallelClassifier
        workspace.setExecution { $0.epochs = 999 }
        XCTAssertEqual(host.owner.document, document)
        XCTAssertTrue(workspace.graph.nodes.isEmpty)
        workspace.renameProject("Renamed workflow")
        XCTAssertEqual(host.owner.document.name, "Renamed workflow")
        XCTAssertTrue(workspace.canUndo)
        workspace.undo()
        XCTAssertEqual(host.owner.document.name, document.name)
        XCTAssertEqual(host.owner.document.nodes, document.nodes)
        XCTAssertTrue(workspace.canRedo)
        workspace.redo()
        XCTAssertEqual(host.owner.document.name, "Renamed workflow")
        let lease = host.owner.lease()
        XCTAssertTrue(workspace.beginProject(ExampleFlows.parallelClassifier))
        XCTAssertNil(workspace.workflowSession)
        XCTAssertNil(workspace.workflowHistory)
        XCTAssertFalse(history.isActive)
        XCTAssertTrue(host.isRetired)
        XCTAssertFalse(host.owner.isCurrent(lease))
    }

    @MainActor func testLegacyBeginCannotReplaceWorkflowIdentityAndFailedOpenKeepsCurrentSession() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        let document = WorkflowDocument(projectID: UUID().uuidString, name: "Retain source")
        let record = try workspace.projectLibrary.saveSource(.workflow(document))
        workspace.openManagedProject(record)
        let host = try XCTUnwrap(workspace.workflowSession)
        XCTAssertFalse(workspace.beginProject(ExampleFlows.parallelClassifier, id: record.id))
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: record.id), .workflow(document))
        XCTAssertFalse(workspace.beginWorkflowProject(id: UUID().uuidString))
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertTrue(host.owner.isCurrent(host.owner.lease()))
        let corruptTarget = WorkflowDocument(projectID: UUID().uuidString, name: "Unreadable private bindings")
        _ = try workspace.projectLibrary.saveSource(.workflow(corruptTarget))
        let bindingDirectory = root.appendingPathComponent("workflow-sources/\(corruptTarget.projectID)")
        try FileManager.default.createDirectory(at: bindingDirectory, withIntermediateDirectories: true)
        try Data("invalid bindings".utf8).write(to: bindingDirectory.appendingPathComponent("bindings.json"))
        XCTAssertFalse(workspace.beginWorkflowProject(id: corruptTarget.projectID))
        XCTAssertTrue(workspace.workflowSession === host)
        XCTAssertEqual(workspace.activeProjectID, document.projectID)
        XCTAssertTrue(host.owner.isCurrent(host.owner.lease()))
    }
    @MainActor func testNewFlowCreatesCanonicalWorkflowAndPreservesGlobalCanvasDefaults() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        let defaults = CanvasPresentationSettings(flowDirection: .topToBottom, nodeDetail: .small)
        XCTAssertTrue(workspace.setCanvasDefaults(defaults))
        workspace.newProject()
        let first = try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(first.owner.document.name, "Untitled flow")
        XCTAssertTrue(first.owner.document.nodes.isEmpty)
        XCTAssertEqual(first.canvas.canvasViewState.presentation, defaults)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: first.owner.projectID), .workflow(first.owner.document))
        workspace.newProject()
        let second = try XCTUnwrap(workspace.workflowSession)
        XCTAssertNotEqual(first.owner.projectID, second.owner.projectID)
        XCTAssertTrue(first.isRetired)
        XCTAssertEqual(second.canvas.canvasViewState.presentation, defaults)
        XCTAssertEqual(workspace.projects.count, 2)
    }

}
