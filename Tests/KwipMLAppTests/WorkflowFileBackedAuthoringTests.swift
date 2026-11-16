import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowFileBackedAuthoringTests: XCTestCase {
    @MainActor func testOrdinaryEditsUndoRedoKeepPinsAfterChildHeadChanges() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let library = ProjectLibrary(directory:root)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source))
        let original = try Data(contentsOf:library.documentURL(id:source.projectID))
        _ = try library.activateFileBackedWorkflow(id:source.projectID)
        let repo = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner = try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:.builtIn(),
            dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let envelope = try XCTUnwrap(owner.pinnedFlowDocument)
        let url = try XCTUnwrap(owner.flowRootURL)
        let link = try XCTUnwrap(envelope.dependencies.first)
        let childURL = url.deletingLastPathComponent().appendingPathComponent(link.relativePath)
        var child = try FlowDocumentCodec.decode(Data(contentsOf:childURL))
        child.revision += 1
        if case .modelBlock(var definition) = child.payload { definition.title = "New head only"; child.payload = .modelBlock(definition) }
        try FlowDocumentStore.save(child,at:childURL,expected:link.revision)
        XCTAssertTrue(owner.rename("Edited root",lease:owner.lease()),owner.error ?? "")
        XCTAssertEqual(owner.document.definitions,source.definitions)
        XCTAssertEqual(owner.pinnedFlowDocument?.dependencies,envelope.dependencies)
        XCTAssertTrue(owner.undo(lease:owner.lease()),owner.error ?? "")
        XCTAssertEqual(owner.document.name,source.name)
        XCTAssertTrue(owner.redo(lease:owner.lease()),owner.error ?? "")
        XCTAssertEqual(owner.document.name,"Edited root")
        XCTAssertEqual(owner.pinnedFlowDocument?.dependencies,envelope.dependencies)
        XCTAssertEqual(try library.loadSource(id:source.projectID),.workflow(owner.document))
        XCTAssertEqual(try Data(contentsOf:library.documentURL(id:source.projectID)),original)
        XCTAssertEqual(try owner.currentProjectRecord().path,url.path)
        // The parent-owned popup draft path must not silently mutate independent files.
        let block = try XCTUnwrap(owner.document.nodes.first { if case .modelBlock = $0.payload { return true }; return false })
        let savedChildBytes = try Data(contentsOf: childURL)
        XCTAssertTrue(owner.beginDraft(root:block.address,lease:owner.lease()), owner.error ?? "")
        XCTAssertEqual(owner.pinnedFlowDocument?.dependencies, envelope.dependencies)
        XCTAssertEqual(try Data(contentsOf: childURL), savedChildBytes)
    }
    @MainActor func testExplicitAdoptionIsUndoableAndRejectsStaleReviewedHead() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = ProjectLibrary(directory: root)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source))
        _ = try library.activateFileBackedWorkflow(id: source.projectID)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner = try WorkflowAuthoringStore(library: library, projectID: source.projectID,
            modelRegistry: .builtIn(), dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
        let envelope = try XCTUnwrap(owner.pinnedFlowDocument)
        let link = try XCTUnwrap(envelope.dependencies.first)
        let url = try XCTUnwrap(owner.flowRootURL).deletingLastPathComponent().appendingPathComponent(link.relativePath)
        var child = try FlowDocumentCodec.decode(Data(contentsOf: url))
        child.revision += 1
        guard case .modelBlock(var definition) = child.payload else { return XCTFail("Expected block") }
        definition.title = "Reviewed block revision"
        child.payload = .modelBlock(definition)
        try FlowDocumentStore.save(child, at: url, expected: link.revision)
        let proposal = try owner.prepareFlowAdoption(definition: link.definition, lease: owner.lease())
        XCTAssertEqual(owner.document.definitions, source.definitions)
        XCTAssertTrue(owner.adoptFlowRevision(proposal, lease: owner.lease()), owner.error ?? "")
        XCTAssertEqual(owner.document.definitions.first { $0.reference == link.definition }?.title, definition.title)
        XCTAssertTrue(owner.undo(lease: owner.lease()), owner.error ?? "")
        XCTAssertEqual(owner.document.definitions, source.definitions)
        XCTAssertTrue(owner.redo(lease: owner.lease()), owner.error ?? "")
        XCTAssertEqual(owner.pinnedFlowDocument?.dependencies.first?.revision, try FlowDocumentCodec.reference(child))
        XCTAssertTrue(owner.undo(lease: owner.lease()), owner.error ?? "")
        let stale = try owner.prepareFlowAdoption(definition: link.definition, lease: owner.lease())
        let before = owner.document
        let savedPin = try FlowDocumentCodec.reference(child)
        child.revision += 1
        definition.title = "Changed since review"; child.payload = .modelBlock(definition)
        try FlowDocumentStore.save(child, at: url, expected: savedPin)
        XCTAssertFalse(owner.adoptFlowRevision(stale, lease: owner.lease()))
        XCTAssertEqual(owner.document, before)
        XCTAssertEqual(try library.loadSource(id: source.projectID), .workflow(before))
        owner.setExecutionBusy(true)
        XCTAssertThrowsError(try owner.prepareFlowAdoption(definition: link.definition, lease: owner.lease()))
        owner.setExecutionBusy(false)
        let currentHead = try FlowDocumentCodec.reference(child)
        child.revision += 1
        definition.outputs[0].id = "renamed_boundary"
        child.payload = .modelBlock(definition)
        try FlowDocumentStore.save(child, at: url, expected: currentHead)
        XCTAssertThrowsError(try owner.prepareFlowAdoption(definition: link.definition, lease: owner.lease()))
        XCTAssertEqual(owner.document, before, "A removed wired boundary cannot strand parent connections")
    }

}
