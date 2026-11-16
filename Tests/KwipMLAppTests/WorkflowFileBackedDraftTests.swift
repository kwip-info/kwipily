import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowFileBackedDraftTests: XCTestCase {
    @MainActor func testDraftBeginEditRestartDiscardApplyUndoRedoInManagedAndExternalRoots() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = try ComponentRegistry.builtIn()
        let data = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        for external in [false, true] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
            let library: ProjectLibrary
            if external {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let url = try FlowDocumentStore.commitMigration(FlowDocumentMigration.plan(source), to: directory.appendingPathComponent("Chosen"))
                library = ProjectLibrary(directory: directory.appendingPathComponent("private"), externalFlowRoot: url, projectID: source.projectID)
            } else {
                library = ProjectLibrary(directory: directory)
                _ = try library.saveSource(.workflow(source)); _ = try library.activateFileBackedWorkflow(id: source.projectID)
            }
            func openOwner() throws -> WorkflowAuthoringStore {
                try WorkflowAuthoringStore(library: library, projectID: source.projectID, modelRegistry: model, dataRegistry: data)
            }
            let owner = try openOwner(), root = WorkflowNodeAddress(domain: .model, localID: "convolution")
            let stale = owner.blockLease(root: root)
            XCTAssertTrue(owner.beginDraft(root: root, lease: owner.lease()), owner.error ?? "")
            XCTAssertEqual(owner.document.drafts.first?.baseRevision, owner.document.revision)
            let main = try XCTUnwrap(owner.flowRootURL)
            let envelope = try FlowDocumentStore.readHead(at: main)
            XCTAssertTrue(try XCTUnwrap(envelope.draftDependencies.first).relativePath.hasPrefix(".kwipflow-draft-"))
            guard var proposal = owner.document.drafts.first?.proposed else { XCTFail("Draft must persist"); continue }
            proposal.title = "Reviewed spatial block"
            XCTAssertFalse(owner.updateDraft(proposal, lease: stale))
            XCTAssertTrue(owner.updateDraft(proposal, lease: owner.blockLease(root: root)), owner.error ?? "")
            let edited = owner.document
            XCTAssertThrowsError(try WorkflowCompiler.compile(edited,
                train: .init(domain: .control, localID: "train"), modelRegistry: model, dataRegistry: data)) { error in
                XCTAssertEqual((error as? WorkflowCompilerError)?.code, .unresolvedDraft)
            }
            XCTAssertEqual(edited.definitions, source.definitions)
            XCTAssertEqual(try library.loadSource(id: source.projectID), .workflow(edited))
            owner.invalidate()
            let reopened = try openOwner()
            XCTAssertEqual(reopened.document, edited)
            XCTAssertTrue(reopened.discardDraft(lease: reopened.blockLease(root: root)), reopened.error ?? "")
            XCTAssertTrue(reopened.document.drafts.isEmpty)
            XCTAssertEqual(reopened.document.definitions, source.definitions)
            XCTAssertTrue(reopened.beginDraft(root: root, lease: reopened.lease()), reopened.error ?? "")
            proposal = try XCTUnwrap(reopened.document.drafts.first?.proposed)
            proposal.title = "Applied private block"
            XCTAssertTrue(reopened.updateDraft(proposal, lease: reopened.blockLease(root: root)), reopened.error ?? "")
            XCTAssertTrue(reopened.applyDraft(lease: reopened.blockLease(root: root)), reopened.error ?? "")
            let applied = reopened.document
            XCTAssertTrue(applied.drafts.isEmpty)
            XCTAssertTrue(applied.definitions.contains { $0.title == "Applied private block" })
            XCTAssertTrue(reopened.undo(lease: reopened.lease()), reopened.error ?? "")
            XCTAssertTrue(reopened.document.drafts.isEmpty)
            XCTAssertEqual(reopened.document.definitions, source.definitions)
            XCTAssertTrue(reopened.redo(lease: reopened.lease()), reopened.error ?? "")
            XCTAssertEqual(reopened.document.definitions, applied.definitions)
            XCTAssertEqual(try library.loadSource(id: source.projectID), .workflow(reopened.document))
            reopened.invalidate()
        }
    }
}
