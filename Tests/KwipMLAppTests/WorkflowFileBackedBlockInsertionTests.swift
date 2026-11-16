import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowFileBackedBlockInsertionTests: XCTestCase {
    @MainActor func testActualPaletteBlockInsertionUndoRedoAndReopenInManagedAndExternalRoots() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for external in [false, true] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = WorkflowDocument(projectID: UUID().uuidString, name: "Empty canvas")
            let library: ProjectLibrary
            if external {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let root = try FlowDocumentStore.commitMigration(FlowDocumentMigration.plan(source), to: directory.appendingPathComponent("Chosen"))
                library = ProjectLibrary(directory: directory.appendingPathComponent("private"), externalFlowRoot: root, projectID: source.projectID)
            } else {
                library = ProjectLibrary(directory: directory)
                _ = try library.saveSource(.workflow(source)); _ = try library.activateFileBackedWorkflow(id: source.projectID)
            }
            let model = try ComponentRegistry.builtIn(), data = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
            let owner = try WorkflowAuthoringStore(library: library, projectID: source.projectID, modelRegistry: model, dataRegistry: data)
            let canvas = try WorkflowCanvasAdapter(owner: owner, modelRegistry: model, dataRegistry: data)
            let block = try XCTUnwrap(canvas.creationEntries.first { $0.definition != nil })
            canvas.addNode(type: block.id, x: 100, y: 120)
            XCTAssertEqual(owner.document.nodes.count, 1, owner.error ?? canvas.statusMessage)
            XCTAssertEqual(owner.document.definitions.count, 1)
            let inserted = owner.document
            XCTAssertTrue(owner.undo(lease: owner.lease()))
            XCTAssertTrue(owner.document.nodes.isEmpty); XCTAssertTrue(owner.document.definitions.isEmpty)
            XCTAssertTrue(owner.redo(lease: owner.lease()))
            XCTAssertEqual(owner.document.nodes, inserted.nodes); XCTAssertEqual(owner.document.definitions, inserted.definitions)
            XCTAssertEqual(try library.loadSource(id: source.projectID), .workflow(owner.document))
            owner.invalidate()
        }
    }
}
