import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModuleCompositionOwnerTests: XCTestCase {
    private func selected(_ definition: CompositeDefinition) -> CompositeDefinition {
        var copy = definition
        copy.composition = .init(kind: .graph, children: (copy.body.nodes.map(\.id) + copy.body.instances.map(\.id)).map { .init(localID: $0) })
        return copy
    }
    @MainActor func testModelApplyUndoRestoresOriginalEnvelopeAndRedoRestoresPolicy() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory), id = UUID().uuidString
        let original = ModelAuthoringDocument(projectID: id, source: CompositeExamples.parallelClassifier)
        _ = try library.saveDocument(.blockModel(original), id: id)
        let owner = try ModelAuthoringStore(library: library, projectID: id)
        XCTAssertTrue(owner.beginDraft(path: ["features"], lease: owner.lease()))
        guard case .blockModel(let draft) = owner.document else { return XCTFail("Expected model draft") }
        XCTAssertTrue(owner.updateDraft(selected(draft.drafts[0].proposed), lease: owner.lease(path: ["features"])), owner.error ?? "")
        XCTAssertTrue(owner.applyDraft(lease: owner.lease(path: ["features"])), owner.error ?? "")
        let applied = owner.document
        XCTAssertTrue(owner.undo(lease: owner.lease()), owner.error ?? "")
        guard case .blockModel(let restored) = owner.document else { return XCTFail("Expected model") }
        XCTAssertEqual(restored.schemaVersion, original.schemaVersion)
        XCTAssertEqual(restored.source, original.source)
        XCTAssertTrue(restored.drafts.isEmpty)
        XCTAssertTrue(owner.redo(lease: owner.lease()), owner.error ?? "")
        guard case .blockModel(let replay) = owner.document, case .blockModel(let accepted) = applied else { return XCTFail("Expected model") }
        XCTAssertEqual(replay.source, accepted.source); XCTAssertEqual(replay.schemaVersion, 2)
    }
    @MainActor func testWorkflowChildProjectionAndWholeOwnerUndoRetainCorrectEnvelope() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory), id = UUID().uuidString
        let model = ModelAuthoringDocument(projectID: id, source: CompositeExamples.parallelClassifier)
        let original = try WorkflowCodec.migrateModel(.blockModel(model), projectID: id).document
        _ = try library.saveSource(.workflow(original), id: id)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner = try WorkflowAuthoringStore(library: library, projectID: id, modelRegistry: .builtIn(),
            dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
        let root = WorkflowNodeAddress(domain: .model, localID: "features")
        XCTAssertTrue(owner.beginDraft(root: root, lease: owner.lease()))
        XCTAssertTrue(owner.updateDraft(selected(owner.document.drafts[0].proposed), lease: owner.blockLease(root: root)), owner.error ?? "")
        let authority = try WorkflowBlockCanvasAuthority(owner: owner, root: root)
        guard case .blockModel(let presentation) = authority.canvasDocument else { return XCTFail("Expected projected model") }
        XCTAssertEqual(presentation.schemaVersion, 2); XCTAssertEqual(presentation.source.schemaVersion, 3)
        let child = try ModelCanvasAdapter(owner: authority, path: ["features"])
        XCTAssertNil(child.projectionError)
        XCTAssertTrue(owner.applyDraft(lease: owner.blockLease(root: root)), owner.error ?? "")
        let applied = owner.document
        XCTAssertTrue(owner.undo(lease: owner.lease()), owner.error ?? "")
        var baseline = original; baseline.revision = owner.document.revision
        XCTAssertEqual(owner.document, baseline)
        XCTAssertTrue(owner.redo(lease: owner.lease()), owner.error ?? "")
        var replay = applied; replay.revision = owner.document.revision
        XCTAssertEqual(owner.document, replay)
    }
}
