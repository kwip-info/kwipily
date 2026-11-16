import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModuleCompositionInspectorTests: XCTestCase {
    @MainActor private func setup(width: Int = 2) throws -> (URL, ModelAuthoringStore, ModelCanvasAdapter) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let reference = CompositeDefinitionReference("steps")
        let definition = CompositeDefinition(reference: reference, title: "Two steps", body: .init(nodes: [
            .init(id: "a", type: "linear", name: "First", config: ["in_features": 2, "out_features": .integer(Int64(width))]),
            .init(id: "b", type: "linear", name: "Second", config: ["in_features": .integer(Int64(width)), "out_features": 2])
        ], edges: [.init(id: "ab", source: "a", target: "b")]), inputs: [.init("in", targets: [.init("a", "in")])], outputs: [.init("out", source: .init("b", "out"))])
        let source = CompositeProjectDocument(root: .init(nodes: [.init(id: "input", type: "input", config: ["features": 2]), .init(id: "output", type: "output")], instances: [.init(id: "block", definition: reference)], edges: [.init(id: "enter", source: "input", target: "block"), .init(id: "leave", source: "block", target: "output")]), definitions: [definition])
        let library = ProjectLibrary(directory: directory), id = UUID().uuidString
        _ = try library.saveDocument(.blockModel(.init(projectID: id, source: source)), id: id)
        let owner = try ModelAuthoringStore(library: library, projectID: id)
        XCTAssertTrue(owner.beginDraft(path: ["block"], lease: owner.lease()))
        return (directory, owner, try ModelCanvasAdapter(owner: owner, path: ["block"]))
    }
    @MainActor func testSequenceEditsRealParentDraftAndStaleCallbackCannotReorder() throws {
        let (directory, owner, adapter) = try setup(); defer { try? FileManager.default.removeItem(at: directory) }
        let old = ModuleCompositionInspectorContext(adapter: adapter)
        XCTAssertTrue(old.perform(.select(.sequential)), old.rejection ?? "")
        XCTAssertEqual(adapter.activeDefinition?.composition?.children.map(\.localID), ["a", "b"])
        XCTAssertFalse(old.perform(.move("b", 0)))
        let current = ModuleCompositionInspectorContext(adapter: adapter)
        XCTAssertTrue(current.perform(.move("b", 0)), current.rejection ?? "")
        XCTAssertEqual(adapter.activeDefinition?.inputs.first?.targets, [.init("b", "in")])
        XCTAssertEqual(adapter.activeDefinition?.body.edges.first?.id, "ab")
        XCTAssertEqual(adapter.activeDefinition?.body.edges.first?.source, "b")
        guard case .blockModel(let model) = owner.document else { return XCTFail("Expected parent") }
        XCTAssertEqual(model.schemaVersion, 2); XCTAssertEqual(model.source.schemaVersion, 3)
        XCTAssertNil(model.source.definitions[0].composition, "Only the owned draft changes before Apply")
    }
    @MainActor func testNamedKeyEditsPreserveLabelsAndRejectReservedNamesAtomically() throws {
        let (directory, owner, adapter) = try setup(); defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(ModuleCompositionInspectorContext(adapter: adapter).perform(.select(.namedMap)))
        let before = owner.document
        let invalid = ModuleCompositionInspectorContext(adapter: adapter)
        XCTAssertFalse(invalid.perform(.rename("a", "forward"))); XCTAssertNotNil(invalid.rejection)
        XCTAssertEqual(owner.document, before)
        XCTAssertTrue(ModuleCompositionInspectorContext(adapter: adapter).perform(.rename("a", "first_step")))
        XCTAssertEqual(adapter.activeDefinition?.composition?.children.first?.key, "first_step")
        XCTAssertEqual(adapter.activeDefinition?.body.nodes.first?.name, "First")
        XCTAssertEqual(adapter.activeDefinition?.body.edges.first?.source, "a")
        adapter.invalidate()
        XCTAssertFalse(ModuleCompositionInspectorContext(adapter: adapter).perform(.rename("a", "after_close")))
    }
    @MainActor func testKnownParentShapeMismatchCannotCommitSequenceReorder() throws {
        let (directory, owner, adapter) = try setup(width: 3); defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(ModuleCompositionInspectorContext(adapter: adapter).perform(.select(.sequential)))
        let before = owner.document, edit = ModuleCompositionInspectorContext(adapter: adapter)
        XCTAssertFalse(edit.perform(.move("b", 0))); XCTAssertNotNil(edit.rejection)
        XCTAssertEqual(owner.document, before)
    }
}
