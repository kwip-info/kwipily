import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowAuthoringBlockTests: XCTestCase {
    private let block = WorkflowNodeAddress(domain: .model, localID: "features")
    private let data = WorkflowNodeAddress(domain: .data, localID: "features")
    @MainActor private func setup(nested: Bool = false) throws -> (URL, ProjectLibrary, WorkflowAuthoringStore) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = ProjectLibrary(directory: directory)
        var source = CompositeExamples.parallelClassifier
        if nested {
            let pin = source.root.instances[0].definition
            let wrapper = CompositeDefinition(reference: .init("wrapper"), title: "Wrapper",
                body: .init(instances: [.init(id: "nested", definition: pin)]),
                inputs: [.init("in", targets: [.init("nested", "in")])],
                outputs: [.init("out", source: .init("nested", "out"))])
            source.definitions.append(wrapper); source.root.instances[0].definition = wrapper.reference
        }
        let model = ModelAuthoringDocument(projectID: UUID().uuidString, revision: 4, source: source)
        var document = try WorkflowCodec.migrateModel(.blockModel(model), projectID: model.projectID).document
        document.nodes.append(.init(address: data, name: "Independent data", payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator": .string(",")])))
        document.nodes.append(.init(address: .init(domain: .control, localID: "setup"), name: "Experiment", payload: .runSetup(config: ["seed": .integer(17)])))
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        return (directory, library, try reopen(library, id: document.projectID))
    }
    @MainActor private func reopen(_ library: ProjectLibrary, id: String) throws -> WorkflowAuthoringStore {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowAuthoringStore(library: library, projectID: id, modelRegistry: .builtIn(),
            dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    @MainActor func testNestedApplyCreatesOneWholeWorkflowUndoBaselineAndRetainsSelection() throws {
        let (directory, library, store) = try setup(nested: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = store.document
        XCTAssertTrue(store.select(.init(nodeAddresses: [block, data]), lease: store.lease()))
        let selected = store.selection
        XCTAssertTrue(store.beginDraft(root: block, nestedPath: ["nested"], lease: store.lease()))
        XCTAssertFalse(store.canUndo); XCTAssertEqual(store.document.revision, original.revision)
        XCTAssertFalse(store.beginDraft(root: block, lease: store.lease()), "A different path cannot replace the active nested draft")
        let first = store.blockLease(root: block, nestedPath: ["nested"])
        var proposal = try XCTUnwrap(store.document.drafts.first).proposed
        proposal.title = "Edited nested block"
        XCTAssertTrue(store.updateDraft(proposal, lease: first))
        XCTAssertFalse(store.isCurrent(first.owner)); XCTAssertEqual(store.document.revision, original.revision)
        XCTAssertTrue(store.applyDraft(lease: store.blockLease(root: block, nestedPath: ["nested"])))
        let applied = store.document
        XCTAssertEqual(applied.revision, original.revision + 1); XCTAssertTrue(applied.drafts.isEmpty)
        XCTAssertEqual(store.selection, selected); XCTAssertEqual(applied.edges, original.edges)
        XCTAssertEqual(applied.nodes.first { $0.address == data }, original.nodes.first { $0.address == data })
        XCTAssertEqual(applied.nodes.filter { $0.address.domain != .model }, original.nodes.filter { $0.address.domain != .model })
        XCTAssertEqual(try library.loadSource(id: store.projectID), .workflow(applied))
        XCTAssertTrue(store.undo(lease: store.lease()))
        var baseline = original; baseline.revision = original.revision + 2
        XCTAssertEqual(store.document, baseline); XCTAssertEqual(store.selection, selected); XCTAssertFalse(store.canUndo)
        XCTAssertTrue(store.redo(lease: store.lease()))
        var replay = applied; replay.revision = original.revision + 3
        XCTAssertEqual(store.document, replay)
    }
    @MainActor func testDraftNoOpsDiscardAndWrongQualifiedContextDoNotInventHistory() throws {
        let (directory, _, store) = try setup(); defer { try? FileManager.default.removeItem(at: directory) }
        let original = store.document
        XCTAssertFalse(store.beginDraft(root: data, lease: store.lease()))
        XCTAssertEqual(store.document, original)
        XCTAssertTrue(store.beginDraft(root: block, lease: store.lease()))
        let lease = store.blockLease(root: block), begun = store.document
        XCTAssertTrue(store.beginDraft(root: block, lease: store.lease()))
        XCTAssertEqual(store.lease(), lease.owner)
        XCTAssertTrue(store.updateDraft(try XCTUnwrap(begun.drafts.first).proposed, lease: lease))
        XCTAssertEqual(store.lease(), lease.owner)
        XCTAssertFalse(store.discardDraft(lease: store.blockLease(root: data)))
        XCTAssertFalse(store.discardDraft(lease: store.blockLease(root: block, nestedPath: ["wrong"])))
        XCTAssertEqual(store.document, begun)
        XCTAssertTrue(store.discardDraft(lease: lease)); XCTAssertEqual(store.document, original)
        XCTAssertFalse(store.canUndo); XCTAssertFalse(store.canRedo)
        XCTAssertTrue(store.beginDraft(root: block, lease: store.lease()))
        XCTAssertFalse(store.discardDraft(lease: lease), "A previous draft's callback cannot discard a new draft")
    }
    @MainActor func testBusyReopenAndFailedPersistenceKeepDraftAndLease() throws {
        let (directory, library, store) = try setup(); defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertTrue(store.beginDraft(root: block, lease: store.lease()))
        let token = store.blockLease(root: block), before = store.document
        var proposed = try XCTUnwrap(before.drafts.first).proposed; proposed.title = "Changed"
        store.setExecutionBusy(true)
        XCTAssertFalse(store.updateDraft(proposed, lease: token)); XCTAssertFalse(store.discardDraft(lease: token))
        XCTAssertFalse(store.applyDraft(lease: token)); XCTAssertEqual(store.document, before)
        store.setExecutionBusy(false)
        let reopened = try reopen(library, id: store.projectID)
        XCTAssertEqual(reopened.document, before)
        XCTAssertFalse(reopened.updateDraft(proposed, lease: token)); XCTAssertFalse(reopened.canUndo)
        let url = try library.documentURL(id: store.projectID), bytes = try Data(contentsOf: url)
        try FileManager.default.removeItem(at: url); try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        XCTAssertFalse(store.updateDraft(proposed, lease: token)); XCTAssertFalse(store.applyDraft(lease: token))
        XCTAssertFalse(store.discardDraft(lease: token)); XCTAssertNotNil(store.error)
        XCTAssertEqual(store.document, before); XCTAssertEqual(store.lease(), token.owner)
        try FileManager.default.removeItem(at: url); try bytes.write(to: url)
        XCTAssertTrue(store.updateDraft(proposed, lease: token))
        store.invalidate()
        XCTAssertFalse(store.applyDraft(lease: store.blockLease(root: block)))
    }
    @MainActor func testInvalidApplyKeepsRecoverableDraftAndEarlierHistory() throws {
        let (directory, _, store) = try setup(); defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertNotNil(store.apply(.rename(address: data, name: "Edited data"), lease: store.lease()))
        XCTAssertTrue(store.beginDraft(root: block, lease: store.lease()))
        var proposal = try XCTUnwrap(store.document.drafts.first).proposed; proposal.body.edges = []
        XCTAssertTrue(store.updateDraft(proposal, lease: store.blockLease(root: block)))
        let before = store.document, token = store.blockLease(root: block)
        XCTAssertFalse(store.applyDraft(lease: token)); XCTAssertEqual(store.document, before)
        XCTAssertEqual(store.lease(), token.owner); XCTAssertFalse(store.canUndo)
        XCTAssertTrue(store.discardDraft(lease: token)); XCTAssertTrue(store.canUndo)
        XCTAssertTrue(store.undo(lease: store.lease()))
        XCTAssertEqual(store.document.nodes.first { $0.address == data }?.name, "Independent data")
    }
}
