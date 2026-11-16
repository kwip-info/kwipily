import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModuleCompositionHistoryFocusTests: XCTestCase {
    func testStableProviderFollowsRootAndChildWithoutDiscardingSourceHistory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let drafts = TensorLearningDraftSession(repository: .init(directory: directory))
        let project = UUID().uuidString
        XCTAssertTrue(drafts.create(projectID: project, operation: ModuleCompositionLearningTask.orderName.rawValue, mode: .guided))
        let prepared = try ModuleCompositionLearningDraft.model(projectID: project, task: .orderName, prepared: true)
        XCTAssertTrue(drafts.update { $0.compositionModel = prepared })
        var busy = false
        let session = try ModuleCompositionLearningCanvasSession(drafts: drafts, registry: .builtIn(), isCurrent: { true }, isBusy: { busy })
        let root = try XCTUnwrap(session.activeCanvas)
        XCTAssertTrue(session.historyProvider.context?.represents(root) == true)
        let before = try XCTUnwrap(drafts.draft?.compositionModel)
        let node = try XCTUnwrap(root.graph.nodes.first)
        root.moveNodes(ids: [node.id], dx: 25, dy: 10)
        XCTAssertTrue(session.historyProvider.context?.canPerform(.undo) == true)
        let rootContext = session.historyProvider.context
        let instance = try XCTUnwrap(before.source.root.instances.first)
        session.open([instance.id])
        let child = try XCTUnwrap(session.activeCanvas)
        XCTAssertTrue(session.historyProvider.context?.represents(child) == true)
        XCTAssertFalse(rootContext?.canPerform(.undo) == true, "A previous root focus cannot consume history in a child tab")
        XCTAssertFalse(session.historyProvider.context?.canPerform(.undo) == true, "Unapplied child changes must be resolved first")
        XCTAssertTrue(session.discard())
        session.open([])
        XCTAssertTrue(session.historyProvider.context?.represents(root) == true)
        busy = true
        XCTAssertFalse(session.historyProvider.context?.canPerform(.undo) == true)
        busy = false
        XCTAssertTrue(session.historyProvider.context?.canPerform(.undo) == true)
        session.historyProvider.context?.perform(.undo)
        XCTAssertEqual(drafts.draft?.compositionModel?.source.root, before.source.root)
        XCTAssertTrue(session.historyProvider.context?.canPerform(.redo) == true)
        session.historyProvider.context?.perform(.redo)
        XCTAssertNotEqual(drafts.draft?.compositionModel?.source.root, before.source.root)
        let finalContext = session.historyProvider.context
        session.invalidate()
        XCTAssertNil(session.historyProvider.context)
        XCTAssertFalse(finalContext?.canPerform(.undo) == true)
    }
}
