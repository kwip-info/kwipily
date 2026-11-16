import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelChildEditorSessionTests:XCTestCase {
    @MainActor func testReopeningAfterExternalApplyCreatesFreshEditableSession() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        let owner=try ModelAuthoringStore(library:library,projectID:id),session=ModelChildEditorSession(owner:owner)
        XCTAssertTrue(session.open(path:["features"]))
        let first=try XCTUnwrap(session.editor)
        XCTAssertTrue(owner.applyDraft(lease:owner.lease(path:["features"])))
        XCTAssertTrue(first.isBusy)
        XCTAssertTrue(session.open(path:["features"]))
        XCTAssertFalse(first === session.editor)
        XCTAssertEqual(session.editor?.editingDraft,true)
        XCTAssertEqual(session.editor?.isBusy,false)
        XCTAssertFalse(first.historyContextIsActive)
    }
    @MainActor func testCloseRetainsDraftReopenSharesOwnerAndApplyIsParentUndo() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        let source=CompositeExamples.parallelClassifier
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        let session=ModelChildEditorSession(owner:owner)
        XCTAssertTrue(session.open(path:["features"]))
        let first=try XCTUnwrap(session.editor)
        XCTAssertTrue(session.open(path:["features"]))
        XCTAssertTrue(first === session.editor)
        first.moveNodes(ids:["branch-a"],dx:20,dy:0)
        session.close()
        XCTAssertNil(session.editor)
        XCTAssertFalse(first.historyContextIsActive)
        XCTAssertThrowsError(try owner.executionSnapshot())
        XCTAssertTrue(session.open(path:["features"]))
        XCTAssertEqual(session.editor?.scope?.nodes.first(where:{$0.id == "branch-a"})?.x,235)
        XCTAssertTrue(session.apply())
        XCTAssertNil(session.editor)
        XCTAssertNoThrow(try owner.executionSnapshot())
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        guard case .blockModel(let restored)=owner.document else{return XCTFail("Block document missing")}
        XCTAssertEqual(restored.source,source)
        XCTAssertTrue(session.open(path:["features"]))
        session.invalidate()
        XCTAssertFalse(session.open(path:["features"]))
    }
    @MainActor func testBusyAndInvalidBoundaryKeepCurrentEditorAndDraft() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        let owner=try ModelAuthoringStore(library:library,projectID:id),session=ModelChildEditorSession(owner:owner)
        owner.setExecutionBusy(true)
        XCTAssertFalse(session.open(path:["features"]))
        owner.setExecutionBusy(false)
        XCTAssertTrue(session.open(path:["features"]))
        let editor=try XCTUnwrap(session.editor)
        var invalid=try XCTUnwrap(editor.activeDefinition);invalid.outputs=[]
        XCTAssertTrue(owner.updateDraft(invalid,lease:owner.lease(path:["features"])))
        XCTAssertFalse(session.apply())
        XCTAssertTrue(editor === session.editor)
        XCTAssertNotNil(session.error)
        XCTAssertTrue(session.discard())
        XCTAssertNil(session.editor)
        XCTAssertNoThrow(try owner.executionSnapshot())
    }
}
