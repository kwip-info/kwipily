import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelAuthoringStoreTests:XCTestCase {
    @MainActor func testRootMutationsPreserveFlatBytesOnNoOpAndHistoryOnSaveFailure() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),record=try library.save(ExampleFlows.parallelClassifier)
        let store=try ModelAuthoringStore(library:library,projectID:record.id),token=store.lease()
        let url=try library.documentURL(id:record.id),original=try Data(contentsOf:url)
        XCTAssertTrue(store.editRootScope({_ in},lease:token))
        XCTAssertEqual(store.lease(),token);XCTAssertFalse(store.canUndo)
        XCTAssertEqual(try Data(contentsOf:url),original)
        XCTAssertTrue(store.editMetadata({name,execution in name="Edited";execution=ExecutionSettings(epochs:7)},lease:token))
        guard case .legacyFlat(let graph)=store.document else{return XCTFail("Flat edits must remain flat")}
        XCTAssertEqual(graph.name,"Edited");XCTAssertEqual(graph.execution?.epochs,7)
        XCTAssertFalse(store.editMetadata({name,_ in name="Stale"},lease:token))
        XCTAssertTrue(store.undo(lease:store.lease()))
        XCTAssertEqual(try Data(contentsOf:url),original)
        let stable=store.lease()
        try FileManager.default.removeItem(at:url);try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertFalse(store.editMetadata({name,_ in name="Unsaved"},lease:stable))
        XCTAssertEqual(store.lease(),stable);XCTAssertTrue(store.canRedo)
    }
    @MainActor func testCanonicalRootScopeMutationsPreservePinsAndRejectDraftOrBusyWrites() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        let source=CompositeExamples.parallelClassifier
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)),id:id)
        let store=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(store.editRootScope({$0.instances[0].name="My Block"},lease:store.lease()))
        guard case .blockModel(let edited)=store.document else{return XCTFail("Expected Block model")}
        XCTAssertEqual(edited.source.root.instances[0].definition,source.root.instances[0].definition)
        XCTAssertEqual(edited.source.definitions,source.definitions)
        XCTAssertEqual(edited.revision,1)
        let stable=store.lease()
        XCTAssertTrue(store.editRootScope({_ in},lease:stable));XCTAssertEqual(store.lease(),stable)
        XCTAssertFalse(store.editRootScope({$0.instances[0].definition.id="Unpinned"},lease:stable))
        XCTAssertEqual(store.lease(),stable);XCTAssertEqual(store.document,.blockModel(edited))
        XCTAssertTrue(store.beginDraft(path:[source.root.instances[0].id],lease:stable))
        XCTAssertFalse(store.editRootScope({$0.instances[0].name="Blocked"},lease:store.lease()))
        let child=store.lease(path:[source.root.instances[0].id])
        XCTAssertTrue(store.editDraftScope({_ in},lease:child));XCTAssertEqual(store.lease(path:child.instancePath),child)
        XCTAssertFalse(store.editDraftScope({_ in},lease:store.lease(path:["wrong-child"])))
        XCTAssertNotNil(store.applyDraftCommand(.move(nodeIDs:[source.definitions[0].body.nodes[0].id],offset:.init(x:12,y:0)),lease:child))
        guard case .blockModel(let drafted)=store.document else{return XCTFail("Expected draft")}
        XCTAssertEqual(drafted.revision,1);XCTAssertEqual(drafted.source,edited.source)
        XCTAssertFalse(store.canUndo)
        XCTAssertEqual(drafted.drafts[0].proposed.body.nodes[0].x,source.definitions[0].body.nodes[0].x+12)
        XCTAssertNil(store.applyDraftCommand(.move(nodeIDs:[source.definitions[0].body.nodes[0].id],offset:.init(x:12,y:0)),lease:child))
        let editLease=store.lease(path:[source.root.instances[0].id])
        XCTAssertTrue(store.editDraftScope({$0.nodes[0].name="Private node"},lease:editLease))
        guard case .blockModel(let renamed)=store.document else{return XCTFail("Expected draft")}
        XCTAssertEqual(renamed.drafts[0].proposed.body.nodes[0].name,"Private node")
        XCTAssertEqual(renamed.source,edited.source);XCTAssertEqual(renamed.revision,1)
        let fresh=store.lease(path:[source.root.instances[0].id]);store.setExecutionBusy(true)
        XCTAssertNil(store.applyDraftCommand(.select(.init(),mode:.replace),lease:fresh))
    }
    @MainActor func testRootCommandsPersistBeforePublishingAndSelectionIsEphemeral() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),record=try library.save(ExampleFlows.parallelClassifier)
        let store=try ModelAuthoringStore(library:library,projectID:record.id)
        let token=store.lease(),url=try library.documentURL(id:record.id)
        let before=try Data(contentsOf:url)
        let selected=store.applyRootCommand(.select(.init(nodeIDs:["input"]),mode:.replace),lease:token)
        XCTAssertEqual(selected?.after.selection.nodeIDs,["input"])
        XCTAssertEqual(store.lease(),token)
        XCTAssertFalse(store.canUndo)
        XCTAssertEqual(try Data(contentsOf:url),before)
        XCTAssertNotNil(store.applyRootCommand(.move(nodeIDs:["input"],offset:.init(x:20,y:0)),lease:token))
        guard case .legacyFlat(let moved)=store.document else{return XCTFail("Plain edits must stay flat")}
        XCTAssertEqual(moved.nodes[0].x,ExampleFlows.parallelClassifier.nodes[0].x+20)
        XCTAssertTrue(store.undo(lease:store.lease()))
        XCTAssertEqual(try Data(contentsOf:url),before)
        let stable=store.document,stableToken=store.lease()
        try FileManager.default.removeItem(at:url)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertNil(store.applyRootCommand(.move(nodeIDs:["input"],offset:.init(x:20,y:0)),lease:stableToken))
        XCTAssertEqual(store.document,stable)
        XCTAssertEqual(store.lease(),stableToken)
        XCTAssertTrue(store.canRedo)
    }
    @MainActor func testPromotionUndoRestoresFlatDocumentAndRejectsOtherSession() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),record=try library.save(ExampleFlows.parallelClassifier)
        let store=try ModelAuthoringStore(library:library,projectID:record.id)
        let other=try ModelAuthoringStore(library:library,projectID:record.id)
        let original=try Data(contentsOf:library.documentURL(id:record.id))
        XCTAssertFalse(store.replaceSource(CompositeExamples.parallelClassifier,lease:other.lease()))
        let definition=CompositeExamples.parallelClassifier.definitions[0]
        XCTAssertTrue(store.insertBlock(definition:definition,instance:.init(id:"new-block",definition:definition.reference),lease:store.lease()))
        XCTAssertTrue(store.undo(lease:store.lease()),store.error ?? "Undo failed")
        XCTAssertEqual(store.document,.legacyFlat(ExampleFlows.parallelClassifier))
        XCTAssertEqual(try Data(contentsOf:library.documentURL(id:record.id)),original)
        XCTAssertTrue(store.redo(lease:store.lease()))
        guard case .blockModel = store.document else{return XCTFail("Redo must restore Blocks")}
        XCTAssertEqual(try library.loadDocument(id:record.id),store.document)
    }
    @MainActor func testPersistBeforePublishFailureAndStaleLeases() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),record=try library.save(ExampleFlows.parallelClassifier)
        let store=try ModelAuthoringStore(library:library,projectID:record.id)
        let stale=store.lease()
        var source=CompositeExamples.parallelClassifier;source.name="Parent-owned"
        XCTAssertTrue(store.replaceSource(source,lease:stale))
        let accepted=store.document
        XCTAssertEqual(try library.loadDocument(id:record.id),accepted)
        XCTAssertFalse(store.replaceSource(source,lease:stale))
        XCTAssertEqual(store.document,accepted)
        let file=try library.documentURL(id:record.id)
        try FileManager.default.removeItem(at:file)
        try FileManager.default.createDirectory(at:file,withIntermediateDirectories:false)
        source.name="Must not publish"
        let token=store.lease()
        XCTAssertFalse(store.replaceSource(source,lease:token))
        XCTAssertEqual(store.document,accepted)
        XCTAssertEqual(store.lease(),token)
        XCTAssertNotNil(store.error)
    }
    @MainActor func testChildApplyIsOneUndoAndSharedInstanceRemainsPinned() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        var source=CompositeExamples.parallelClassifier
        source.root.instances.append(.init(id:"other",definition:source.root.instances[0].definition))
        let original=ModelAuthoringDocument(projectID:id,source:source)
        _=try library.saveDocument(.blockModel(original),id:id)
        let store=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(store.beginDraft(path:["features"],lease:store.lease()))
        XCTAssertFalse(store.canUndo)
        var draft=source.definitions[0];draft.title="Private features"
        XCTAssertTrue(store.updateDraft(draft,lease:store.lease(path:["features"])))
        XCTAssertFalse(store.canUndo)
        XCTAssertThrowsError(try store.executionSnapshot())
        XCTAssertTrue(store.applyDraft(lease:store.lease(path:["features"])))
        guard case .blockModel(let changed)=store.document else{return XCTFail("Expected nested source")}
        XCTAssertNotEqual(changed.source.root.instances[0].definition,source.root.instances[0].definition)
        XCTAssertEqual(changed.source.root.instances[1],source.root.instances[1])
        XCTAssertTrue(store.canUndo)
        XCTAssertTrue(store.undo(lease:store.lease()))
        guard case .blockModel(let undone)=store.document else{return XCTFail("Expected nested source")}
        XCTAssertEqual(undone.source,source)
        XCTAssertTrue(undone.drafts.isEmpty)
        XCTAssertTrue(store.redo(lease:store.lease()))
        guard case .blockModel(let redone)=store.document else{return XCTFail("Expected nested source")}
        XCTAssertEqual(redone.source,changed.source)
        let reopened=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertEqual(reopened.document,store.document)
        store.setExecutionBusy(true)
        XCTAssertFalse(store.undo(lease:store.lease()))
        XCTAssertFalse(store.beginDraft(path:["features"],lease:store.lease()))
    }
}
