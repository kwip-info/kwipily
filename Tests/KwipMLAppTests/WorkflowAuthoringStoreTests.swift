import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowAuthoringStoreTests:XCTestCase {
    private let data=WorkflowNodeAddress(domain:.data,localID:"shared")
    private let model=WorkflowNodeAddress(domain:.model,localID:"shared")
    @MainActor private func setup(revision:UInt64=10)throws->(URL,ProjectLibrary,WorkflowAuthoringStore) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),library=ProjectLibrary(directory:root)
        let doc=WorkflowDocument(projectID:UUID().uuidString,revision:revision,name:"Mixed workflow",nodes:[
            .init(address:data,name:"Observations",x:10,y:20,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:model,name:"Legacy Linear",x:30,y:40,payload:.modelPrimitive(type:"linear",configurationVersion:1,config:["in_features":.integer(2),"out_features":.integer(2)]))])
        _=try library.saveSource(.workflow(doc),id:doc.projectID)
        return(root,library,try makeStore(library:library,id:doc.projectID))
    }
    @MainActor private func makeStore(library:ProjectLibrary,id:String)throws->WorkflowAuthoringStore {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowAuthoringStore(library:library,projectID:id,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    @MainActor func testGuidedStageSavesNodeAndWiresAsOneUndoableEdit()throws {
        let(root,library,unused)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        unused.invalidate()
        let plan=try WorkflowGuidedBuildPlan.parallelClassifier()
        _=try library.saveSource(.workflow(plan.document(completedSteps:1)))
        let store=try makeStore(library:library,id:plan.completeDocument.projectID)
        let before=store.document
        XCTAssertNotNil(store.apply(.advanceGuidedBuild(expected:before.guidedBuild!),lease:store.lease()))
        XCTAssertEqual(store.document.nodes.count,2);XCTAssertEqual(store.document.edges.count,1)
        let inserted=store.document
        XCTAssertEqual(inserted.guidedBuild?.completedSteps,2)
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(inserted))
        XCTAssertTrue(store.undo(lease:store.lease()))
        XCTAssertEqual(store.document.nodes,before.nodes);XCTAssertEqual(store.document.edges,before.edges)
        XCTAssertEqual(store.document.guidedBuild,before.guidedBuild)
        XCTAssertFalse(store.canUndo)
        XCTAssertTrue(store.redo(lease:store.lease()))
        XCTAssertEqual(store.document.nodes,inserted.nodes);XCTAssertEqual(store.document.edges,inserted.edges)
        let accepted=store.document
        XCTAssertNil(store.apply(.advanceGuidedBuild(expected:before.guidedBuild!),lease:store.lease()))
        XCTAssertEqual(store.document,accepted)
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(accepted))
    }
    @MainActor func testMixedCommandPersistsWholeSourceAndHistoryRestoresSelectionWithMonotonicRevisions()throws {
        let(root,library,store)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let original=store.document
        XCTAssertTrue(store.select(.init(nodeAddresses:[data,model]),lease:store.lease()))
        let selection=store.selection
        XCTAssertNotNil(store.apply(.move(addresses:[data,model],offset:.init(x:25,y:5)),lease:store.lease()))
        XCTAssertEqual(store.document.revision,11)
        XCTAssertEqual(store.document.nodes.map(\.x),[35,55])
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(store.document))
        XCTAssertTrue(store.select(.init(nodeAddresses:[model]),lease:store.lease()))
        XCTAssertNotNil(store.apply(.rename(address:model,name:"Edited model"),lease:store.lease()))
        XCTAssertEqual(store.document.revision,12)
        XCTAssertTrue(store.undo(lease:store.lease()));XCTAssertEqual(store.document.revision,13)
        XCTAssertEqual(store.document.nodes[1].name,original.nodes[1].name)
        XCTAssertEqual(store.selection.nodeAddresses,[model])
        XCTAssertTrue(store.undo(lease:store.lease()));XCTAssertEqual(store.document.revision,14)
        XCTAssertEqual(store.document.nodes,original.nodes);XCTAssertEqual(store.selection,selection)
        XCTAssertTrue(store.redo(lease:store.lease()));XCTAssertEqual(store.document.revision,15)
        XCTAssertEqual(store.document.nodes.map(\.x),[35,55])
        if case .modelPrimitive(_,let version,let config)=store.document.nodes[1].payload {XCTAssertEqual(version,1);XCTAssertNil(config["bias"])} else {XCTFail("Model payload changed")}
        XCTAssertEqual(try library.loadSource(id:store.projectID),.workflow(store.document))
    }
    @MainActor func testNoOpSelectionAndRedoBranchDoNotInventHistoryOrInvalidateLease()throws {
        let(root,library,store)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let url=try library.documentURL(id:store.projectID),bytes=try Data(contentsOf:url),initial=store.lease()
        XCTAssertTrue(store.select(.init(nodeAddresses:[data]),lease:initial))
        XCTAssertEqual(store.lease(),initial);XCTAssertFalse(store.canUndo)
        XCTAssertNotNil(store.apply(.rename(address:data,name:"Observations"),lease:initial))
        XCTAssertEqual(store.lease(),initial);XCTAssertEqual(try Data(contentsOf:url),bytes)
        XCTAssertEqual(store.selection.nodeAddresses,[data]);XCTAssertFalse(store.canUndo)
        XCTAssertNotNil(store.apply(.rename(address:data,name:"First"),lease:store.lease()))
        XCTAssertTrue(store.undo(lease:store.lease()));XCTAssertTrue(store.canRedo)
        let undoLease=store.lease()
        XCTAssertNotNil(store.apply(.move(addresses:[],offset:.init()),lease:undoLease))
        XCTAssertEqual(store.lease(),undoLease);XCTAssertTrue(store.canRedo)
        XCTAssertNotNil(store.apply(.rename(address:model,name:"Different branch"),lease:store.lease()))
        XCTAssertFalse(store.canRedo)
    }
    @MainActor func testStaleReopenedBusyAndInvalidatedLeasesCannotMutate()throws {
        let(root,library,store)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let old=store.lease()
        XCTAssertNotNil(store.apply(.rename(address:data,name:"Next"),lease:old))
        XCTAssertFalse(store.isCurrent(old));XCTAssertNil(store.apply(.rename(address:model,name:"Stale"),lease:old))
        let current=store.lease(),before=store.document
        store.setExecutionBusy(true)
        XCTAssertTrue(store.isCurrent(current),"Captured execution callbacks may remain current while busy")
        XCTAssertNil(store.apply(.delete(.init(nodeAddresses:[data])),lease:current));XCTAssertFalse(store.undo(lease:current))
        XCTAssertTrue(store.select(.init(nodeAddresses:[model]),lease:current),"Read-only selection remains available while authoring is locked")
        store.setExecutionBusy(false);XCTAssertEqual(store.document,before)
        let reopened=try makeStore(library:library,id:store.projectID)
        XCTAssertNotEqual(reopened.sessionID,store.sessionID);XCTAssertFalse(reopened.isCurrent(current))
        XCTAssertNil(reopened.apply(.rename(address:data,name:"Wrong session"),lease:current))
        XCTAssertFalse(reopened.canUndo);XCTAssertTrue(reopened.selection.nodeAddresses.isEmpty)
        store.invalidate();XCTAssertFalse(store.isCurrent(current));XCTAssertFalse(store.canUndo)
        store.setExecutionBusy(false);XCTAssertNil(store.apply(.rename(address:data,name:"Retired"),lease:store.lease()))
        XCTAssertEqual(store.document,before)
    }
    @MainActor func testFailedSavePreservesDocumentSelectionHistoryAndLease()throws {
        let(root,library,store)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        XCTAssertTrue(store.select(.init(nodeAddresses:[data,model]),lease:store.lease()))
        XCTAssertNotNil(store.apply(.rename(address:model,name:"Changed"),lease:store.lease()))
        XCTAssertTrue(store.undo(lease:store.lease()));XCTAssertTrue(store.canRedo)
        let before=store.document,selection=store.selection,lease=store.lease(),url=try library.documentURL(id:store.projectID)
        let saved=try Data(contentsOf:url)
        try FileManager.default.removeItem(at:url);try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
        XCTAssertNil(store.apply(.delete(.init(nodeAddresses:[data])),lease:lease))
        XCTAssertNotNil(store.error);XCTAssertEqual(store.document,before);XCTAssertEqual(store.selection,selection);XCTAssertEqual(store.lease(),lease)
        XCTAssertTrue(store.canRedo);XCTAssertFalse(store.redo(lease:lease))
        XCTAssertEqual(store.document,before);XCTAssertEqual(store.lease(),lease);XCTAssertTrue(store.canRedo)
        try FileManager.default.removeItem(at:url);try saved.write(to:url)
        XCTAssertTrue(store.redo(lease:lease));XCTAssertEqual(store.document.nodes[1].name,"Changed")
    }
    @MainActor func testRevisionOverflowAndBoundedHistory()throws {
        let(root,library,store)=try setup(revision:UInt64.max-1);defer{try? FileManager.default.removeItem(at:root)}
        XCTAssertNotNil(store.apply(.rename(address:data,name:"Last revision"),lease:store.lease()))
        XCTAssertEqual(store.document.revision,UInt64.max)
        let lease=store.lease(),before=store.document,bytes=try Data(contentsOf:library.documentURL(id:store.projectID))
        XCTAssertNotNil(store.apply(.rename(address:data,name:"Last revision"),lease:lease),"No-op does not consume a revision")
        XCTAssertEqual(store.lease(),lease)
        XCTAssertFalse(store.undo(lease:lease));XCTAssertNil(store.apply(.rename(address:data,name:"Overflow"),lease:lease))
        XCTAssertEqual(store.document,before);XCTAssertEqual(store.lease(),lease);XCTAssertFalse(store.canUndo,"No further undo can persist after revision exhaustion")
        XCTAssertEqual(try Data(contentsOf:library.documentURL(id:store.projectID)),bytes)
        let(other,_,bounded)=try setup(revision:0);defer{try? FileManager.default.removeItem(at:other)}
        for index in 1...105 {XCTAssertNotNil(bounded.apply(.rename(address:data,name:"Edit \(index)"),lease:bounded.lease()))}
        for _ in 0..<100 {XCTAssertTrue(bounded.undo(lease:bounded.lease()))}
        XCTAssertFalse(bounded.canUndo);XCTAssertEqual(bounded.document.nodes[0].name,"Edit 5")
        XCTAssertEqual(bounded.document.revision,205)
    }
    @MainActor func testInitRejectsLegacyAndBlockFormatsWithoutConversion()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let flat=try library.saveSource(.legacyFlat(ExampleFlows.parallelClassifier))
        let blockID=UUID().uuidString
        _=try library.saveSource(.blockModel(.init(projectID:blockID,source:CompositeExamples.parallelClassifier)),id:blockID)
        let flatURL=try library.documentURL(id:flat.id),before=try Data(contentsOf:flatURL)
        XCTAssertThrowsError(try makeStore(library:library,id:flat.id)) {XCTAssertTrue($0.localizedDescription.contains("format"))}
        XCTAssertThrowsError(try makeStore(library:library,id:blockID)) {XCTAssertTrue($0.localizedDescription.contains("format"))}
        XCTAssertEqual(try Data(contentsOf:flatURL),before)
    }
}
