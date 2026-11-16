import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasVersionHistoryTests:XCTestCase {
    func testDiffsReconstructExactSourceAfterReopenAndNoOpSave() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var source=WorkflowDocument(projectID:UUID().uuidString,name:"Before")
        let journal=try CanvasVersionJournal(root:root,projectID:source.projectID)
        XCTAssertEqual(try journal.save(source).number,1)
        source.name="After";source.revision=1
        let next=try journal.save(source)
        XCTAssertEqual(next.number,2)
        XCTAssertFalse(next.changes.isEmpty)
        XCTAssertEqual(try journal.save(source).number,2)
        let reopened=try CanvasVersionJournal(root:root,projectID:source.projectID)
        XCTAssertEqual(try reopened.document(version:next.id),source)
        XCTAssertEqual(reopened.versions.count,2)
        XCTAssertNotNil(reopened.versions.last?.base)
        XCTAssertNil(reopened.versions.first?.base)
    }
    func testCorruptDeltaAndSymlinkRefuseWithoutReplacingJournal()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=WorkflowDocument(projectID:UUID().uuidString)
        let journal=try CanvasVersionJournal(root:root,projectID:source.projectID)
        _=try journal.save(source)
        let url=journal.fileURL
        var raw=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
        var records=raw["records"] as! [[String:Any]];records[0]["hash"]=String(repeating:"0",count:64);raw["records"]=records
        try JSONSerialization.data(withJSONObject:raw).write(to:url)
        XCTAssertThrowsError(try CanvasVersionJournal(root:root,projectID:source.projectID))
        try FileManager.default.removeItem(at:url)
        let elsewhere=root.appendingPathComponent("unrelated");try Data("untouched".utf8).write(to:elsewhere)
        try FileManager.default.createSymbolicLink(at:url,withDestinationURL:elsewhere)
        XCTAssertThrowsError(try CanvasVersionJournal(root:root,projectID:source.projectID))
        XCTAssertEqual(try String(contentsOf:elsewhere,encoding:.utf8),"untouched")
    }
    func testNamedConfigurationAndMoveChangesAreSeparate()throws {
        let address=WorkflowNodeAddress(domain:.model,localID:"layer")
        let before=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Projection",x:0,y:0,payload:.modelPrimitive(type:"linear",configurationVersion:1,config:["in_features":.integer(2),"out_features":.integer(3)]))])
        var after=before;after.nodes[0].x=20;after.nodes[0].payload = .modelPrimitive(type:"linear",configurationVersion:1,config:["in_features":.integer(2),"out_features":.integer(4)])
        let rows=CanvasVersionJournal.changes(from:before,to:after)
        XCTAssertTrue(rows.contains {$0.component=="Projection" && $0.kind == .layout})
        XCTAssertTrue(rows.contains {$0.component=="Projection" && $0.kind == .configuration})
        after.revision=99
        XCTAssertEqual(rows,CanvasVersionJournal.changes(from:before,to:after))
    }
    func testIndependentWritersRefuseLostUpdateAndRevisionOnlySavesAreNoOps()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var source=WorkflowDocument(projectID:UUID().uuidString,name:"Original")
        let a=try CanvasVersionJournal(root:root,projectID:source.projectID)
        _=try a.save(source)
        let b=try CanvasVersionJournal(root:root,projectID:source.projectID)
        source.name="A";_ = try a.save(source)
        source.name="B";XCTAssertThrowsError(try b.save(source))
        try b.reload();_ = try b.save(source)
        source.revision=999
        XCTAssertEqual(try b.save(source).number,3)
        XCTAssertEqual(try a.document(version:a.versions.first!.id).name,"A")
    }
    @MainActor func testRestorePreservesUnsavedCanvasAndRefusesStaleOrBusyAuthority()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:root)
        defer {try? FileManager.default.removeItem(at:root)}
        var source=WorkflowDocument(projectID:UUID().uuidString,name:"Original");source.schemaVersion=18
        _ = try library.saveSource(.workflow(source),id:source.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        try owner.configureCanvasVersions(root:root)
        let first=owner.canvasVersions!.versions[0].id
        let stale=owner.lease()
        XCTAssertTrue(owner.rename("Saved later",lease:owner.lease()))
        try owner.saveCanvasVersion()
        XCTAssertTrue(owner.rename("Unsaved work",lease:owner.lease()))
        XCTAssertFalse(owner.restoreCanvasVersion(first,lease:stale))
        owner.setExecutionBusy(true)
        XCTAssertFalse(owner.restoreCanvasVersion(first,lease:owner.lease()))
        owner.setExecutionBusy(false)
        XCTAssertEqual(owner.document.name,"Unsaved work")
        XCTAssertTrue(owner.restoreCanvasVersion(first,lease:owner.lease()))
        XCTAssertEqual(owner.document.name,"Original")
        let journal=owner.canvasVersions!
        XCTAssertEqual(journal.versions.count,4)
        XCTAssertEqual(try journal.document(version:journal.versions[1].id).name,"Unsaved work")
        XCTAssertEqual(try library.loadSource(id:source.projectID),.workflow(owner.document))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.name,"Unsaved work")
    }

    @MainActor func testFileBackedRestoreUsesSavedChildPinsRatherThanCurrentHead()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString),library=ProjectLibrary(directory:root)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=try RetiredToolAuthoringUpgrade.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        _ = try library.saveSource(.workflow(source));_ = try library.activateFileBackedWorkflow(id:source.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        try owner.configureCanvasVersions(root:root)
        let first=owner.canvasVersions!.versions[0].id,before=owner.document
        let pin=try XCTUnwrap(owner.pinnedFlowDocument?.dependencies.first),url=try XCTUnwrap(owner.flowRootURL)
        let childURL=url.deletingLastPathComponent().appendingPathComponent(pin.relativePath)
        var child=try FlowDocumentStore.readHead(at:childURL)
        child.revision+=1
        if case .modelBlock(var definition)=child.payload {definition.title="New child head";child.payload = .modelBlock(definition)}
        try FlowDocumentStore.save(child,at:childURL,expected:pin.revision)
        let proposal=try owner.prepareFlowAdoption(definition:pin.definition,lease:owner.lease())
        XCTAssertTrue(owner.adoptFlowRevision(proposal,lease:owner.lease()))
        try owner.saveCanvasVersion()
        XCTAssertNotEqual(owner.pinnedFlowDocument?.dependencies.first?.revision,pin.revision)
        XCTAssertTrue(owner.restoreCanvasVersion(first,lease:owner.lease()),owner.error ?? "")
        XCTAssertEqual(owner.document.definitions,before.definitions)
        XCTAssertEqual(owner.pinnedFlowDocument?.dependencies.first,pin)
        XCTAssertEqual(try FlowDocumentStore.readHead(at:childURL),child,"Restoring parent must not rewrite newer child head")
    }

    func testDisplayLabelChangesAreRecomputedWithoutInvalidatingExactSourceDeltas()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var source=WorkflowDocument(projectID:UUID().uuidString,name:"Before")
        let journal=try CanvasVersionJournal(root:root,projectID:source.projectID)
        _ = try journal.save(source);source.name="After";let version=try journal.save(source)
        var raw=try JSONSerialization.jsonObject(with:Data(contentsOf:journal.fileURL)) as! [String:Any]
        var records=raw["records"] as! [[String:Any]],changes=records[1]["changes"] as! [[String:Any]]
        changes[0]["field"]="Old internal label";changes[0]["after"]="Misleading presentation"
        records[1]["changes"]=changes;raw["records"]=records
        try JSONSerialization.data(withJSONObject:raw).write(to:journal.fileURL)
        let reopened=try CanvasVersionJournal(root:root,projectID:source.projectID)
        XCTAssertEqual(try reopened.document(version:version.id),source)
        XCTAssertEqual(reopened.versions[0].changes[0].field,"Name")
        XCTAssertEqual(reopened.versions[0].changes[0].after,"After")
    }

}
