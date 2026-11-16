import XCTest
@testable import KwipMLApp
import KwipMLCore
final class MCPBatchEditingTests:XCTestCase {
    @MainActor func fixture()throws->WorkflowAuthoringStore {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("mcp-batch-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock{try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let project=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:.init(domain:.data,localID:"view"),payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))])
        _=try library.saveSource(.workflow(project))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowAuthoringStore(library:library,projectID:project.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    @MainActor func testAtomicFailureDoesNotPersistFirstOperation()throws {
        let owner=try fixture(),before=owner.document,address=before.nodes[0].address
        XCTAssertNil(owner.applyBatch([.rename(address:address,name:"Changed"),.connect(from:.init(address:address,port:"table"),to:.init(address:address,port:"table"))],lease:owner.lease()))
        XCTAssertEqual(owner.document,before);XCTAssertFalse(owner.canUndo)
    }
    @MainActor func testBatchIsOneUndoAndRejectsOldLease()throws {
        let owner=try fixture(),before=owner.document,address=before.nodes[0].address,lease=owner.lease()
        XCTAssertNotNil(owner.applyBatch([.rename(address:address,name:"Named view"),.move(addresses:[address],offset:.init(x:30,y:20))],lease:lease))
        XCTAssertEqual(owner.document.revision,before.revision+1);XCTAssertEqual(owner.document.nodes[0].x,30)
        XCTAssertNil(owner.applyBatch([.rename(address:address,name:"Stale")],lease:lease))
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,before.nodes);XCTAssertFalse(owner.canUndo)
    }
    @MainActor func testBatchUsesStagedContainerOwnershipAndOneUndo()throws {
        let owner=try fixture(),canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer{canvas.invalidate();owner.invalidate()}
        let entry=try XCTUnwrap(canvas.creationEntries.first{$0.component.title=="Parallel features"})
        let definition=try XCTUnwrap(entry.definition),root=WorkflowNodeAddress(domain:.model,localID:"features")
        let inserted=owner.apply(.insertBlock(.init(address:root,payload:entry.payload),definition:definition),lease:owner.lease())
        _=try XCTUnwrap(inserted,owner.error ?? "")
        let child=try XCTUnwrap(definition.body.nodes.first?.id)
        XCTAssertNotNil(owner.apply(.inlineContainer(root:root,command:.rename(childID:child,name:"Primed")),lease:owner.lease()))
        let before=owner.document
        XCTAssertNotNil(owner.applyBatch([.inlineContainer(root:root,command:.rename(childID:child,name:"Second")),.inlineContainer(root:root,command:.rename(childID:child,name:"Final"))],lease:owner.lease()),owner.error ?? "")
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:owner.document,root:root).body.nodes.first?.name,"Final")
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,before.nodes)
        XCTAssertEqual(owner.document.definitions,before.definitions)
    }

}
