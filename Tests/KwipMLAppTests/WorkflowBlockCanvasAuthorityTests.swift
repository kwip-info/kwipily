import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowBlockCanvasAuthorityTests:XCTestCase {
    private let rootAddress=WorkflowNodeAddress(domain:.model,localID:"features")
    @MainActor private func setup(nested:Bool=false)throws->(URL,ProjectLibrary,WorkflowAuthoringStore) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:directory)
        var source=CompositeExamples.parallelClassifier
        if nested {
            let pin=source.root.instances[0].definition
            let wrapper=CompositeDefinition(reference:.init("wrapper"),title:"Wrapper",body:.init(instances:[.init(id:"nested",definition:pin)]),inputs:[.init("in",targets:[.init("nested","in")])],outputs:[.init("out",source:.init("nested","out"))])
            source.definitions.append(wrapper);source.root.instances[0].definition=wrapper.reference
        }
        let model=ModelAuthoringDocument(projectID:UUID().uuidString,revision:5,source:source)
        var workflow=try WorkflowCodec.migrateModel(.blockModel(model),projectID:model.projectID).document
        workflow.nodes.append(.init(address:.init(domain:.data,localID:"features"),name:"Keep data",payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(";")])))
        workflow.nodes.append(.init(address:.init(domain:.control,localID:"setup"),payload:.runSetup(config:["seed":42])))
        _=try library.saveSource(.workflow(workflow))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:workflow.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        return(directory,library,owner)
    }
    @MainActor func testBorrowedEditorEditsOnlyCanonicalDraftAndCloseReopensItsSavedContents()throws {
        let(directory,library,owner)=try setup();defer{try? FileManager.default.removeItem(at:directory)}
        let original=owner.document
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:rootAddress)
        let session=ModelChildEditorSession(owner:authority)
        XCTAssertTrue(session.open(path:["features"]))
        let first=try XCTUnwrap(session.editor)
        XCTAssertEqual(first.primitiveRegistry,owner.modelRegistry)
        XCTAssertEqual(owner.document.nodes,original.nodes);XCTAssertEqual(owner.document.edges,original.edges)
        first.moveNodes(ids:["branch-a"],dx:19,dy:5)
        XCTAssertEqual(owner.document.drafts.first?.proposed.body.nodes.first{$0.id=="branch-a"}?.x,234)
        XCTAssertEqual(owner.document.revision,original.revision);XCTAssertFalse(owner.canUndo)
        XCTAssertEqual(try library.loadSource(id:owner.projectID),.workflow(owner.document))
        XCTAssertTrue(session.close());XCTAssertNil(session.editor)
        XCTAssertFalse(first.historyContextIsActive)
        XCTAssertTrue(session.open(path:["features"]))
        XCTAssertEqual(session.editor?.scope?.nodes.first{$0.id=="branch-a"}?.x,234)
        XCTAssertTrue(session.discard());XCTAssertEqual(owner.document,original)
    }
    @MainActor func testNestedApplyUsesParentHistoryAndPreservesEveryOtherDomain()throws {
        let(directory,_,owner)=try setup(nested:true);defer{try? FileManager.default.removeItem(at:directory)}
        let original=owner.document
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:rootAddress)
        let session=ModelChildEditorSession(owner:authority)
        XCTAssertTrue(session.open(path:["features","nested"]))
        let editor=try XCTUnwrap(session.editor)
        editor.moveNodes(ids:["branch-a"],dx:7,dy:0)
        XCTAssertTrue(session.apply());XCTAssertNil(session.editor)
        XCTAssertEqual(owner.document.revision,original.revision+1)
        XCTAssertEqual(owner.document.edges,original.edges)
        XCTAssertEqual(owner.document.nodes.filter{$0.address != rootAddress},original.nodes.filter{$0.address != rootAddress})
        XCTAssertEqual(owner.document.definitions.count,original.definitions.count+2)
        XCTAssertTrue(owner.canUndo);XCTAssertTrue(owner.undo(lease:owner.lease()))
        var expected=original;expected.revision=owner.document.revision
        XCTAssertEqual(owner.document,expected)
    }
    @MainActor func testStaleExactLeaseAndQualifiedRootCannotBeReinterpreted()throws {
        let(directory,_,owner)=try setup();defer{try? FileManager.default.removeItem(at:directory)}
        XCTAssertThrowsError(try WorkflowBlockCanvasAuthority(owner:owner,root:.init(domain:.data,localID:"features")))
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:rootAddress)
        XCTAssertThrowsError(try ModelCanvasAdapter(owner:authority,path:[]))
        let session=ModelChildEditorSession(owner:authority)
        XCTAssertTrue(session.open(path:["features"]))
        let editor=try XCTUnwrap(session.editor)
        editor.selectNode("branch-a",extending:false)
        let old=authority.lease(path:["features"])
        editor.moveNodes(ids:["branch-a"],dx:1,dy:0)
        let before=owner.document
        editor.renameSelected("Stale callback",lease:old)
        XCTAssertEqual(owner.document,before)
        XCTAssertFalse(session.open(path:["other-root"]))
        XCTAssertEqual(owner.document,before)
        owner.invalidate()
        editor.moveNodes(ids:["branch-a"],dx:1,dy:0)
        XCTAssertEqual(owner.document,before)
        XCTAssertFalse(session.apply())
    }
    @MainActor func testBoundaryTypingKeepsViewIdentityAndSaveFailureDoesNotPublish()throws {
        let(directory,library,owner)=try setup();defer{try? FileManager.default.removeItem(at:directory)}
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:rootAddress)
        let session=ModelChildEditorSession(owner:authority)
        XCTAssertTrue(session.open(path:["features"]))
        let editor=try XCTUnwrap(session.editor),identity=editor.boundaryRevision
        editor.updateBoundary(context:identity){$0.title="M"}
        editor.updateBoundary(context:identity){$0.title="My Block"}
        XCTAssertEqual(owner.document.drafts.first?.proposed.title,"My Block")
        XCTAssertEqual(editor.boundaryRevision,identity)
        let before=owner.document,url=try library.documentURL(id:owner.projectID),backup=directory.appendingPathComponent("backup.json")
        try FileManager.default.moveItem(at:url,to:backup)
        try FileManager.default.createSymbolicLink(at:url,withDestinationURL:backup)
        editor.moveNodes(ids:["branch-a"],dx:44,dy:0)
        XCTAssertEqual(owner.document,before);XCTAssertNotNil(authority.error)
        XCTAssertFalse(session.apply());XCTAssertTrue(session.editor === editor)
    }
    @MainActor func testPresentationFlushVetoAndConflictingDraftKeepTheBorrowedEditor()throws {
        let(directory,_,owner)=try setup(nested:true);defer{try? FileManager.default.removeItem(at:directory)}
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:rootAddress)
        let session=ModelChildEditorSession(owner:authority)
        var configured=0
        session.configurePresentation={_ in configured+=1}
        XCTAssertTrue(session.open(path:["features"]))
        let editor=try XCTUnwrap(session.editor)
        XCTAssertEqual(configured,1)
        XCTAssertFalse(session.open(path:["features","nested"]))
        XCTAssertTrue(session.editor === editor)
        session.persistPresentation={throw NSError(domain:"Disk full",code:1)}
        XCTAssertFalse(session.close());XCTAssertTrue(session.editor === editor)
        session.persistPresentation={}
        XCTAssertTrue(session.close());XCTAssertTrue(session.open(path:["features"]))
        XCTAssertEqual(configured,2)
        XCTAssertEqual(owner.document.drafts.first?.instancePath,["features"])
    }
    @MainActor func testLayerControlsPersistThroughChildApplyAndRejectDepartedSelection()throws {
        let(directory,_,owner)=try setup();defer{try? FileManager.default.removeItem(at:directory)}
        let authority=try WorkflowBlockCanvasAuthority(owner:owner,root:rootAddress)
        let session=ModelChildEditorSession(owner:authority)
        XCTAssertTrue(session.open(path:["features"]))
        let canvas=try XCTUnwrap(session.editor)
        canvas.selectNode("branch-a",extending:false)
        let node=try XCTUnwrap(canvas.graph.nodes.first {$0.id=="branch-a"})
        let editor=LayerParametersInspectorContext(store:canvas,node:node)
        XCTAssertTrue(editor.setAlgorithm("zeros",parameter:"weight"))
        XCTAssertTrue(editor.setMode("frozen",parameter:"bias"))
        XCTAssertNotNil(owner.document.drafts.first?.proposed.body.nodes.first {$0.id==node.id}?.parameterSettings)
        canvas.selectNode("branch-b",extending:false)
        canvas.selectNode("branch-a",extending:false)
        XCTAssertFalse(editor.setMode("frozen",parameter:"weight"))
        XCTAssertTrue(session.apply(),owner.error ?? "")
        XCTAssertEqual(owner.document.schemaVersion,15)
        XCTAssertTrue(owner.document.drafts.isEmpty)
        XCTAssertTrue(owner.document.definitions.contains {$0.body.nodes.contains {$0.id==node.id && $0.parameterSettings != nil}})
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(owner.document.definitions.contains {$0.body.nodes.contains {$0.parameterSettings != nil}})
    }

}
