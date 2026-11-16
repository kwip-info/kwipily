import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowChildWorkspaceTests:XCTestCase {
    private let address=WorkflowNodeAddress(domain:.model,localID:"features")
    private func make(nested:Bool=false)throws->(WorkspaceStore,WorkflowWorkspaceSession,WorkflowChildWorkspace) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:directory,detectCapabilities:false)
        var source=CompositeExamples.parallelClassifier
        if nested {
            let pin=source.root.instances[0].definition
            let wrapper=CompositeDefinition(reference:.init("wrapper"),title:"Wrapper",body:.init(instances:[.init(id:"nested",definition:pin)]),inputs:[.init("in",targets:[.init("nested","in")])],outputs:[.init("out",source:.init("nested","out"))])
            source.definitions.append(wrapper);source.root.instances[0].definition=wrapper.reference
        }
        let original=ModelAuthoringDocument(projectID:UUID().uuidString,source:source)
        var document=try WorkflowCodec.migrateModel(.blockModel(original),projectID:original.projectID).document
        document.nodes.append(.init(address:.init(domain:.data,localID:"features"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(";")])))
        document.nodes.append(.init(address:.init(domain:.control,localID:"setup"),payload:.runSetup(config:["seed":42])))
        _=try workspace.projectLibrary.saveSource(.workflow(document))
        workspace.activeProjectID=document.projectID
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner);workspace.workflowSession=host
        let children=WorkflowChildWorkspace(workspace:workspace,host:host)
        children.present={session,path in session.open(path:path)}
        return(workspace,host,children)
    }
    func testQualifiedLeaseGuardReuseAndCloseReopenPersistedDraft()throws {
        let(workspace,host,children)=try make()
        defer {withExtendedLifetime(workspace) {}}
        let old=host.owner.lease()
        XCTAssertFalse(children.open(address:.init(domain:.data,localID:"features"),lease:old))
        XCTAssertTrue(children.open(address:address,lease:old))
        let session=try XCTUnwrap(children.session),editor=try XCTUnwrap(session.editor)
        XCTAssertFalse(children.open(address:address,lease:old),"Opening the draft advanced the owner lease; the supplied old lease cannot be substituted.")
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease()))
        XCTAssertTrue(children.session === session);XCTAssertTrue(session.editor === editor)
        editor.moveNodes(ids:["branch-a"],dx:9,dy:0)
        let draft=host.owner.document.drafts
        XCTAssertTrue(children.close());XCTAssertNil(session.editor)
        XCTAssertEqual(host.owner.document.drafts,draft)
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease()))
        XCTAssertEqual(children.session?.editor?.scope?.nodes.first {$0.id == "branch-a"}?.x,224)
    }
    func testNestedDraftReopensFromRootThenApplyAndUndoPreserveOtherDomains()throws {
        let(workspace,host,children)=try make(nested:true)
        defer {withExtendedLifetime(workspace) {}}
        let original=host.owner.document
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease(),nestedPath:["nested"]))
        children.session?.editor?.moveNodes(ids:["branch-a"],dx:7,dy:0)
        XCTAssertTrue(children.close())
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease()))
        XCTAssertEqual(children.session?.editor?.path,["features","nested"])
        XCTAssertTrue(try XCTUnwrap(children.session).apply())
        XCTAssertEqual(host.owner.document.nodes.filter {$0.address.domain != .model},original.nodes.filter {$0.address.domain != .model})
        XCTAssertEqual(host.owner.document.edges,original.edges)
        XCTAssertTrue(host.owner.undo(lease:host.owner.lease()))
        var expected=original;expected.revision=host.owner.document.revision
        XCTAssertEqual(host.owner.document,expected)
    }
    func testPresentationRestoresWithoutSourceEditAndStaleCallbackCannotCrossSession()throws {
        let(workspace,host,children)=try make()
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease()))
        let editor=try XCTUnwrap(children.session?.editor)
        let document=host.owner.document
        let state=CanvasViewState(zoom:0.34,scrollX:80,scrollY:90,selectedNodeIDs:["branch-a"],presentation:.init(flowDirection:.topToBottom,nodeDetail:.large))
        editor.restoreCanvasViewState(state)
        let late=editor.onCanvasViewChange
        XCTAssertTrue(children.close());XCTAssertEqual(host.owner.document,document)
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease()))
        XCTAssertEqual(children.session?.editor?.capturedCanvasViewState(),state)
        let saved=workspace.blockCanvasStates
        children.invalidate()
        late?(.init(zoom:1.4))
        XCTAssertEqual(workspace.blockCanvasStates,saved)
        XCTAssertFalse(children.open(address:address,lease:host.owner.lease()))
        XCTAssertTrue(host.owner.isCurrent(host.owner.lease()),"Invalidating a borrowed child does not retire its parent.")
    }
    func testFailedFlushVetoKeepsBorrowedEditorAndDraft()throws {
        let(workspace,host,children)=try make()
        XCTAssertTrue(children.open(address:address,lease:host.owner.lease()))
        XCTAssertTrue(children.flushPresentation())
        let editor=try XCTUnwrap(children.session?.editor),draft=host.owner.document.drafts
        let file=workspace.workspaceDirectory.appendingPathComponent("workspace-arrangements/projects/\(host.owner.projectID).json")
        try FileManager.default.removeItem(at:file);try FileManager.default.createDirectory(at:file,withIntermediateDirectories:false)
        XCTAssertFalse(children.close());XCTAssertTrue(children.session?.editor === editor)
        XCTAssertNotNil(children.error);XCTAssertEqual(host.owner.document.drafts,draft)
        XCTAssertFalse(children.flushPresentation())
    }
}
