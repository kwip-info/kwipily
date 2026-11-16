import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowInspectorSessionTests:XCTestCase {
    private let setup=WorkflowNodeAddress(domain:.control,localID:"setup")
    private let other=WorkflowNodeAddress(domain:.control,localID:"other")
    private func fixture()throws->(WorkflowAuthoringStore,WorkflowInspectorSession) {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:setup,name:"Training budget",payload:.runSetup(config:["epochs":12])),
            .init(address:other,name:"Other settings",payload:.runSetup(config:["seed":8]))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[setup]),lease:owner.lease()))
        return(owner,WorkflowInspectorSession(owner:owner,sources:InspectorSourceProbe(owner:owner)))
    }
    private func control(_ session:WorkflowInspectorSession)throws->WorkflowControlNodeContext {
        guard case .control(let context)=session.selection else {throw DataFlowContractError("Expected selected control inspector.")}
        return context
    }
    private func eventually(_ condition:@MainActor()->Bool)async->Bool {
        let end=ContinuousClock.now.advanced(by:.seconds(2))
        while !condition(),ContinuousClock.now<end {try? await Task.sleep(for:.milliseconds(1))}
        return condition()
    }
    func testPositionOnlyMoveKeepsInspectorDraftIdentityButConfigEditRetiresIt() throws {
        let(owner,session)=try fixture();_=session.activate()
        let original=try control(session)
        let origin=original.numericOrigin(key:"epochs")
        original.setNumericPending(origin:origin,key:"epochs",pending:true)
        XCTAssertNotNil(owner.apply(.move(addresses:[setup],offset:.init(x:20,y:10)),lease:owner.lease()))
        session.reconcile()
        XCTAssertTrue(original.isCurrent)
        XCTAssertEqual(try control(session).contextIdentity,original.contextIdentity)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        original.setNumericPending(origin:origin,key:"epochs",pending:false)
        XCTAssertNotNil(owner.apply(.configure(address:setup,config:["epochs":30]),lease:owner.lease()))
        session.reconcile()
        XCTAssertFalse(original.isCurrent)
    }
    func testOwnPolicyCommitAndBusyCycleKeepInspectorAndPendingFieldIdentity()async throws {
        let(owner,session)=try fixture();_=session.activate()
        let context=try control(session),identity=context.contextIdentity
        XCTAssertTrue(context.setPreparationCachePolicy(.materialize,expected:context.configuration))
        let origin=context.numericOrigin(key:"epochs")
        context.setNumericPending(origin:origin,key:"epochs",pending:true)
        owner.setExecutionBusy(true);owner.setExecutionBusy(false)
        await Task.yield();session.reconcile()
        XCTAssertEqual(try control(session).contextIdentity,identity)
        XCTAssertEqual(context.preparationCachePolicy,.materialize)
        XCTAssertEqual(context.authoredName,"Training budget")
        XCTAssertTrue(context.isCurrent);XCTAssertTrue(owner.hasPendingInspectorEdits)
        context.setNumericPending(origin:origin,key:"epochs",pending:false)
    }
    func testUndoRedoAndSelectionDepartureReplaceContextWithoutExtraClick()async throws {
        let(owner,session)=try fixture();_=session.activate()
        let original=try control(session)
        XCTAssertTrue(original.setPreparationCachePolicy(.materialize,expected:original.configuration))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        let renewed=await eventually {session.selection?.isCurrent == true && session.selection?.identity != original.contextIdentity}
        XCTAssertTrue(renewed)
        let undone=try control(session)
        XCTAssertEqual(undone.preparationCachePolicy,.off)
        XCTAssertTrue(owner.redo(lease:owner.lease()))
        let redone=await eventually {session.selection?.isCurrent == true && session.selection?.identity != undone.contextIdentity}
        XCTAssertTrue(redone);XCTAssertEqual(try control(session).preparationCachePolicy,.materialize)
        XCTAssertTrue(owner.select(.init(nodeAddresses:[other]),lease:owner.lease()))
        let moved=await eventually {session.selection?.address == self.other && session.selection?.isCurrent == true}
        XCTAssertTrue(moved);XCTAssertEqual(try control(session).authoredName,"Other settings")
    }
    func testTrueExitClearsPendingAndReturnRecreatesSameSelectedNode()throws {
        let(owner,session)=try fixture(),appearance=session.activate()
        let original=try control(session),origin=original.numericOrigin(key:"epochs")
        original.setNumericPending(origin:origin,key:"epochs",pending:true)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        session.deactivate(appearance:appearance)
        XCTAssertFalse(owner.hasPendingInspectorEdits);XCTAssertNil(session.selection)
        XCTAssertFalse(original.isCurrent);XCTAssertFalse(session.isVisible)
        _=session.activate()
        let returned=try control(session)
        XCTAssertTrue(returned.isCurrent);XCTAssertEqual(returned.authoredName,"Training budget")
        XCTAssertNotEqual(returned.contextIdentity,original.contextIdentity)
    }
    func testRetiredVisibleContextAutomaticallyRecoversButHiddenOrInvalidOwnerDoesNot()async throws {
        let(owner,session)=try fixture(),appearance=session.activate()
        let original=try control(session);original.retire()
        let restored=await eventually {session.selection?.isCurrent == true && session.selection?.identity != original.contextIdentity}
        XCTAssertTrue(restored)
        session.deactivate(appearance:appearance)
        XCTAssertTrue(owner.rename("Changed while hidden",lease:owner.lease()))
        await Task.yield();session.reconcile()
        XCTAssertNil(session.selection);XCTAssertFalse(session.isVisible)
        _=session.activate();owner.invalidate();session.reconcile()
        XCTAssertNil(session.selection,"An invalidated owner must not acquire a fresh editing context.")
    }
    func testForeignSourceSessionNeverConstructsAnInspector()throws {
        let(owner,_)=try fixture(),(foreign,_)=try fixture()
        let session=WorkflowInspectorSession(owner:owner,sources:InspectorSourceProbe(owner:foreign))
        _=session.activate()
        XCTAssertNil(session.selection);XCTAssertNotNil(session.error)
        XCTAssertFalse(owner.hasPendingInspectorEdits)
    }
    func testAppearanceTaskCancellationCannotDetachReplacementAppearance()async throws {
        let(owner,session)=try fixture()
        let first=Task {@MainActor in await session.runAppearanceLifetime()}
        let appeared=await eventually {session.isVisible && session.selection?.isCurrent == true}
        XCTAssertTrue(appeared)
        // Explicit activation models the replacement task starting before the
        // outgoing task receives cancellation. Its token must remain independent.
        let replacement=session.activate()
        let current=try control(session),origin=current.numericOrigin(key:"epochs")
        current.setNumericPending(origin:origin,key:"epochs",pending:true)
        first.cancel();await first.value
        XCTAssertTrue(session.isVisible);XCTAssertTrue(current.isCurrent)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        session.deactivate(appearance:replacement)
        XCTAssertFalse(session.isVisible);XCTAssertFalse(owner.hasPendingInspectorEdits)
        let returning=Task {@MainActor in await session.runAppearanceLifetime()}
        let returned=await eventually {session.isVisible && session.selection?.isCurrent == true}
        XCTAssertTrue(returned)
        returning.cancel();await returning.value
        XCTAssertFalse(session.isVisible);XCTAssertNil(session.selection)
    }
}

@MainActor private final class InspectorSourceProbe:WorkflowDataSources {
    let projectID:String
    let sessionID:UUID
    var isBusy=false
    var statusMessage="Test source"
    var changes:AnyPublisher<Void,Never> {Empty(completeImmediately:false).eraseToAnyPublisher()}
    init(owner:WorkflowAuthoringStore) {projectID=owner.projectID;sessionID=owner.sessionID}
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument)->[DataColumnSchema] {[]}
    func bindingDisplayName(for address:WorkflowNodeAddress)->String? {nil}
    func chooseSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
    func inspectSource(for address:WorkflowNodeAddress,lease:WorkflowEditorLease)->Bool {false}
}
