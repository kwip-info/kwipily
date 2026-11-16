import XCTest
import AppKit
import Combine
import KwipMLCore
@testable import KwipMLApp

final class EditorHistoryRoutingTests: XCTestCase {
    @MainActor func testDataHistoryUndoRedoCannotTouchModelAndRejectsProjectChange() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.beginProject(ExampleFlows.parallelClassifier)
        let data=DataFlowWorkspaceStore(workspace:workspace)
        let before=data.project;let model=workspace.graph
        data.moveNodes(ids:["cast"],dx:30,dy:20)
        let moved=data.project;let context=EditorHistoryContext(data)
        XCTAssertTrue(EditorHistoryDispatch.canPerform(.undo,context:context,text:nil))
        EditorHistoryDispatch.perform(.undo,context:context,text:nil)
        XCTAssertEqual(data.project,before);XCTAssertEqual(workspace.graph,model)
        EditorHistoryDispatch.perform(.redo,context:context,text:nil)
        XCTAssertEqual(data.project,moved)
        workspace.isPreparingData=true
        XCTAssertFalse(EditorHistoryDispatch.canPerform(.undo,context:context,text:nil))
        workspace.isPreparingData=false
        workspace.beginProject(GraphDocument(name:"Other"))
        EditorHistoryDispatch.perform(.undo,context:context,text:nil)
        XCTAssertEqual(data.project,moved)
        XCTAssertFalse(EditorHistoryDispatch.canPerform(.undo,context:nil,text:nil))
    }
    @MainActor func testModelAndCompositeUseIndependentHistories() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.beginProject(ExampleFlows.parallelClassifier)
        let model=workspace.graph
        workspace.moveNodes(ids:["input"],dx:25,dy:0)
        EditorHistoryDispatch.perform(.undo,context:EditorHistoryContext(workspace),text:nil)
        XCTAssertEqual(workspace.graph,model)
        let composite=CompositeWorkspaceStore(workspace:workspace)
        let before=composite.state
        let id=try XCTUnwrap(composite.graph.nodes.first?.id)
        composite.moveNodes(ids:[id],dx:40,dy:15)
        let moved=composite.state
        XCTAssertNotEqual(moved,before)
        let context=EditorHistoryContext(composite)
        EditorHistoryDispatch.perform(.undo,context:context,text:nil)
        XCTAssertEqual(composite.state,before)
        EditorHistoryDispatch.perform(.redo,context:EditorHistoryContext(composite),text:nil)
        XCTAssertEqual(composite.state,moved);XCTAssertEqual(workspace.graph,model)
        let instance=try XCTUnwrap(composite.state.project.root.instances.first?.id)
        composite.openDefinition(instanceID:instance)
        let entered=composite.state
        EditorHistoryDispatch.perform(.undo,context:context,text:nil)
        XCTAssertEqual(composite.state,entered,"A queued root command cannot edit a different composite context")
    }
    @MainActor func testNativeTextHistoryTakesPriorityAndNeverFallsThroughWhenEmpty() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        workspace.activeProjectID=UUID().uuidString
        workspace.moveNodes(ids:["input"],dx:25,dy:0)
        let model=workspace.graph,context=EditorHistoryContext(workspace)
        let text=HistoryTextView();text.allowsUndo=true
        text.string="old"
        text.insertText("new",replacementRange:NSRange(location:0,length:3))
        text.breakUndoCoalescing()
        let target=EditorTextHistory(manager:text.undoManager)
        XCTAssertTrue(EditorHistoryDispatch.canPerform(.undo,context:context,text:target))
        EditorHistoryDispatch.perform(.undo,context:context,text:target)
        XCTAssertEqual(text.string,"old");XCTAssertEqual(workspace.graph,model)
        EditorHistoryDispatch.perform(.redo,context:context,text:target)
        XCTAssertEqual(text.string,"new");XCTAssertEqual(workspace.graph,model)
        text.undoManager?.removeAllActions()
        XCTAssertTrue(context.canPerform(.undo), "There is a graph edit that an incorrect fallback would consume")
        XCTAssertFalse(EditorHistoryDispatch.canPerform(.undo,context:context,text:target))
        EditorHistoryDispatch.perform(.undo,context:context,text:target)
        XCTAssertEqual(workspace.graph,model)
        EditorHistoryDispatch.perform(.undo,context:context,text:EditorTextHistory(manager:nil))
        XCTAssertEqual(workspace.graph,model)
    }
    @MainActor func testFocusAndNativeHistoryNotificationsRefreshAvailabilityWithoutPolling() async throws {
        let text=HistoryTextView();text.allowsUndo=true;text.string="old"
        var responder:NSResponder?=text
        let focus=EditorHistoryFocus(responder:{responder})
        let initial=focus.revision
        XCTAssertFalse(EditorHistoryDispatch.canPerform(.undo,context:nil,text:focus.textHistory))
        text.insertText("new",replacementRange:NSRange(location:0,length:3));text.breakUndoCoalescing()
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertGreaterThan(focus.revision,initial)
        XCTAssertTrue(EditorHistoryDispatch.canPerform(.undo,context:nil,text:focus.textHistory))
        let edited=focus.revision
        NotificationCenter.default.post(name:NSWindow.didUpdateNotification,object:nil)
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertEqual(focus.revision,edited,"Unchanged window updates do not invalidate menus repeatedly")
        responder=nil
        NotificationCenter.default.post(name:NSText.didEndEditingNotification,object:text)
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertGreaterThan(focus.revision,edited)
        XCTAssertNil(focus.textHistory)
    }

    @MainActor func testSameProjectReloadInvalidatesCapturedCommandsButKeepsFreshHistoryUsable() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.beginProject(ExampleFlows.parallelClassifier)
        let projectID=try XCTUnwrap(workspace.activeProjectID)
        let oldModel=EditorHistoryContext(workspace)
        workspace.beginProject(try workspace.projectLibrary.load(id:projectID),id:projectID)
        let modelBefore=workspace.graph
        workspace.moveNodes(ids:["input"],dx:35,dy:0)
        let modelMoved=workspace.graph
        EditorHistoryDispatch.perform(.undo,context:oldModel,text:nil)
        XCTAssertEqual(workspace.graph,modelMoved,"Old model context cannot undo edits made after same-project reload")
        let freshModel=EditorHistoryContext(workspace)
        EditorHistoryDispatch.perform(.undo,context:freshModel,text:nil)
        XCTAssertEqual(workspace.graph,modelBefore)
        EditorHistoryDispatch.perform(.redo,context:freshModel,text:nil)
        XCTAssertEqual(workspace.graph,modelMoved,"Ordinary undo/redo must not expire the active history session")

        let data=DataFlowWorkspaceStore(workspace:workspace);data.save()
        let oldData=EditorHistoryContext(data);data.reopen()
        let dataBefore=data.project
        data.moveNodes(ids:["cast"],dx:35,dy:0);let dataMoved=data.project
        EditorHistoryDispatch.perform(.undo,context:oldData,text:nil)
        XCTAssertEqual(data.project,dataMoved,"Old data context cannot undo edits made after reopening")
        EditorHistoryDispatch.perform(.undo,context:EditorHistoryContext(data),text:nil)
        XCTAssertEqual(data.project,dataBefore)
        let beforeReplacement=EditorHistoryContext(data)
        data.loadExample(columns:["replacement"]);let replaced=data.project
        EditorHistoryDispatch.perform(.undo,context:beforeReplacement,text:nil)
        XCTAssertEqual(data.project,replaced,"Explicit replacement invalidates a captured command")
        EditorHistoryDispatch.perform(.undo,context:EditorHistoryContext(data),text:nil)
        XCTAssertEqual(data.project,dataBefore,"The replacement itself remains normally undoable")

        let composite=CompositeWorkspaceStore(workspace:workspace);composite.save()
        let oldComposite=EditorHistoryContext(composite);composite.reopen()
        let compositeBefore=composite.state,id=try XCTUnwrap(composite.graph.nodes.first?.id)
        composite.moveNodes(ids:[id],dx:35,dy:0);let compositeMoved=composite.state
        EditorHistoryDispatch.perform(.undo,context:oldComposite,text:nil)
        XCTAssertEqual(composite.state,compositeMoved,"Old composite context cannot undo edits after same-context reopen")
        let freshComposite=EditorHistoryContext(composite)
        EditorHistoryDispatch.perform(.undo,context:freshComposite,text:nil)
        XCTAssertEqual(composite.state,compositeBefore)
        EditorHistoryDispatch.perform(.redo,context:freshComposite,text:nil)
        XCTAssertEqual(composite.state,compositeMoved)
    }

    @MainActor func testFocusedContextPublishesAvailabilityAfterActualGraphUndoRedo() async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.beginProject(ExampleFlows.parallelClassifier)
        workspace.moveNodes(ids:["input"],dx:25,dy:0)
        let context=EditorHistoryContext(workspace)
        var changes=0
        let observation=context.objectWillChange.sink { changes += 1 }
        defer {observation.cancel()}
        XCTAssertTrue(context.canPerform(.undo));XCTAssertFalse(context.canPerform(.redo))
        workspace.undo()
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertGreaterThan(changes,0,"Focused-object menus must be invalidated after history mutation")
        XCTAssertFalse(context.canPerform(.undo));XCTAssertTrue(context.canPerform(.redo))
        let afterUndo=changes
        workspace.redo()
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertGreaterThan(changes,afterUndo)
        XCTAssertTrue(context.canPerform(.undo));XCTAssertFalse(context.canPerform(.redo))
        let afterRedo=changes
        workspace.statusMessage="Unrelated status update"
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertEqual(changes,afterRedo,"Unchanged history availability need not invalidate the menu")
    }

    @MainActor func testStableProviderChangesEditorAndForwardsDataAndCompositeHistoryAvailability() async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        workspace.beginProject(ExampleFlows.parallelClassifier)
        workspace.moveNodes(ids:["input"],dx:25,dy:0)
        let provider=EditorHistoryProvider();provider.attach(workspace)
        let model=try XCTUnwrap(provider.context);provider.attach(workspace)
        XCTAssertTrue(provider.context === model,"Ordinary view updates retain the context object")
        let data=DataFlowWorkspaceStore(workspace:workspace)
        provider.attach(data)
        XCTAssertFalse(model.canPerform(.undo),"A no-longer-visible model context cannot handle a data command")
        let dataContext=try XCTUnwrap(provider.context)
        var updates=0
        let observation=provider.objectWillChange.sink { updates += 1 };defer {observation.cancel()}
        data.moveNodes(ids:["cast"],dx:25,dy:0)
        try await Task.sleep(nanoseconds:20_000_000)
        let beforeUndo=updates
        dataContext.perform(.undo)
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertGreaterThan(updates,beforeUndo);XCTAssertTrue(dataContext.canPerform(.redo))
        let composite=CompositeWorkspaceStore(workspace:workspace);provider.attach(composite)
        XCTAssertFalse(dataContext.canPerform(.redo))
        let id=try XCTUnwrap(composite.graph.nodes.first?.id)
        composite.moveNodes(ids:[id],dx:25,dy:0)
        try await Task.sleep(nanoseconds:20_000_000)
        let compositeContext=try XCTUnwrap(provider.context),beforeCompositeUndo=updates
        compositeContext.perform(.undo)
        try await Task.sleep(nanoseconds:20_000_000)
        XCTAssertGreaterThan(updates,beforeCompositeUndo);XCTAssertTrue(compositeContext.canPerform(.redo))
        provider.detach()
        XCTAssertNil(provider.context);XCTAssertFalse(compositeContext.canPerform(.redo))
    }

}
private final class HistoryTextView:NSTextView {
    private let history=UndoManager()
    override var undoManager:UndoManager? {history}
}
