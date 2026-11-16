import Foundation
import AppKit
import Combine
import KwipMLCore

extension WorkspaceStore {
    var arrangementLibrary:WorkspaceArrangementLibrary {WorkspaceArrangementLibrary(directory:workspaceDirectory.appendingPathComponent("workspace-arrangements"))}
    func initializeWorkspaceArrangements() {
        refreshWorkspaceArrangements()
        arrangementObservation=objectWillChange.sink { [weak self] _ in self?.scheduleWorkspaceViewSave() }
        arrangementTerminationObservation=NotificationCenter.default.publisher(for:NSApplication.willTerminateNotification).sink { [weak self] _ in self?.persistWorkspaceView() }
    }
    func scheduleWorkspaceViewSave() {
        guard !isRestoringArrangement,let id=activeProjectID else{return}
        arrangementSaveTask?.cancel()
        let task=DispatchWorkItem { [weak self] in
            guard let self,self.activeProjectID==id,!self.isRestoringArrangement else{return}
            self.persistWorkspaceView()
        }
        arrangementSaveTask=task;DispatchQueue.main.asyncAfter(deadline:.now()+0.25,execute:task)
    }
    func captureWorkspaceView()->WorkspaceViewState {
        var model=modelCanvasViewState
        if let host=workflowSession,activeWorkflowCanvas(host) {
            model=workflowCanvasState(host.canvas.canvasViewState,host:host)
        } else {
            model.selectedNodeIDs=editorState.selection.nodeIDs;model.selectedEdgeIDs=editorState.selection.edgeIDs
        }
        var blocks = blockCanvasStates
        if let session = workflowChildren?.session ?? modelChildWindows?.session, ownsChildCanvasAuthority(session.owner),
           activeProjectID == session.owner.projectID, let editor = session.editor,
           let key = try? ModelChildCanvasKey.make(path: editor.path) {
            blocks[key] = editor.capturedCanvasViewState()
        }
        return WorkspaceViewState(showLibrary:showLibrary,showInspector:showInspector,showPreview:showPreview,activeSection:activeSection,modelCanvas:model,dataCanvas:dataCanvasViewState,panelLayouts:panelLayouts,compactNavigation:compactNavigation,compositeCanvasStates:compositeCanvasStates,blockCanvasStates:blocks)
    }
    func recordPanelSizes(_ sizes:[Double],scope:WorkspacePanelScope,projectID:String,revision:UUID) {
        guard activeProjectID==projectID,canvasViewRevision==revision,!isRestoringArrangement,sizes.count==scope.panes.count else{return}
        let preference=PanelLayoutPreference(sizes:Dictionary(uniqueKeysWithValues:zip(scope.panes.map(\.id),sizes)))
        guard (try? preference.validate(scope:scope)) != nil,panelLayouts[scope.rawValue] != preference else{return}
        panelLayouts[scope.rawValue]=preference
    }
    func persistWorkspaceView() {
        arrangementSaveTask?.cancel();arrangementSaveTask=nil
        guard !isRestoringArrangement,let id=activeProjectID else{return}
        do {try arrangementLibrary.saveProjectState(captureWorkspaceView(),projectID:id);if arrangementError != nil {arrangementError=nil}}
        catch {if arrangementError != error.localizedDescription {arrangementError=error.localizedDescription}}
    }
    func restoreWorkspaceView() {
        guard let id=activeProjectID else{return}
        do {installWorkspaceView(try arrangementLibrary.projectState(projectID:id) ?? initialWorkspaceView())}
        catch {arrangementError=error.localizedDescription;installWorkspaceView(WorkspaceViewState())}
    }
    func installWorkspaceView(_ state:WorkspaceViewState) {
        guard (try? state.validate()) != nil else{return}
        let restoring=isRestoringArrangement;isRestoringArrangement=true;defer{isRestoringArrangement=restoring}
        showLibrary=state.showLibrary;showInspector=state.showInspector;showPreview=state.showPreview;activeSection=state.activeSection
        modelCanvasViewState=state.modelCanvas;dataCanvasViewState=state.dataCanvas;compositeCanvasStates=state.compositeCanvasStates;blockCanvasStates=state.blockCanvasStates;panelLayouts=WorkspacePanelSizing.ribbonPreferences(state.panelLayouts);compactNavigation=state.compactNavigation ?? legacyCompactNavigationPreference()
        if let host=workflowSession,activeWorkflowCanvas(host) {
            restoreWorkflowCanvas(state.modelCanvas,host:host)
        } else {
        selectedNodeIDs=state.modelCanvas.selectedNodeIDs.intersection(Set(graph.nodes.map(\.id)))
        selectedEdgeIDs=state.modelCanvas.selectedEdgeIDs.intersection(Set(graph.edges.map(\.id)))
        selectedNodeID=graph.nodes.first{selectedNodeIDs.contains($0.id)}?.id
        selectedEdgeID=selectedNodeIDs.isEmpty ? graph.edges.first{selectedEdgeIDs.contains($0.id)}?.id:nil
        }
        canvasViewRevision = UUID()
        modelChildWindows?.session.restorePresentation()
        workflowChildren?.restorePresentation()
    }
    /// Install after canonical host activation and workspace-sidecar restoration.
    /// The caller retains this observation only for the lifetime of that host.
    func installWorkflowCanvasPersistence(_ host:WorkflowWorkspaceSession)->AnyCancellable {
        guard activeWorkflowCanvas(host) else {return AnyCancellable {}}
        let restoring=isRestoringArrangement;isRestoringArrangement=true
        restoreWorkflowCanvas(modelCanvasViewState,host:host)
        isRestoringArrangement=restoring
        return host.canvas.canvasViewStateUpdates.dropFirst().sink {[weak self,weak host] next in
            guard let self,let host else {return}
            // State updates arrive before the canvas property changes. Record `next`,
            // never recapture its old value or assign recursively into this publisher.
            self.recordWorkflowCanvasState(next,host:host,revision:host.canvas.canvasViewRevision)
        }
    }
    func recordWorkflowCanvasState(_ state:CanvasViewState,host:WorkflowWorkspaceSession,revision:UUID) {
        guard activeWorkflowCanvas(host),host.canvas.canvasViewRevision == revision,!isRestoringArrangement else {return}
        let next=workflowCanvasState(state,host:host)
        guard (try? WorkspaceViewState(modelCanvas:next).validate()) != nil else {return}
        // Active camera intent already lives synchronously in the canvas. Capture
        // and transition flush read it directly. Republishing every wheel/pan
        // packet here invalidated all workbench panels, despite debounced disk IO.
        let prior=modelCanvasViewState
        if prior.presentation != next.presentation || prior.overviewVisible != next.overviewVisible
            || prior.selectedNodeIDs != next.selectedNodeIDs || prior.selectedEdgeIDs != next.selectedEdgeIDs {
            modelCanvasViewState=next
        }
        scheduleWorkspaceViewSave()
    }
    /// Transition callers must honor false before retiring any current session.
    @discardableResult func flushWorkflowCanvasBeforeTransition()->Bool {
        guard let host=workflowSession else {return true}
        guard activeWorkflowCanvas(host),!isRestoringArrangement else {
            arrangementError="The workflow canvas cannot be saved during a session transition."
            statusMessage=arrangementError!;return false
        }
        arrangementSaveTask?.cancel();arrangementSaveTask=nil
        do {
            try arrangementLibrary.saveProjectState(captureWorkspaceView(),projectID:host.owner.projectID)
            arrangementError=nil;return true
        } catch {
            arrangementError=error.localizedDescription
            statusMessage="Workspace layout could not be saved: \(error.localizedDescription)"
            return false
        }
    }
    private func activeWorkflowCanvas(_ host:WorkflowWorkspaceSession)->Bool {
        workflowSession === host && activeProjectID == host.owner.projectID && !host.isRetired
            && host.owner.isCurrent(host.owner.lease())
    }
    private func workflowCanvasState(_ state:CanvasViewState,host:WorkflowWorkspaceSession)->CanvasViewState {
        var next=state
        next.selectedNodeIDs=Set(host.owner.selection.nodeAddresses.compactMap {host.canvas.projection.displayIDByAddress[$0]})
        next.selectedEdgeIDs=Set(host.owner.selection.edgeIDs.compactMap {host.canvas.projection.displayIDByEdgeID[$0]})
        return next
    }
    private func restoreWorkflowCanvas(_ state:CanvasViewState,host:WorkflowWorkspaceSession) {
        guard activeWorkflowCanvas(host) else {return}
        let selection=WorkflowSelection(
            nodeAddresses:Set(state.selectedNodeIDs.compactMap {host.canvas.projection.addressByDisplayID[$0]}),
            edgeIDs:Set(state.selectedEdgeIDs.compactMap {host.canvas.projection.edgeIDByDisplayID[$0]}))
        _=host.owner.select(selection,lease:host.owner.lease())
        let restored=workflowCanvasState(state,host:host)
        // Rotate before assignment so any old renderer's final callback is rejected.
        host.canvas.canvasViewRevision=UUID()
        host.canvas.canvasViewState=restored
        modelCanvasViewState=restored
    }
    func refreshWorkspaceArrangements() {
        do {workspaceArrangements=try arrangementLibrary.list();arrangementError=nil}
        catch {arrangementError=error.localizedDescription}
    }
    func saveWorkspaceArrangement(name:String,section:String?=nil) {
        var state=captureWorkspaceView();if let section {state.activeSection=section}
        do {_ = try arrangementLibrary.save(name:name,state:state);refreshWorkspaceArrangements()}
        catch {arrangementError=error.localizedDescription}
    }
    func applyWorkspaceArrangement(_ arrangement:WorkspaceArrangement) {installWorkspaceView(arrangement.state);persistWorkspaceView()}
    func renameWorkspaceArrangement(_ arrangement:WorkspaceArrangement,name:String) {
        do {_ = try arrangementLibrary.rename(id:arrangement.id,name:name);refreshWorkspaceArrangements()}
        catch {arrangementError=error.localizedDescription}
    }
    func deleteWorkspaceArrangement(_ arrangement:WorkspaceArrangement) {
        do {try arrangementLibrary.delete(id:arrangement.id);refreshWorkspaceArrangements()}
        catch {arrangementError=error.localizedDescription}
    }
}
