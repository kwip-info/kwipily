import Foundation
import KwipMLCore

@MainActor enum MeasuredDiagnosticNavigation {
    static func show(_ target:WorkflowNodeAddress,workflowSHA256:String,host:WorkflowWorkspaceSession,workspace:WorkspaceStore)throws {
        guard workspace.workflowSession === host,host.owner.isCurrent(host.owner.lease()),!host.owner.isBusy,!host.owner.hasPendingInspectorEdits,
              try WorkflowArchiveCodec.hash(WorkflowCodec.encode(host.owner.document))==workflowSHA256 else {throw DiagnosticContractError("Current canvas differs from the captured workflow. Inspect the captured evidence instead.")}
        guard host.owner.document.nodes.contains(where:{$0.address==target}),host.owner.select(.init(nodeAddresses:[target]),lease:host.owner.lease()) else {throw DiagnosticContractError("Captured component is not available in this canvas.")}
        // Build remount restores selection from the saved viewport. Persist the
        // canonical selection through its existing bridge while still in History,
        // retaining the user's zoom/origin instead of letting stale view state win.
        let view=host.canvas.canvasViewState
        guard CanvasViewportPersistence().request(store:host.canvas,revision:host.canvas.canvasViewRevision,
            zoom:view.zoom,origin:CGPoint(x:view.scrollX,y:view.scrollY)) else {throw DiagnosticContractError("The canvas viewport could not retain this selection. Reopen Build and select the component directly.")}
        workspace.showInspector=true;workspace.activeSection="Build"
    }
}
