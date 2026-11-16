import SwiftUI
import KwipMLCore

extension WorkspaceStore {
    /// Current-document presentation only: no graph command, source revision or run policy.
    @discardableResult func configureProjectCanvas(projectID:String?,revision:UUID,update:(inout CanvasViewState)->Void)->Bool {
        guard activeProjectID == projectID,!isBusy else {return false}
        if let host=workflowSession {
            guard !host.isRetired,host.owner.projectID == projectID,
                  host.owner.isCurrent(host.owner.lease()),!host.canvas.isBusy,
                  host.canvas.canvasViewRevision == revision else {return false}
            var next=host.canvas.canvasViewState;update(&next)
            guard (try? WorkspaceViewState(modelCanvas:next).validate()) != nil else {return false}
            host.canvas.canvasViewState=next
        } else {
            guard canvasViewRevision == revision else {return false}
            var next=modelCanvasViewState;update(&next)
            guard (try? WorkspaceViewState(modelCanvas:next).validate()) != nil else {return false}
            modelCanvasViewState=next
        }
        return true
    }
}

/// One shared project presentation template for legacy and universal root canvases.
struct ProjectCanvasSettingsView<Canvas:CanvasWorkspace>:View {
    @ObservedObject var workspace:WorkspaceStore
    @ObservedObject var canvas:Canvas
    var body:some View {
        let projectID=workspace.activeProjectID,revision=canvas.canvasViewRevision
        VStack(alignment:.leading,spacing:18) {
            SectionLabel(title:"Canvas display")
            visibility("Mini map",visible:Binding(get:{canvas.canvasViewState.overviewVisible},set:{value in
                _=workspace.configureProjectCanvas(projectID:projectID,revision:revision) {$0.overviewVisible=value}
            }))
            Text("All canvases use left-to-right table blocks. Overview visibility is saved with this canvas’s layout; authored positions and computation are preserved.").font(.caption).foregroundStyle(MLTheme.muted)
            Divider()
            SectionLabel(title:"Project panels")
            visibility("Component ribbon",visible:$workspace.showLibrary)
            visibility("Configuration panel",visible:$workspace.showInspector)
            if workspace.workflowSession == nil { visibility("Results dock",visible:$workspace.showPreview) }
            Text("App defaults, appearance, keyboard shortcuts and reusable arrangements are available in \(ProductBrand.name) → Settings.").font(.caption).foregroundStyle(MLTheme.muted)
        }.disabled(workspace.isBusy || canvas.isBusy)
    }
    private func visibility(_ title:String,visible:Binding<Bool>)->some View {
        Button {visible.wrappedValue.toggle()} label: {
            HStack(spacing:12) {
                Image(systemName:visible.wrappedValue ? "checkmark.square.fill":"square")
                    .foregroundStyle(visible.wrappedValue ? MLTheme.rose:MLTheme.muted)
                Text(title);Spacer()
                Text(visible.wrappedValue ? "Visible":"Hidden").font(.caption).foregroundStyle(MLTheme.muted)
            }.frame(maxWidth:430)
        }.buttonStyle(FlatButtonStyle()).accessibilityLabel(title).accessibilityValue(visible.wrappedValue ? "Visible":"Hidden")
    }
}
