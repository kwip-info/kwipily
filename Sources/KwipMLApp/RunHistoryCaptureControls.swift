import SwiftUI

struct RunHistoryCaptureControls:View {
    @Environment(\.appearanceTheme) private var theme
    @ObservedObject var owner:WorkflowAuthoringStore
    var workspace:WorkspaceStore?
    @State private var issue:String?
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Toggle("Capture run details",isOn:Binding(get:{owner.runHistoryPolicy.enabled},set:{save($0 ? .all:.none)})).tint(theme.accentColor)
            if owner.runHistoryPolicy.enabled {
                Toggle("Diagnostics & performance",isOn:Binding(get:{owner.runHistoryPolicy.metricsAndDiagnostics},set:{save(.init(metricsAndDiagnostics:$0,componentOutputs:owner.runHistoryPolicy.componentOutputs))})).toggleStyle(.checkbox).tint(theme.accentColor)
                Toggle("Keep component data in History",isOn:Binding(get:{owner.runHistoryPolicy.componentOutputs},set:{save(.init(metricsAndDiagnostics:owner.runHistoryPolicy.metricsAndDiagnostics,componentOutputs:$0))})).toggleStyle(.checkbox).tint(theme.accentColor)
            }
            Text("Canvas versions are saved separately. Live table previews do not require keeping intermediate data in History. Required model and replay evidence is preserved.")
                .font(.caption2).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
            if let error=owner.runHistoryPolicyError {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
            if let issue {Text(issue).font(.caption).foregroundStyle(MLTheme.rose)}
        }.font(.system(size:12)).disabled(workspace == nil || workspace?.isBusy == true || workspace?.workflowSession?.hasActiveOperation == true || owner.isBusy || owner.hasPendingInspectorEdits)
    }
    private func save(_ policy:RunHistoryPolicy) {
        do {
            guard let workspace,workspace.workflowSession?.owner === owner else {throw ProjectSettingsFailure(code:"project_unavailable")}
            let snapshot=try workspace.projectSettingsSnapshot(projectID:owner.projectID)
            _ = try workspace.applyProjectSettings(workspace.previewProjectSettings(.historyCapture(policy),baseline:snapshot))
            issue=nil
        } catch {issue=error.localizedDescription}
    }
}
