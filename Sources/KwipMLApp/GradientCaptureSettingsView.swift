import SwiftUI
import KwipMLCore

struct GradientCaptureSettingsView:View {
    @ObservedObject var workspace:WorkspaceStore
    @ObservedObject var capture:GradientCaptureStore
    var body:some View {
        let origin=workspace.activeProjectID ?? "unsaved"
        let expected=workspace.execution.gradientCapture
        GradientCaptureConfigurationEditor(configuration:workspace.execution.gradientCapture,canCapture:capture.canCapture,isBusy:workspace.isBusy,origin:origin,currentOrigin:{workspace.isBusy ? nil : workspace.activeProjectID ?? "unsaved"}) {next in
            guard !workspace.isBusy,workspace.execution.gradientCapture == expected,(next == nil || capture.canCapture) else {return false}
            workspace.setExecution {$0.gradientCapture=next};return true
        }.frame(maxWidth:600,alignment:.leading)
    }
}
