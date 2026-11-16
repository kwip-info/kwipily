import KwipMLCore
import SwiftUI
import AppKit

@main struct KwipMLApp: App {
    @NSApplicationDelegateAdaptor(FlowApplicationDelegate.self) private var applicationDelegate
    @StateObject private var launcher: ProjectWindowSession
    init() {
        InteractionProfileSession.startFromEnvironment()
        let workspace = WorkspaceStore()
        _ = MCPController.shared(directory:workspace.applicationDirectory)
        _ = ScheduledRunController.shared(workspace:workspace)
        _ = WebhookController.shared(workspace:workspace)
        FlowCanvasWindowRegistry.shared.appearance = workspace.appearanceState
        let session = ProjectWindowSession(workspace: workspace)
        ApplicationNavigationWindows.shared.configure(session: session)
        RunFeedbackController.shared.onOpenSettings = { ApplicationNavigationWindows.shared.showSettings() }
        RunFeedbackController.shared.start(preferences: workspace.appPreferences.effectiveRunFeedback)
        _launcher = StateObject(wrappedValue: session)
    }
    var body: some Scene {
        Window("\(ProductBrand.name) · Projects", id: "project-library") {
            ProjectLibrarySceneView(session: launcher)
        }.defaultSize(width: 900, height: 560)
            .commands { ProjectWindowCommands(registry: .shared, launcher: launcher) }
    }
}
