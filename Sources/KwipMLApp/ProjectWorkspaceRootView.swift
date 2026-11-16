import SwiftUI
import AppKit

struct ProjectWorkspaceRootView: View {
    @ObservedObject var session: ProjectWindowSession
    var body: some View {
        ProjectWorkspaceContent(store: session.workspace, affine: session.affine,
            normalization: session.normalization, tensor: session.tensor, preparation: session.preparation)
    }
}
private struct ProjectWorkspaceContent: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject var affine: AffineLearningStore
    @ObservedObject var normalization: NormalizationLearningStore
    @ObservedObject var tensor: TensorLearningStore
    @ObservedObject var preparation: PreparationSession
    var body: some View {
        Group {
            if store.activeSection == "Projects" { ProjectLauncherView(store: store) }
            else if store.activeSection == "Examples" { LearningLibraryView(store: store, affineLearning: affine, normalizationLearning: normalization, tensorLearning: tensor, navigation: ApplicationNavigationWindows.shared.learnNavigation) }
            else { WorkspaceView(store: store, preparationSession: preparation, affineLearning: affine, normalizationLearning: normalization, tensorLearning: tensor) }
        }.frame(minWidth: store.activeSection == "Projects" ? 820 : 1050,
            minHeight: store.activeSection == "Projects" ? 500 : 700).appAppearance(store.appearanceState)
    }
}

struct ProjectLauncherAttachment: NSViewRepresentable {
    let session: ProjectWindowSession
    let registry: ProjectWindowRegistry
    func makeNSView(context: Context) -> Attachment { Attachment() }
    func updateNSView(_ view: Attachment, context: Context) {
        view.session = session; view.registry = registry; view.register()
    }
    final class Attachment: NSView {
        weak var session: ProjectWindowSession?
        weak var registry: ProjectWindowRegistry?
        private weak var registeredWindow: NSWindow?
        private weak var registeredSession: ProjectWindowSession?
        private weak var registeredRegistry: ProjectWindowRegistry?
        override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); register() }
        func register() {
            guard let window, let session, let registry else {
                registeredWindow = nil; registeredSession = nil; registeredRegistry = nil
                return
            }
            guard registeredWindow !== window || registeredSession !== session || registeredRegistry !== registry else { return }
            // Set these before registration can publish and schedule another view update.
            registeredWindow = window; registeredSession = session; registeredRegistry = registry
            registry.registerLauncher(window, session: session)
        }
    }
}

struct ProjectLibrarySceneView: View {
    @ObservedObject var session: ProjectWindowSession
    @Environment(\.openWindow) private var openWindow
    @ObservedObject private var registry = ProjectWindowRegistry.shared
    var body: some View {
        ProjectWorkspaceRootView(session: session)
            .background(ProjectLauncherAttachment(session: session, registry: .shared))
            .onAppear {
                ProjectWindowRegistry.shared.reopenLibrary = { [weak session] section in
                    session?.workspace.refreshProjects()
                    session?.workspace.activeSection = section
                    openWindow(id: "project-library")
                }
                NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true)
            }
    }
}

extension CanvasTabSession.Target {
    var restorationLabel: String {
        switch self {
        case .library: return "Projects"
        case .managed(let id): return "Project " + String(id.uuidString.prefix(8))
        case .external(let id): return "External location " + String(id.uuidString.prefix(8))
        case .subflow(_, let path, _), .standalone(_, let path, _): return path
        case .inlineScope(_, _, let path): return "Block " + path.joined(separator: " › ")
        }
    }
}
