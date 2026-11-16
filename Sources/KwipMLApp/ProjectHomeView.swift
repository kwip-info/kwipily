import KwipMLCore
import SwiftUI
import AppKit

/// The application landing surface is navigation, not a second workbench.
struct ProjectHomeView: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject private var registry = ProjectWindowRegistry.shared
    @AppStorage("kwipml.presentation.kwipilyTransitionDismissed.v1") private var transitionDismissed = false
    private static let welcomeIcon: NSImage? = {
        guard let url = Bundle.module.url(forResource: "KwipMLWelcomeIcon", withExtension: "png") else { return nil }
        return NSImage(contentsOf: url)
    }()
    private var version: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Development" }
    var body: some View {
        let recentProjects = store.recentProjects
        HStack(alignment: .center, spacing: 64) {
            VStack(spacing: 0) {
                if let icon = Self.welcomeIcon {
                    Image(nsImage: icon).resizable().scaledToFit().frame(width: 116, height: 116)
                        .accessibilityHidden(true)
                }
                Text("\(ProductBrand.name)").font(.system(size: 34, weight: .bold)).padding(.top, 12)
                Text("Formerly \(ProductBrand.formerName) · Version \(version)").font(.system(size: 12)).foregroundStyle(MLTheme.muted).padding(.top, 3)
                VStack(spacing: 17) {
                    WelcomeAction(title: "Create a new flow", subtitle: "Start with an empty canvas", symbol: "plus.app") { store.newProject() }
                        .disabled(store.isRunning)
                    WelcomeAction(title: "Start from a template", subtitle: "Choose a useful workflow and connect your data", symbol: "square.grid.2x2") { ApplicationNavigationWindows.shared.showTemplates() }
                    WelcomeAction(title: "Open a flow", subtitle: "Browse project files on your Mac", symbol: "folder") { store.openProject() }
                        .disabled(store.isRunning)
                    WelcomeAction(title: "Explore examples & lessons", subtitle: "Learn by building on the canvas", symbol: "square.stack.3d.up") { ApplicationNavigationWindows.shared.showLearn() }
                }.padding(.top, 34)
                HStack(spacing: 18) {
                    Button { ApplicationNavigationWindows.shared.showSettings() } label: { Label("Settings", systemImage: "gearshape") }
                        .buttonStyle(.plain)
                    if let restoration = registry.restoration {
                        WelcomeRestorationActions(coordinator: restoration, registry: registry)
                    }
                }.font(.system(size: 11)).foregroundStyle(MLTheme.muted).padding(.top, 22)
            }.frame(width: 310)
            VStack(alignment: .leading, spacing: 16) {
                if !transitionDismissed {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(ProductBrand.transitionMessage)
                            .font(.system(size: 12)).fixedSize(horizontal: false, vertical: true)
                        Button("Got it") { transitionDismissed = true }
                            .buttonStyle(FlatButtonStyle())
                            .accessibilityIdentifier("brand.transition.dismiss")
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                        .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                        .accessibilityIdentifier("brand.transition.notice")
                }
                if recentProjects.isEmpty {
                    Text("No Recent Projects").font(.system(size: 15, weight: .medium)).foregroundStyle(MLTheme.muted)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                } else {
                    Text("Recent Projects").font(.system(size: 15, weight: .semibold)).foregroundStyle(MLTheme.muted)
                    ThemedScrollView {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(RecentProjectGroup.make(recentProjects)) {group in
                                if group.entries.count==1,let entry=group.entries.first {recentRow(entry)}
                                else {
                                    DisclosureGroup {
                                        ForEach(group.entries) {entry in recentRow(entry)}
                                    } label: {
                                        HStack {Text(group.title).font(.system(size:13,weight:.medium));Spacer();Text("\(group.entries.count) projects").font(.caption).foregroundStyle(MLTheme.muted)}.padding(.vertical,9)
                                    }.tint(MLTheme.rose)
                                }
                            }
                        }
                    }
                }
            }.frame(maxWidth: .infinity).frame(height: 340)
        }.frame(maxWidth: 800).padding(.horizontal, 64).padding(.vertical, 44)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    private func recentRow(_ entry:RecentProjectEntry)->some View {
        Button {store.openRecentProject(entry)} label: {
            recentLabel(entry.title,detail:entry.detail,symbol:entry.isExternal ? "doc.badge.gearshape":"doc")
        }.buttonStyle(WelcomeRowStyle()).help(entry.url.path)
            .contextMenu {
                Button("Open flow") {store.openRecentProject(entry)}
                if case .managed=entry.source {Button("View generated code") {store.openRecentProject(entry,section:"Code")}}
                if case .external(let record)=entry.source {
                    Button("Allow folder access…") {store.renewExternalProjectAccess?(record)}
                    Button("Locate moved flow…") {store.locateExternalProject?(record)}
                }
            }
    }
    private func recentLabel(_ title: String, detail: String, symbol: String) -> some View {
        HStack(spacing: 11) {
            Image(systemName: symbol).font(.system(size: 23, weight: .light)).foregroundStyle(MLTheme.rose).frame(width: 27)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.system(size: 13, weight: .medium)).lineLimit(1)
                Text(detail).font(.system(size: 11)).foregroundStyle(MLTheme.muted).lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 0)
        }.padding(.horizontal, 8).padding(.vertical, 9).frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
    }
}

private struct WelcomeAction: View {
    let title: String
    let subtitle: String
    let symbol: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: symbol).font(.system(size: 25, weight: .light)).foregroundStyle(MLTheme.rose).frame(width: 28)
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.system(size: 13, weight: .semibold))
                    Text(subtitle).font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                }
                Spacer(minLength: 0)
            }.padding(6).contentShape(Rectangle())
        }.buttonStyle(WelcomeRowStyle())
    }
}

private struct WelcomeRowStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.foregroundStyle(Color.primary)
            .background(configuration.isPressed ? MLTheme.rose.opacity(0.12) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .opacity(enabled ? 1 : 0.4)
    }
}

struct CourseCoachView: View {
    @ObservedObject var store: WorkspaceStore
    private let lessons = [
        ("Begin with measurements", "Input declares the two numeric features that arrive from the Data view."),
        ("Add the first learnable projection", "Linear maps two features into eight learned features."),
        ("Introduce a nonlinearity", "ReLU keeps positive values. A network needs nonlinear operations to learn curved boundaries."),
        ("Create a second branch", "A separate Linear learns independent weights from the same input."),
        ("Give that branch a different behavior", "Tanh produces a smooth bounded representation. Both paths can learn complementary features."),
        ("Combine the paths", "Concatenate preserves both eight-feature outputs, creating sixteen features."),
        ("Project to two scores", "The classifier maps sixteen features to one logit for each class."),
        ("Expose the output", "Output identifies the tensor used by the training objective. The complete flow is ready to run.")
    ]
    var body: some View {
        if let step = store.tutorialStep {
            HStack(alignment: .top, spacing: 16) {
                Rectangle().fill(MLTheme.rose).frame(width: 2)
                VStack(alignment: .leading, spacing: 5) {
                    SectionLabel(title: "Guided build · \(min(step+1,8)) of 8")
                    Text(step < 8 ? lessons[step].0 : "Your architecture is ready").font(.system(size: 14, weight: .semibold))
                    Text(step < 8 ? lessons[step].1 : "Inspect the connections, choose compute in Settings, then Run flow. You can edit every block.").font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                }
                Spacer()
                if step < 8 { Button("Place next block") { store.advanceCourse() }.disabled(store.isRunning) }
                else { Button("Run this flow") { store.run() }.disabled(store.isRunning || !store.validation.isValid) }
                Button { store.tutorialStep = nil } label: { Image(systemName: "xmark") }.accessibilityLabel("Close guided build")
            }.padding(16).background(MLTheme.panel).fixedSize(horizontal: false, vertical: true)
        }
    }
}

struct ProjectLauncherView: View {
    @ObservedObject var store: WorkspaceStore
    @Environment(\.appearanceTheme) private var theme
    @Environment(\.colorSchemeContrast) private var contrast
    var body: some View {
        ProjectHomeView(store: store).background {
            ZStack {
                MLTheme.background
                WelcomeBackdrop().opacity(contrast == .increased ? 0 : (theme.colorScheme == .dark ? 1 : 0.45))
                    .allowsHitTesting(false).accessibilityHidden(true)
            }
        }
    }
}

struct WelcomeRestorationActions: View {
    @ObservedObject var coordinator: CanvasTabSessionCoordinator
    let registry: ProjectWindowRegistry
    @State private var showsDetails = false
    var body: some View {
        HStack(spacing: 10) {
            Button { registry.restoreSavedCanvasLayout() } label: {
                Label(coordinator.failures.isEmpty ? "Restore workspace" : "Retry workspace", systemImage: "arrow.counterclockwise")
            }.buttonStyle(.plain).help("Reopen your saved canvas windows and tab arrangement")
            if coordinator.status != nil {
                Button { showsDetails.toggle() } label: { Image(systemName: "exclamationmark.circle") }
                    .buttonStyle(.plain).accessibilityLabel("Workspace restoration details")
                    .popover(isPresented: $showsDetails) {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(coordinator.status ?? "").font(.caption)
                            ForEach(coordinator.failures) { failure in
                                Text(failure.target.restorationLabel + ": " + failure.reason).font(.caption).textSelection(.enabled)
                                if case .standalone = failure.target {
                                    Button("Renew folder access") { registry.renewStandaloneAccess(failure.target) }
                                }
                            }
                        }.padding(18).frame(width: 360)
                    }
            }
        }
    }
}
