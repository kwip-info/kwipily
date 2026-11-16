import SwiftUI
import Combine
import KwipMLCore

enum WorkspaceToolbarSection {
    static func title(_ section: String) -> String { section == "Settings" ? "Configure" : section }
    static func symbol(_ section: String) -> String {
        switch section {
        case "Build": return "point.3.connected.trianglepath.dotted"
        case "History", "Experiments": return "clock"
        case "Code": return "chevron.left.forwardslash.chevron.right"
        case "Settings": return "slider.horizontal.3"
        case "Data": return "tablecells"
        case "Run": return "play"
        default: return "square.grid.2x2"
        }
    }
}

extension WorkspaceStore {
    var runConfigurationNodes: [WorkflowNode] {
        workflowSession?.owner.document.nodes.filter {
            switch $0.payload { case .runSetup, .train, .estimator: return true; default: return false }
        } ?? []
    }
    @discardableResult func revealRunConfiguration(_ address: WorkflowNodeAddress) -> Bool {
        guard let host = workflowSession, !isBusy, !host.canvas.isBusy,
              host.owner.projectID == activeProjectID,
              runConfigurationNodes.contains(where: { $0.address == address }),
              let displayID = host.canvas.projection.displayIDByAddress[address] else { return false }
        host.canvas.selectNode(displayID, extending: false)
        guard host.owner.selection.nodeAddresses == [address] else { return false }
        activeSection = "Build"; showInspector = true
        host.canvas.revealSelection()
        return true
    }
}

struct WorkspaceRunControl: View {
    @ObservedObject var store: WorkspaceStore
    @Environment(\.appearanceTheme) private var theme
    @State private var expanded = false
    @State private var readinessRevision=0
    private var readinessChanges:AnyPublisher<Void,Never> {
        store.workflowSession?.canvas.objectWillChange.eraseToAnyPublisher() ?? Empty<Void,Never>().eraseToAnyPublisher()
    }
    private var targets: [WorkflowNode] {
        store.runConfigurationNodes.filter { switch $0.payload { case .train, .estimator: return true; default: return false } }
    }
    private var menuHeight: CGFloat { CGFloat(580 + store.runConfigurationNodes.count * 55 + (targets.count > 1 ? targets.count * 40 : 0)) }
    var body: some View {
        let _=readinessRevision
        HStack(spacing: 0) {
            Button(store.isBusy ? (store.isPreviewing ? "Cancel preview" : "Cancel run") : "Run Flow") {
                store.isBusy ? store.cancelRun() : store.run()
            }.buttonStyle(FlatButtonStyle(prominent:store.isBusy || store.canRunCurrentFlow)).disabled(!store.isBusy && !store.canRunCurrentFlow)
                .help(store.currentRunConfigurationFinding?.message ?? (store.isCheckingRunConfiguration ? "Checking configuration…":"Run the current workflow"))
            if !store.isBusy,let finding=store.currentRunConfigurationFinding {
                Button {_=store.revealRunConfigurationFinding()} label: {Image(systemName:"exclamationmark.triangle.fill").foregroundStyle(Color.orange)}
                    .buttonStyle(.plain).padding(.horizontal,8).accessibilityLabel("Review run configuration")
                    .help(finding.message)
            }
            Rectangle().fill(theme.accentForeground.opacity(0.25)).frame(width: 1, height: 22)
            Button { expanded.toggle() } label: {
                Image(systemName: "chevron.down").font(.system(size: 10, weight: .semibold))
                    .frame(width: 28, height: 30).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityLabel("Run Flow options")
        }.foregroundStyle(store.isBusy || store.canRunCurrentFlow ? theme.accentForeground:MLTheme.muted)
            .background(store.isBusy || store.canRunCurrentFlow ? theme.accentColor:MLTheme.panel)
            .attachedDropdown(isPresented: $expanded, preferredHeight: min(640, menuHeight), width: 340) {
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        action(store.isBusy ? "Cancel active run" : "Run current flow", symbol: store.isBusy ? "stop" : "play",
                               enabled: store.isBusy || store.canRunCurrentFlow) {
                            store.isBusy ? store.cancelRun() : store.run()
                        }
                        PreparationSamplingControl(store: store)
                        Divider()
                        if targets.count > 1 {
                            SectionLabel(title: "Training paths").padding(10)
                            ForEach(targets) { node in
                                action("Run \(node.name.isEmpty ? node.address.localID : node.name)", symbol: "play", enabled: store.canRunCurrentFlow) {
                                    _ = store.workflowSession?.run(train: node.address)
                                }
                            }
                        }
                        Divider()
                        if store.workflowSession != nil {
                            VStack(alignment: .leading, spacing: 6) {
                                SectionLabel(title: "Run configuration")
                                Text("Open a component to edit its run settings.").font(.caption).foregroundStyle(MLTheme.muted)
                            }.padding(12)
                            ForEach(store.runConfigurationNodes) { node in
                                action(node.name.isEmpty ? node.address.localID : node.name, symbol: "slider.horizontal.3", subtitle: configurationPurpose(node), enabled: !store.isBusy) {
                                    _ = store.revealRunConfiguration(node.address)
                                }
                            }
                            if store.runConfigurationNodes.isEmpty {
                                Text("Add a Train or estimator component to configure execution.").font(.caption).foregroundStyle(MLTheme.muted).padding(10)
                            }
                        } else {
                            action("Run configuration…", symbol: "slider.horizontal.3") { store.activeSection = "Settings" }
                        }
                        action("Configure project…", symbol: "cpu") { store.activeSection = "Settings" }
                        Divider()
                        VStack(alignment: .leading, spacing: 14) {
                            VStack(alignment: .leading, spacing: 6) {
                                SectionLabel(title: "Run feedback")
                                Text("Applies to runs across all projects.").font(.caption).foregroundStyle(MLTheme.muted)
                            }
                            Toggle("Notify when a run completes", isOn: runFeedbackBinding(store: store, path: \.notifyCompleted, requestsPermission: true))
                            Toggle("Notify when a run fails", isOn: runFeedbackBinding(store: store, path: \.notifyFailed, requestsPermission: true))
                            Toggle("Show run status in menu bar", isOn: runFeedbackBinding(store: store, path: \.showStatusIcon))
                        }.padding(12)
                    }.font(.system(size: 12)).padding(6)
                }.background(MLTheme.panel).foregroundStyle(Color.primary)
            }
            // Observe the leaf canvas locally. Forwarding this to WorkspaceStore would
            // cycle through source-provider observations back into the workflow host.
            .onReceive(readinessChanges) {_ in readinessRevision &+= 1}
            .onChange(of: store.activeProjectID) { _, _ in expanded = false }
    }
    private func configurationPurpose(_ node: WorkflowNode) -> String {
        switch node.payload {
        case .runSetup: return "Historical Run setup · Reopen to move settings into Train"
        case .train: return "Train · Device, epochs, seed and training settings"
        case .estimator: return "Estimator · Backend and fitting settings"
        default: return "Component configuration"
        }
    }
    private func action(_ title: String, symbol: String, subtitle: String? = nil, enabled: Bool = true, perform: @escaping () -> Void) -> some View {
        Button { expanded = false; perform() } label: {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: symbol).frame(width: 16).padding(.top, 1)
                VStack(alignment: .leading, spacing: 5) {
                    Text(title).lineLimit(1)
                    if let subtitle { Text(subtitle).font(.system(size: 10)).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true) }
                }
            }.frame(maxWidth: .infinity, alignment: .leading).padding(11).contentShape(Rectangle())
        }.help(subtitle.map { title + " · " + $0 } ?? title).buttonStyle(.plain).disabled(!enabled)
    }
}
