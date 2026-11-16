import SwiftUI
import KwipMLCore

/// Stable page instances preserve explicit Apply/Revert drafts during navigation.
struct AppSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    @ObservedObject var navigation = AppSettingsNavigation()
    private var pages: [SettingsPage] { SettingsPage.matches(navigation.search) }
    var body: some View {
        SettingsPageShell(title: "Settings", searchPlaceholder: "Search settings",
            emptyTitle: "No matching settings", clearSearchLabel: "Clear settings search", identifierPrefix: "settings",
            footer: "\(ProductBrand.name) \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "Development")",
            groups: ["Everyday", "Resources", "Access"], pages: SettingsPage.allCases,
            matchingPages: pages, subtitle: { $0.subtitle }, selected: $navigation.selected, search: $navigation.search) { page in
                content(page)
                if let error = store.appPreferencesError { Text(error).font(.caption).foregroundStyle(MLTheme.rose) }
            }.buttonStyle(FlatButtonStyle())
    }
    @ViewBuilder private func content(_ page: SettingsPage) -> some View {
        switch page {
        case .webhooks:WebhookSettingsView(workspace:store)
        case .schedules:ScheduledRunsSettingsView(workspace:store)
        case .models: ModelLibrarySettingsView(store:store)
        case .mcp: MCPSettingsView(controller:.shared(directory:store.applicationDirectory))
        case .support: SupportDiagnosticSettingsView()
        case .updates: DirectUpdateSettingsView()
        case .appearance: AppearanceSettingsView(store: store)
        case .canvas: canvasDefaults
        case .shortcuts: ShortcutSettingsView(store: store)
        case .feedback: RunFeedbackSettingsView(store: store)
        case .learning: LearningResetSettingsView(store: store)
        case .execution:
            runtimeStatus
            Divider()
            ExecutionResourceSettingsView(store: store, scope: .execution)
            Divider()
            TrainingCaptureSettingsView(store: store)
        case .storage:
            ExecutionResourceSettingsView(store: store, scope: .storage)
            Divider()
            ScratchCleanupSettingsView(store: store).id(store.workspaceDirectory)
            Divider()
            SectionLabel(title: "Window recovery")
            Toggle("Reopen canvas windows at launch", isOn: Binding(get: { store.appPreferences.restoreCanvasSessionAtLaunch != false }, set: { _ = store.setRestoreCanvasSessionAtLaunch($0) }))
            Text("Saved project and Block tabs reopen without starting runs. Unavailable files remain available for retry from the welcome window.").font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:"Scroll zoom sensitivity",selection:String(store.appPreferences.effectiveCanvasZoomSensitivity),options:[
                .init(id:"0.25",title:"25% · Very slow"),.init(id:"0.5",title:"50% · Slow"),.init(id:"0.75",title:"75%"),.init(id:"1.0",title:"100% · Standard"),.init(id:"1.25",title:"125%"),.init(id:"1.5",title:"150% · Fast"),.init(id:"2.0",title:"200% · Very fast")]) {if let value=Double($0) {_=store.setCanvasZoomSensitivity(value)}}
            Text("Adjust wheel and trackpad scroll zoom on every canvas. Panning, Fit and typed zoom levels stay the same.").font(.caption).foregroundStyle(MLTheme.muted)
            Button("Restore saved workspace") { ProjectWindowRegistry.shared.restoreSavedCanvasLayout() }
        case .permissions:
            SourcePermissionSettingsView(store:store)
            Divider()
            SectionLabel(title:"Other access controls")
            Button("Models, assets & Hub credentials") {navigation.open(.models)}
            Button("MCP clients, actions & project access") {navigation.open(.mcp)}
            Text("Local-source grants, provider credentials and MCP permissions are separate. Source revocation does not erase already-captured data.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private var runtimePresentation: RuntimeSettingsPresentation {
        RuntimeSettingsPresentation(status: store.capabilityStatus, version: store.runtimeContractVersion,
            contractVerified: store.runtimeExecutionContract != nil, devices: store.availableDevices,
            componentCount: store.verifiedRuntimeComponentCount, isBusy: store.isBusy,
            isChecking: store.capabilityOperation != nil || store.pendingCapabilityOperation != nil)
    }
    private var runtimeStatus: some View {
        let presentation = runtimePresentation
        return VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Runtime status")
            Text(presentation.status).font(.callout).textSelection(.enabled)
                .accessibilityIdentifier("settings.runtime.status")
            VStack(alignment: .leading, spacing: 8) {
                runtimeRow("Runtime", value: presentation.versionText)
                runtimeRow("Verified devices", value: presentation.devicesText)
                runtimeRow("Components", value: presentation.componentsText)
            }
            if let error = store.runtimeError {
                Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
            }
            HStack {
                Button("Refresh runtime status") {
                    guard runtimePresentation.canRefresh else { return }
                    store.refreshCapabilities()
                }.disabled(!presentation.canRefresh)
                    .accessibilityIdentifier("settings.runtime.refresh")
                if store.capabilityOperation != nil || store.pendingCapabilityOperation != nil {
                    ProgressView().controlSize(.small)
                    Text("Checking runtime…").font(.caption).foregroundStyle(MLTheme.muted)
                } else if store.isBusy {
                    Text("Available after the current operation finishes.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            Text("Checks the local runner and optional component support. Verified components describe runtime availability, not release coverage. Refreshing does not change your execution defaults.")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }.padding(14).frame(maxWidth: .infinity, alignment: .leading)
            .overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func runtimeRow(_ title: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(title).font(.caption).foregroundStyle(MLTheme.muted).frame(width: 116, alignment: .leading)
            Text(value).font(.callout).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    private var canvasDefaults: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionLabel(title: "Canvas presentation")
            Toggle("Reopen canvas windows at launch",isOn:Binding(get:{store.appPreferences.restoreCanvasSessionAtLaunch != false},set:{_=store.setRestoreCanvasSessionAtLaunch($0)}))
            FlatChoiceControl(title:"Scroll zoom sensitivity",selection:String(store.appPreferences.effectiveCanvasZoomSensitivity),options:[
                .init(id:"0.25",title:"25% · Very slow"),.init(id:"0.5",title:"50% · Slow"),.init(id:"0.75",title:"75%"),.init(id:"1.0",title:"100% · Standard"),.init(id:"1.25",title:"125%"),.init(id:"1.5",title:"150% · Fast"),.init(id:"2.0",title:"200% · Very fast")]) {if let value=Double($0) {_=store.setCanvasZoomSensitivity(value)}}
            Text("Adjust wheel and trackpad scroll zoom on every canvas. Panning, Fit and typed zoom levels stay the same.").font(.caption).foregroundStyle(MLTheme.muted)
            Button("Restore saved workspace") {ProjectWindowRegistry.shared.restoreSavedCanvasLayout()}
            Text("Panel visibility and mini map are configured per project in Configure → Canvas. They do not change computation.").font(.caption).foregroundStyle(MLTheme.muted)
            Text("Every component uses one rounded table frame with named, typed connection rows. Layout is left-to-right, with inputs on the left and outputs on the right. Existing authored positions and saved arrangements are retained.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
}

/// Presentation only: default device guesses remain hidden until the exact contract is verified.
struct RuntimeSettingsPresentation {
    let status: String
    let versionText: String
    let devicesText: String
    let componentsText: String
    let canRefresh: Bool
    init(status: String, version: String?, contractVerified: Bool, devices: [String], componentCount: Int,
         isBusy: Bool, isChecking: Bool) {
        self.status = status
        let verified = contractVerified && version != nil
        versionText = verified ? "PyTorch \(version!)" : "Not verified"
        devicesText = verified && !devices.isEmpty ? devices.map {
            $0 == "cpu" ? "CPU" : ($0 == "mps" ? "Apple GPU (MPS)" : $0)
        }.joined(separator: ", ") : "Not verified"
        componentsText = verified ? "\(componentCount) verified components" : "Not verified"
        canRefresh = !isBusy && !isChecking
    }
}
