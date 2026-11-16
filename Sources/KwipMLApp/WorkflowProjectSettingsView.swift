import SwiftUI
import AppKit
import KwipMLCore

/// Project-level identity and file actions; computation stays on canonical components.
struct WorkflowProjectSettingsView:View {
    @ObservedObject var workspace:WorkspaceStore
    @ObservedObject var owner:WorkflowAuthoringStore
    @State private var titleDraft=""
    @State private var titleBase=""
    @State private var titleLease:WorkflowEditorLease?
    @State private var localError:String?
    private var isCurrent:Bool {
        workspace.workflowSession?.owner === owner && workspace.activeProjectID == owner.projectID
            && owner.isCurrent(owner.lease())
    }
    private var isBusy:Bool {!isCurrent || workspace.isBusy || owner.isBusy}
    var body:some View {
        ProjectConfigureShell(universal: true) { page in
            VStack(alignment: .leading, spacing: 18) {
                switch page {
                case .identity:
                    SectionLabel(title:"Project identity")
                    HStack(spacing:8) {
                    TextField("Project name",text:$titleDraft).flatField().accessibilityLabel("Workflow project name").onSubmit(applyTitle)
                    Button("Apply name",action:applyTitle).buttonStyle(FlatButtonStyle()).disabled(titleDraft == titleBase)
                    }.frame(maxWidth:530).disabled(isBusy || !owner.document.drafts.isEmpty)
                    if !owner.document.drafts.isEmpty {
                    Text("Finish or discard the open Block draft before changing the project name.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                VStack(alignment:.leading,spacing:6) {
                    metadata("Project ID",owner.projectID)
                    metadata("Source revision",String(owner.document.revision))
                    metadata("Components",String(owner.document.nodes.count))
                    metadata("Connections",String(owner.document.edges.count))
                    metadata("Pinned definitions",String(owner.document.definitions.count))
                }
                case .files:
                    OCRPackSettingsView(store:workspace)
                    Divider()
                    MediaLibrarySettingsView(store:workspace,projectID:owner.projectID)
                    Divider()
                    SectionLabel(title:"Project files")
                    HStack(spacing:8) {
                    Button("Save project") {
                        guard isCurrent else {return}
                        localError=workspace.saveManagedProject() ? nil : workspace.saveError ?? workspace.statusMessage
                    }.buttonStyle(FlatButtonStyle())
                    Button("Export project…") {guard isCurrent else {return};workspace.saveProject()}.buttonStyle(FlatButtonStyle())
                    Button("Reveal project") {
                        guard isCurrent else {return}
                        do { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: try owner.currentProjectRecord().path)]) }
                        catch { localError = error.localizedDescription }
                    }.buttonStyle(FlatButtonStyle())
                    }.disabled(isBusy)
                    Text("The portable export includes authored components, connections and pinned definitions. Local source bindings and file permissions stay on this Mac.").font(.caption).foregroundStyle(MLTheme.muted)
                case .training:
                    EmptyView() // Legacy training settings are not a universal-workflow route.
                case .webhooks:
                    WebhookSettingsView(workspace:workspace,projectID:owner.projectID)
                case .schedules:
                    ScheduledRunsSettingsView(workspace:workspace,projectID:owner.projectID)
                case .resources:
                    ExecutionResourceSettingsView(store: workspace, projectID: owner.projectID)
                case .capture:
                    TrainingCaptureSettingsView(store: workspace, projectID: owner.projectID)
                case .workspace:
                    if let host=workspace.workflowSession, host.owner === owner {
                        ProjectCanvasSettingsView(workspace:workspace,canvas:host.canvas).disabled(!isCurrent)
                    }
                }
                if let error=localError ?? workspace.saveError ?? owner.error {
                    Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                }
            }.frame(maxWidth: 650, alignment: .leading).frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear(perform:resetTitle)
        .onChange(of:owner.document) {_,_ in
            // Never overwrite unfinished typing with a different source revision.
            if titleDraft == titleBase {resetTitle()}
        }
    }
    private func metadata(_ label:String,_ value:String)->some View {
        HStack(alignment:.top,spacing:12) {
            Text(label).font(.caption).foregroundStyle(MLTheme.muted).frame(width:112,alignment:.leading)
            Text(value).font(.system(size:11,design:.monospaced)).textSelection(.enabled)
        }
    }
    private func resetTitle() {titleBase=owner.document.name;titleDraft=titleBase;titleLease=owner.lease()}
    private func applyTitle() {
        guard !isBusy,let lease=titleLease,owner.isCurrent(lease),owner.document.name == titleBase else {
            localError="The project changed. Reopen Configure before applying this name.";return
        }
        guard !titleDraft.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,titleDraft.count<=80 else {
            localError="Use a project name of 1–80 characters.";return
        }
        workspace.renameProject(titleDraft)
        guard owner.document.name == titleDraft else {localError=owner.error ?? workspace.saveError ?? workspace.statusMessage;return}
        localError=nil;resetTitle()
    }
}
