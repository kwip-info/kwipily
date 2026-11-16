import SwiftUI
import KwipMLCore

/// Selection is local to this Train. Listing does not imply that a result is
/// reusable; the host verifies its pinned preparation evidence on selection.
struct WorkflowCacheSelectionView: View {
    @ObservedObject var context: WorkflowControlNodeContext
    @ObservedObject var host: WorkflowWorkspaceSession
    @ObservedObject var history: WorkflowHistoryStore
    var inlinePresentation=false
    @State private var choosing = false

    private var active: Bool {
        context.isCurrent && !context.isBusy && host.owner === context.authority.owner
            && !host.isRetired && history.isActive && history.sessionID == host.sessionID
    }
    private var selected: WorkflowArchiveCacheReference? {
        host.cacheBindings.reference(for: context.authority.address)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prior preparation").font(.caption.weight(.semibold))
            if !inlinePresentation {Text("Used when Train’s preparation cache selects Reuse. Sources are checked again on every run.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let selected {
                Text(inlinePresentation ? "Selected preparation result":"Archive \(selected.directoryID.prefix(12))").font(.caption).textSelection(.enabled)
                if !inlinePresentation {Text("Manifest \(selected.preparationManifestSHA256.prefix(12))…").font(.system(size:10,design:.monospaced)).foregroundStyle(MLTheme.muted).help(selected.preparationManifestSHA256)}
            } else {
                Text("No prior result selected.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            HStack {
                Button("Choose prior result") { choosing = true; history.refresh() }
                    .buttonStyle(FlatButtonStyle())
                if selected != nil {
                    Button("Clear") {
                        guard active else { return }
                        _ = host.clearPreparationCache(for: context.authority.address, lease: context.authority.token)
                    }.buttonStyle(FlatButtonStyle())
                }
            }.disabled(!active)
            if !inlinePresentation {Text(host.statusMessage).font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
        }
        .sheet(isPresented: $choosing) { chooser }
        .onChange(of: context.isCurrent) { _, current in if !current { choosing = false } }
    }
    private var chooser:some View {
        WorkflowHistoryPicker(title:"Choose prior preparation",
            explanation:"Choose a run that stored a preparation cache. Selection verifies its recorded evidence; reuse is checked against your next run.",
            emptyMessage:"No matching recorded runs. Run preparation with Store enabled to create a reusable result.",
            history:history,active:active,status:host.statusMessage,onClose:{choosing=false}) {entry in
                guard active else {return false}
                return host.selectPreparationCache(reference:entry.reference,for:context.authority.address,lease:context.authority.token)
            }
    }
}
