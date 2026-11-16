import SwiftUI
import KwipMLCore

@MainActor struct ScratchCleanupSettingsView: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    @ObservedObject var store: WorkspaceStore
    @State private var report: WorkflowScratchCleanup.Report?
    @State private var task: Task<Void, Never>?
    @State private var busy = false
    @State private var message = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionLabel(title: "Temporary capture files")
            Text("Review disposable capture stages in the current workspace. Only stages registered by this version and no longer in use can be removed.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if let report {
                Text("\(report.candidates.count) removable stages · \(ByteCountFormatter.string(fromByteCount: report.reclaimableBytes, countStyle: .file))")
                    .font(.callout)
                if report.retainedStages > 0 {
                    Text("\(report.retainedStages) other stages retained: active, unregistered or not safely verifiable.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
            HStack(spacing: 10) {
                Button("Review temporary files") { run(removing: false) }.buttonStyle(FlatButtonStyle()).disabled(busy || store.isBusy)
                Button("Remove reviewed temporary files") { run(removing: true) }.buttonStyle(FlatButtonStyle())
                    .disabled(busy || store.isBusy || report?.candidates.isEmpty != false)
                if busy { ProgressView().controlSize(.small); Button("Cancel") { task?.cancel() }.buttonStyle(FlatButtonStyle()) }
            }
            if !message.isEmpty { Text(message).font(.caption).textSelection(.enabled) }
            DisclosureGroup("What is retained") {
                Text("Projects, saved History, source snapshots, table-query indexes, exports, licenses and window recovery are excluded. Interrupted flow drafts and publication records are recovery evidence; their existing retention allowance bounds growth but cleanup does not remove them. Older unregistered stages remain untouched. This is an explicit cleanup, not automatic History retention or a disk quota. Completed History requires a separate deliberate retention decision.")
                    .font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 4)
            }.font(.caption).tint(appearanceTheme.accentColor)
        }.onDisappear { task?.cancel() }
    }
    private func run(removing: Bool) {
        guard !busy, !store.isBusy else { return }
        let workspace = store.workspaceDirectory
        let candidates = removing ? report?.candidates : nil
        guard !removing || candidates?.isEmpty == false else { return }
        busy = true; message = ""
        task = Task {
            let worker = Task.detached(priority: .utility) { () throws -> (WorkflowScratchCleanup.Report, Int64?) in
                let removed = try candidates.map { try WorkflowScratchCleanup.remove($0, workspace: workspace) }
                return (try WorkflowScratchCleanup.scan(workspace: workspace), removed)
            }
            defer { busy = false; task = nil }
            do {
                let (value, removed) = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
                report = value
                if let removed { message = "Removed \(ByteCountFormatter.string(fromByteCount: removed, countStyle: .file)) of temporary captures." }
            } catch is CancellationError {
                report = nil; message = "Cleanup stopped. Some reviewed temporary files may already be removed; review again to continue."
            } catch {
                report = nil; message = error.localizedDescription + " Review again before retrying."
            }
        }
    }
}
