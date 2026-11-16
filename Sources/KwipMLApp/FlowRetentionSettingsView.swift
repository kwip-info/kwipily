import SwiftUI
import KwipMLCore

/// Separate authoring evidence from execution outputs and user document ownership.
struct FlowRetentionSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    @State private var records = "4096"
    @State private var mebibytes = "256"
    @State private var baselineRecords = "4096"
    @State private var baselineMiB = "256"
    @State private var loaded = false
    @State private var error: String?
    @State private var notice: String?
    private var dirty: Bool { records != baselineRecords || mebibytes != baselineMiB }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionLabel(title: "Flow authoring evidence")
            Text("Global allowances applied separately to each managed or external flow folder. Open canvases read the saved values on their next draft or component publication.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                Text("Retained records").font(.system(size: 12)).frame(width: 130, alignment: .leading)
                TextField("Retained authoring records", text: $records).flatField().frame(width: 110)
                    .accessibilityLabel("Retained authoring records").onSubmit(apply)
                Text("1–100,000 per folder").font(.caption).foregroundStyle(MLTheme.muted)
            }
            HStack {
                Text("Evidence allowance").font(.system(size: 12)).frame(width: 130, alignment: .leading)
                TextField("Authoring evidence MiB", text: $mebibytes).flatField().frame(width: 110)
                    .accessibilityLabel("Authoring evidence MiB").onSubmit(apply)
                Text("MiB per folder · up to 1 TiB").font(.caption).foregroundStyle(MLTheme.muted)
            }
            Text("Counts retained private-draft files and publication recovery records (.kwipflow-draft-* and .kwipflow-publication-*). Excludes revision archives, completed flow files, datasets and execution outputs. Both allowances must fit each new publication.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            Text("Lowering these values never deletes files. A full folder refuses new evidence; raise the allowance to retry. Discarding an active draft remains available and retains its recovery evidence.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                Button("Apply authoring allowance", action: apply).disabled(!loaded || !dirty)
                Button("Revert") { load() }.disabled(!dirty && error == nil)
                Spacer()
                Button("Use defaults") { records = "4096"; mebibytes = "256"; notice = "Defaults selected. Apply to save." }.disabled(!loaded)
            }.buttonStyle(FlatButtonStyle())
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
            if let notice { Text(notice).font(.caption).foregroundStyle(MLTheme.teal) }
        }.onAppear(perform: load)
        .onChange(of: store.appPreferences.authoringRetention) { _, _ in if !dirty { load() } }
    }
    private func load() {
        do {
            let policy = try store.appPreferenceStore.load().effectiveAuthoringRetention
            records = String(policy.maximumFiles)
            mebibytes = String(Double(policy.maximumBytes) / 1_048_576)
            baselineRecords = records; baselineMiB = mebibytes; loaded = true; error = nil; notice = nil
        } catch { loaded = false; self.error = error.localizedDescription }
    }
    private func apply() {
        guard loaded else { return }
        do {
            guard let count = Int(records.trimmingCharacters(in: .whitespaces)),
                  let amount = Double(mebibytes.trimmingCharacters(in: .whitespaces)), amount.isFinite,
                  amount > 0, amount <= 1_048_576, amount * 1_048_576 >= 1 else {
                throw AppPreferencesError.invalid("enter a whole record count and a positive MiB allowance up to 1 TiB.")
            }
            let policy = try FlowAuthoringRetentionPreferences(maximumFiles: count, maximumBytes: Int((amount * 1_048_576).rounded()))
            guard store.setAuthoringRetention(policy) else { error = store.appPreferencesError; return }
            load(); notice = "Saved. Open canvases use this allowance on their next publication."
        } catch { self.error = error.localizedDescription; notice = nil }
    }
}
