import SwiftUI
import KwipMLCore

/// Drafts are explicit; saving changes only future operation captures.
struct ExecutionResourceSettingsView: View {
    @ObservedObject var store: WorkspaceStore
    var projectID: String? = nil
    var scope: ExecutionSettingsScope = .all
    @State private var draft = ExecutionSettingsDraft(global: .factory)
    @State private var baseline = ExecutionSettingsDraft(global: .factory)
    @State private var settingsBaseline: ProjectSettingsSnapshot?
    @State private var applicationBaseline: ApplicationSettingsSnapshot?
    @State private var loaded = false
    @State private var error: String?
    @State private var notice: String?
    @State private var usage: WorkflowStorageAdmission.Snapshot?
    @State private var usageError: String?
    @State private var refreshing = false
    @State private var refreshID = UUID()
    private var current: Bool { projectID == nil || store.activeProjectID == projectID }
    private var effective: ResolvedExecutionPreferences? { try? store.executionPolicy(for: projectID) }
    private var dirty: Bool { draft != baseline }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: projectID == nil ? (scope == .storage ? "Storage allowances" : "Execution defaults") : "Execution overrides")
            if scope == .execution {
                Text("Defaults apply to future runs. Train selects the compute device; Automatic prefers the available Apple GPU. CPU threads and sampled worker RSS are not GPU quotas or a whole-app memory cap. Imported models also have separate file and memory-admission limits in Models & downloads.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            Text(projectID == nil ? (scope == .storage ? "Storage allowances for future runs. Existing files are never deleted by changing these limits." : "Defaults for projects on this Mac. Projects can override time, CPU threads and worker memory.")
                : "Inherit app defaults or choose limits for this project on this Mac.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            VStack(alignment: .leading, spacing: 10) {
                if scope != .storage {
                numberRow("Time allowance", value: $draft.seconds, unit: "seconds", enabled: projectID == nil || draft.overrideSeconds)
                if projectID != nil { inheritance("Time allowance", overriding: $draft.overrideSeconds) }
                numberRow("CPU threads", value: $draft.threads, unit: "1–64", enabled: projectID == nil || draft.overrideThreads)
                if projectID != nil { inheritance("CPU threads", overriding: $draft.overrideThreads) }
                Text("Controls worker compute thread pools. It does not change the selected CPU or Apple GPU device.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                FlatChoiceControl(title: "Worker memory policy", selection: draft.memoryMode,
                    options: (projectID == nil ? [] : [.init(id: "inherit", title: "Memory · inherit app default")]) + [
                        .init(id: "unlimited", title: "Memory · no RSS threshold"),
                        .init(id: "limit", title: "Memory · stop above threshold")]) { draft.memoryMode = $0 }
                if draft.memoryMode == "limit" {
                    numberRow("Worker memory", value: $draft.memoryGiB, unit: "GiB · 0.125–1024", enabled: true)
                }
                Text("Sampled resident memory (RSS) for the supervised worker. A threshold can be exceeded between samples; this is not a GPU-memory or whole-app hard cap.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                ExecutionWorkloadSettingsView(draft: $draft.workload)
                }
                if projectID == nil && scope != .execution {
                    numberRow("Preparation cache",value:$draft.preparationCacheBytes,unit:"bytes · 1–\(draft.preparationCacheMaximum)",enabled:true)
                    Text("Maximum retained table segments and fitted-state bytes for each materialize or reuse run. Captured when queued; changing this does not change queued work or delete saved History. This is separate from total managed storage.").font(.caption).foregroundStyle(MLTheme.muted)
                    FlatChoiceControl(title: "Managed storage policy", selection: draft.storageLimited ? "limited" : "unlimited", options: [
                        .init(id: "unlimited", title: "Managed storage · unlimited"),
                        .init(id: "limited", title: "Managed storage · set allowance")]) { draft.storageLimited = $0 == "limited" }
                    if draft.storageLimited { numberRow("Storage allowance", value: $draft.storageGiB, unit: "GiB", enabled: true) }
                } else if projectID != nil {
                    Text("Preparation cache is per run; managed storage is workspace-wide. Both allowances use app defaults in \(ProductBrand.name) → Settings.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
                if scope != .execution {
                Text("Admission includes retained managed outputs and reserved growth for active work. A smaller allowance refuses new work; it does not delete existing files.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                }
                HStack(spacing: 8) {
                    Button("Apply", action: apply).disabled(!dirty)
                    Button("Revert") { load(); notice = nil }.disabled(!dirty && error == nil)
                    Spacer()
                    Button(projectID == nil ? (scope == .storage ? "Reset storage allowances" : "Reset execution defaults") : "Reset to inheritance", action: reset)
                }.buttonStyle(FlatButtonStyle())
            }.disabled(!loaded || !current)
            if !loaded && current { Button("Reload saved settings", action: load).buttonStyle(FlatButtonStyle()) }
            if !current { Text("This project is no longer active. Reopen its settings to edit these values.").font(.caption).foregroundStyle(MLTheme.rose) }
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
            if let notice { Text(notice).font(.caption).foregroundStyle(MLTheme.teal) }
            Divider()
            effectiveSummary
            Text("Active operations keep their captured policy. Tool-specific time ceilings may shorten this allowance; the saved run records the effective limit.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if scope != .execution {
            HStack {
                Text("Managed storage usage").font(.system(size: 12, weight: .medium))
                Spacer()
                Button(refreshing ? "Refreshing…" : "Refresh usage", action: refreshUsage)
                    .buttonStyle(FlatButtonStyle()).disabled(refreshing)
            }
            if let usageError { Text(usageError).font(.caption).foregroundStyle(MLTheme.rose) }
            else if let usage {
                Text("Retained \(bytes(usage.retainedBytes)) · reserved growth \(bytes(usage.reservedGrowthBytes))")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            } else { Text("Refresh to measure managed storage. Other files on this Mac are excluded.").font(.caption).foregroundStyle(MLTheme.muted) }
            if projectID == nil {
                Divider()
                FlowRetentionSettingsView(store: store)
            }
            }
        }.frame(maxWidth: 600, alignment: .leading)
        .onAppear { load() }
        .onChange(of: projectID) { _, _ in load() }
        .onChange(of: store.appPreferences.execution) { _, _ in if !dirty { load() } }
        .onDisappear { refreshID = UUID(); refreshing = false }
    }
    @ViewBuilder private var effectiveSummary: some View {
        if let value = effective {
            Text("Saved effective policy").font(.system(size: 12, weight: .medium))
            if scope != .storage {
                Text("\(value.budget.maximumRunSeconds) seconds · \(origin(value.budget.source))\n\(value.budget.resources?.cpuThreads ?? 1) CPU threads · \(origin(value.cpuThreadsSource))\nWorker RSS: \(value.budget.resources?.maximumResidentBytes.map { bytes(Int64($0)) } ?? "no threshold") · \(origin(value.maximumResidentBytesSource))")
                    .font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                if let workload = value.workloadPolicy {
                    Text("Workload: \(workload.preset.rawValue) · \(projectID != nil && baseline.workload.selection != "inherit" ? "project override" : "app default")\nTemporary working data: \(workload.limit(.scratchBytes)) bytes · retained run data: \(workload.limit(.retainedBytes)) bytes")
                        .font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                }
            }
            if scope != .execution {
                Text("Managed storage: \(value.storageLimitBytes.map(bytes) ?? "unlimited") · app default\nPreparation cache: \(value.preparationCacheLimitBytes) bytes per run · app default")
                    .font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            }
        } else { Text("Saved policy unavailable. Revert to reload, or review the settings error.").font(.caption).foregroundStyle(MLTheme.rose) }
    }
    private func numberRow(_ title: String, value: Binding<String>, unit: String, enabled: Bool) -> some View {
        HStack(spacing: 10) {
            Text(title).font(.system(size: 12)).frame(width: 130, alignment: .leading)
            TextField(title, text: value).flatField().frame(width: 110).disabled(!enabled)
                .accessibilityLabel(title).onSubmit(apply)
            Text(unit).font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func inheritance(_ title: String, overriding: Binding<Bool>) -> some View {
        FlatChoiceControl(title: "\(title) inheritance", selection: overriding.wrappedValue ? "override" : "inherit", options: [
            .init(id: "inherit", title: "\(title) · inherit app default"),
            .init(id: "override", title: "\(title) · project override")]) { overriding.wrappedValue = $0 == "override" }
    }
    private func origin(_ source: ResolvedExecutionBudget.Source) -> String { source == .project ? "project override" : "app default" }
    private func bytes(_ value: Int64) -> String { ByteCountFormatter.string(fromByteCount: value, countStyle: .binary) }
    private func load() {
        guard current else { loaded = false; return }
        do {
            applicationBaseline = projectID == nil ? try store.applicationSettingsSnapshot() : nil
            settingsBaseline = try projectID.map { try store.projectSettingsSnapshot(projectID: $0, allowLegacy: true) }
            let global = try store.appPreferenceStore.load().execution
            let project = try projectID.map { try store.projectExecutionPreferenceStore.load(projectID: $0) }
            draft = .init(global: global, project: project); baseline = draft; loaded = true; error = nil
        } catch { loaded = false; self.error = error.localizedDescription }
    }
    private func apply() {
        guard current, loaded else { return }
        do {
            let saved: Bool
            if projectID != nil { saved = store.applyProjectSettings(.execution(try draft.project()), baseline: settingsBaseline) }
            else {
                let value=try draft.application(merging: store.appPreferenceStore.load().execution, scope: scope)
                saved=store.applyApplicationSettings(scope == .storage ? .storage(value):.execution(value),baseline:applicationBaseline)
            }
            guard saved else { error = store.appPreferencesError ?? "Settings were not saved."; return }
            load(); notice = "Saved for future operations."
        } catch { self.error = error.localizedDescription; notice = nil }
    }
    private func reset() {
        guard current, loaded else { return }
        do {
            let saved: Bool
            if projectID != nil { saved = store.applyProjectSettings(.reset(.execution), baseline: settingsBaseline) }
            else {
                saved=store.applyApplicationSettings(.reset(scope == .storage ? .storage:.execution),baseline:applicationBaseline)
            }
            guard saved else { error = store.appPreferencesError ?? "Settings were not reset."; return }
            load(); notice = projectID == nil ? "This page’s resource defaults reset. Capture and other pages are unchanged." : "Project resource limits now inherit app defaults. Capture choices are unchanged."
        } catch { self.error = error.localizedDescription }
    }
    private func refreshUsage() {
        let token = UUID(), workspace = store.workspaceDirectory, accountingRoot = store.applicationDirectory
        refreshID = token; refreshing = true; usageError = nil
        Task {
            let result = await Task.detached(priority: .utility) { Result { try WorkflowStorageAdmission(workspace: workspace, accountingRoot: accountingRoot).snapshot() } }.value
            guard refreshID == token else { return }
            refreshing = false
            switch result {
            case .success(let snapshot): usage = snapshot
            case .failure(let failure): usage = nil; usageError = "Storage usage unavailable. \(failure.localizedDescription)"
            }
        }
    }
}
