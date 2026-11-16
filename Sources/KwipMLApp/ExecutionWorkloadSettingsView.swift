import SwiftUI
import KwipMLCore

struct ExecutionWorkloadSettingsView: View {
    @Binding var draft: ExecutionWorkloadDraft
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionLabel(title: "Per-run workload")
            FlatChoiceControl(title: "Workload profile", selection: draft.selection,
                options: (draft.allowsInheritance ? [.init(id: "inherit", title: "Inherit app workload")] : []) + [
                    .init(id: "legacy", title: "Conservative · existing limits"),
                    .init(id: "light", title: "Light"), .init(id: "balanced", title: "Balanced"),
                    .init(id: "throughput", title: "Throughput"), .init(id: "custom", title: "Custom")]) {
                do { try draft.choose($0); error = nil } catch { self.error = error.localizedDescription }
            }
            if draft.selection == "custom" {
                ForEach(ExecutionWorkloadPolicy.Setting.allCases, id: \.rawValue) { setting in
                    HStack(spacing: 10) {
                        Text(setting.title).font(.system(size: 12)).frame(width: 150, alignment: .leading)
                        TextField(setting.title, text: Binding(get: { draft.values[setting.rawValue, default: ""] },
                            set: { draft.values[setting.rawValue] = $0 })).flatField()
                            .accessibilityLabel("Workload " + setting.title)
                        Text(setting.unit).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                }
            } else if draft.selection != "inherit", let policy = displayPolicy {
                ForEach([ExecutionWorkloadPolicy.Setting.scratchBytes, .retainedBytes, .tableRows, .parallelTasks], id: \.rawValue) { setting in
                    HStack {
                        Text(setting.title).foregroundStyle(MLTheme.muted)
                        Spacer()
                        Text(String(policy.limit(setting)) + " " + setting.unit).monospacedDigit()
                    }.font(.caption)
                }
            }
            Text("Captured for future runs. These allowances do not grant files, reserve memory or change active jobs. CPU threads separately limit actual parallel work. Tool-specific format and safety bounds still apply.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose) }
        }
    }

    private var displayPolicy: ExecutionWorkloadPolicy? {
        (try? draft.policy()) ?? (try? ExecutionWorkloadPolicy(preset: .legacy))
    }
}

private extension ExecutionWorkloadPolicy.Setting {
    var title: String {
        switch self {
        case .scratchBytes: "Temporary working data"
        case .retainedBytes: "Retained run data"
        case .tableRows: "Table rows"
        case .tableCells: "Table cells"
        case .tableColumns: "Table columns"
        case .sourceBytes: "Source file"
        case .collectionMembers: "Collection files"
        case .collectionBytes: "Collection total"
        case .collectionMemberBytes: "Each collection file"
        case .planNodes: "Plan components"
        case .planEdges: "Plan connections"
        case .parallelTasks: "Parallel tasks"
        }
    }
    var unit: String {
        switch self {
        case .scratchBytes, .retainedBytes, .sourceBytes, .collectionBytes, .collectionMemberBytes: "bytes"
        default: "count"
        }
    }
}
