import SwiftUI

/// Read the original local snapshot only when requested; never substitute today's evaluation.
struct RecordedDiagnosticView: View {
    @ObservedObject var store: WorkspaceStore
    let record: ExperimentRecord
    @State private var expanded = false
    @State private var loaded: DiagnosticArchiveLoad?
    var body: some View {
        DisclosureGroup("Recorded diagnostics · \(record.id.prefix(6))", isExpanded: $expanded) {
            Group {
                if let loaded {
                    switch loaded {
                    case .legacy:
                        Text("No diagnostic report was recorded for this experiment.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    case .unavailable(let message):
                        Text("Saved diagnostics unavailable: \(message)").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                        Button("Retry verification") { load() }.buttonStyle(FlatButtonStyle())
                    case .verified(let report):
                        Text("Original recorded evaluation · verified against its saved inputs and rule version. Outcome: \(record.status).")
                            .font(.caption).foregroundStyle(MLTheme.teal)
                        DiagnosticReportPanel(result: .success(report))
                    }
                } else {
                    ProgressView("Verifying saved diagnostics…").font(.caption)
                }
            }.padding(.top, 8)
        }
        .accessibilityIdentifier("recorded-diagnostics-\(record.id)")
        .onChange(of: expanded) { _, open in if open { load() } else { loaded = nil } }
        .onChange(of: record.id) { _, _ in loaded = nil; expanded = false }
    }
    private func load() { loaded = store.loadExperimentDiagnostics(record) }
}
