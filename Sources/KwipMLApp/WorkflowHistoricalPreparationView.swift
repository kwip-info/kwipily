import SwiftUI
import KwipMLCore

struct WorkflowHistoricalPreparationView: View {
    @ObservedObject var history: WorkflowHistoryStore
    @ObservedObject var workspace: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("SAVED PREPARATION").font(.system(size: 10, weight: .medium)).foregroundStyle(MLTheme.muted)
                Spacer()
                Button("Inspect saved preparation") { history.inspectPreparation() }
                    .buttonStyle(FlatButtonStyle()).disabled(!history.isActive || workspace.isBusy)
            }
            Text("Inspect the data and fitted state captured by this run. Current source files and canvas edits are not used.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if let inspection = history.preparationInspection {
                WorkflowHistoricalPreparationContent(inspection: inspection)
            } else {
                Text("Load the archived outputs to choose a component and inspect its ports.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                Spacer()
            }
        }
    }
}

private struct WorkflowHistoricalPreparationContent: View {
    @ObservedObject var inspection: WorkflowPreparationInspectionSession
    @State private var selectedID = ""
    private func cacheDescription(_ decision: DataFlowCacheReport.Decision) -> String {
        switch decision.reason {
        case "verifiedHit": return "Reused verified saved output"
        case "sourceRevalidated": return "Read source again"
        case "identityChanged": return "Recomputed after an input or configuration change"
        case "notMaterialized": return "Computed and saved for later reuse"
        case "disabled": return "Computed with caching off"
        default: return decision.status + " · " + decision.reason
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(inspection.statusMessage).font(.caption).textSelection(.enabled)
                Spacer()
                if inspection.isBusy {
                    ProgressView().controlSize(.small)
                    Button("Cancel inspection") { inspection.cancel() }.buttonStyle(FlatButtonStyle())
                }
            }
            if let snapshot = inspection.snapshot {
                let nodes = snapshot.input.nodes
                let node = nodes.first { $0.address.localID == selectedID } ?? nodes.first
                FlatChoiceControl(title: "Captured component", selection: node?.address.localID ?? "",
                    options: nodes.map { .init(id: $0.address.localID, title: $0.name.isEmpty ? $0.operatorTitle : $0.name) }) { selectedID = $0 }
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        if let node {
                            Text(node.operatorTitle).help(node.operatorID).font(.caption).foregroundStyle(MLTheme.muted)
                            if let decision = snapshot.cacheReport.nodes.first(where: { $0.nodeID == node.address.localID }) {
                                Text("Recorded cache decision · \(cacheDescription(decision))").font(.caption)
                            } else { Text("Recorded cache policy · off · computed during this run").font(.caption) }
                            if let ports = snapshot.outputs[node.address] {
                                ForEach(ports.keys.sorted(), id: \.self) { port in
                                    DataPortPreviewView(port: port, output: WorkflowCapturedDataOutput.render(ports[port]!) as? [String: Any] ?? [:])
                                }
                            }
                        }
                        DisclosureGroup("Captured provenance") {
                            Text("Run \(snapshot.input.runID)\nSource revision \(snapshot.input.capturedRevision)\nTerminal status \(snapshot.input.terminalStatus.rawValue)\nPreparation \(snapshot.input.manifestSHA256)\nRecorded NumPy \(snapshot.input.numpyVersion)\nHistorical evidence; current source validity and cache reuse eligibility are unchecked.")
                                .font(.system(size: 10, design: .monospaced)).textSelection(.enabled)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(.trailing, 8)
                }
            } else { Spacer() }
        }
    }
}
