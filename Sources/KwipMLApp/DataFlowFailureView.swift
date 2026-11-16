import SwiftUI

struct DataFlowFailureView: View {
    @ObservedObject var store: DataFlowWorkspaceStore
    @State private var expanded = true
    var body: some View {
        if let failure = store.preparationFailure {
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 9) {
                    Image(systemName: "exclamationmark.triangle").foregroundStyle(MLTheme.rose)
                    Text("\(failure.detail.nodeLabel ?? failure.detail.componentTitle) · \(failure.detail.componentTitle)").font(.system(size: 12, weight: .medium)).lineLimit(1)
                    Text(store.failureIsCurrent ? "Last failed preparation" : "Previous request · graph or source changed").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                    Spacer()
                    Button("Select block") { store.selectFailureNode() }
                        .disabled(!store.project.graph.nodes.contains { $0.id == failure.detail.nodeID && $0.operatorID == failure.detail.operatorID })
                        .accessibilityLabel("Select failed preparation block")
                    Button(expanded ? "Hide details" : "Show details") { expanded.toggle() }
                }
                if expanded {
                    HStack(alignment: .top, spacing: 18) {
                        VStack(alignment: .leading, spacing: 4) {
                            SectionLabel(title: "Observed failure")
                            Text(failure.detail.cause).font(.system(size: 11)).textSelection(.enabled)
                            Text("Block \(failure.detail.nodeID) · \(failure.detail.code)" + (failure.detail.inputRowCount.map { " · \($0) input rows" } ?? ""))
                                .font(.system(size: 10, design: .monospaced)).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                        VStack(alignment: .leading, spacing: 4) {
                            SectionLabel(title: "Check next")
                            Text(failure.detail.suggestion).font(.system(size: 11)).textSelection(.enabled)
                            Text("No automatic repair was applied.").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }.buttonStyle(.borderless).padding(11).background(MLTheme.panel)
        }
    }
}
