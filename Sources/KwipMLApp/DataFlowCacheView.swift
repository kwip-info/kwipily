import SwiftUI

struct DataFlowCacheView: View {
    @ObservedObject var store: DataFlowWorkspaceStore
    var body: some View {
        HStack(spacing: 12) {
            Text("Execution").font(.system(size: 11, weight: .semibold))
            FlatChoiceControl(title: "Preparation cache policy", selection: store.cachePolicy.rawValue, options: [
                FlatChoiceOption(id: "materialize", title: "Recompute & retain"),
                FlatChoiceOption(id: "reuse", title: "Reuse verified outputs"),
                FlatChoiceOption(id: "off", title: "Recompute only")
            ]) { value in if let policy = DataFlowCachePolicy(rawValue: value) { store.setCachePolicy(policy) } }
                .frame(width: 210).disabled(store.isBusy)
            Text(store.cacheSummary).font(.system(size: 10)).foregroundStyle(MLTheme.muted).lineLimit(2)
            Spacer(minLength: 0)
            if store.cachePolicy == .reuse && store.lastCacheReference == nil {
                Text("Recompute first").font(.caption).foregroundStyle(.orange)
            }
        }.padding(.horizontal, 12).padding(.vertical, 7).background(MLTheme.panel)
    }
}
