import SwiftUI

/// Retains one context while editing values. Selection/reload generations create a new
/// authority; removed views retire their old callbacks rather than retargeting them.
struct PreparationDataNodeInspectorHost: View {
    @ObservedObject var store: DataFlowWorkspaceStore
    var body: some View {
        if let id=store.selectedNodeID {
            PreparationSelectedNodeInspector(store:store,nodeID:id)
                .id(store.numericEditOrigin(nodeID:id,key:"context"))
        } else {
            Text(store.policy == .normalizationLesson ? "Select a block to configure its columns and connections." : "Select a block to configure it, inspect outputs, or choose its local source.")
                .font(.caption).foregroundStyle(MLTheme.muted).padding(16)
        }
    }
}

private struct PreparationSelectedNodeInspector: View {
    @State private var context: PreparationDataNodeContext
    private let store: DataFlowWorkspaceStore
    private let nodeID: String
    init(store:DataFlowWorkspaceStore,nodeID:String) {
        self.store=store; self.nodeID=nodeID
        _context=State(initialValue:PreparationDataNodeContext(store:store,nodeID:nodeID))
    }
    var body: some View {
        DataNodeInspectorView(context:context)
            .onAppear { if !context.isCurrent { context=PreparationDataNodeContext(store:store,nodeID:nodeID) } }
            .onDisappear { context.retire() }
    }
}
