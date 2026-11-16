import SwiftUI
import KwipMLCore

/// Shared authoring surface only. Hosts own preparation or lesson lifecycle and actions.
struct DataFlowEditorView: View {
    @ObservedObject var store: DataFlowWorkspaceStore
    var body: some View {
        VStack(spacing:0) {
            if store.workspace.showLibrary { DataFlowPaletteView(store:store);Divider() }
            if store.workspace.showInspector {
                HSplitView {
                    PreparationDataNodeInspectorHost(store:store).frame(minWidth:250,idealWidth:290,maxWidth:360)
                        .workspaceSplitLayout(store:store.workspace,scope:.dataRibbonInspector)
                    GraphCanvasView(store:store).frame(minWidth:380)
                }
            } else { GraphCanvasView(store:store) }
            if store.policy == .normalizationLesson {
                Divider()
                HStack(alignment:.top,spacing:8) {
                    Text("Canvas").font(.caption.weight(.semibold)).foregroundStyle(MLTheme.muted)
                    Text(store.statusMessage).font(.caption).fixedSize(horizontal:false,vertical:true)
                        .textSelection(.enabled)
                    Spacer(minLength:0)
                }
                .padding(.horizontal,10).padding(.vertical,6).background(MLTheme.panel)
                .accessibilityElement(children:.combine)
            }
        }
    }
}
