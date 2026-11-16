import SwiftUI
import KwipMLCore

struct DataFlowWorkspaceView: View {
    @ObservedObject var store: DataFlowWorkspaceStore
    @State private var trainingSelection = false
    @State private var lessonVisible = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Text("Data preparation").font(.headline)
                FlatChoiceControl(title: "Preparation examples", selection: "", options: [FlatChoiceOption(id: "scaler", title: "Training-only standardization"), FlatChoiceOption(id: "fitted", title: "Missing values & categorical features"), FlatChoiceOption(id: "sensor", title: "Sensor selection & filtering")], placeholder: "Load example…") { selection in
                    if selection == "fitted" { store.loadBundledFittedExample() }
                    else if selection == "sensor" { store.loadBundledSensorExample() }
                    else { store.loadExample(columns: store.sourceColumns.isEmpty ? ["x1", "x2"] : Array(store.sourceColumns.prefix(2))) }
                }.frame(maxWidth: 210).disabled(store.workspace.isBusy)
                Button("Lesson") { lessonVisible = true }.popover(isPresented: $lessonVisible) {
                    PreparationLessonView(text: store.fittedLessonText) { lessonVisible = false }
                }
                Button("Save") { store.save() }
                Button("Reopen") { store.reopen() }.disabled(store.isBusy)
                Button("Export flow") { store.exportPortableFlow() }
                Button("Use for training") { trainingSelection = true }
                    .disabled(!store.resultsCurrent || store.isBusy)
                    .popover(isPresented: $trainingSelection) { PreparedTrainingView(preparation: store) }
                Button("Undo") { store.undo() }.disabled(store.isBusy || !store.history.canUndo)
                Button("Redo") { store.redo() }.disabled(store.isBusy || !store.history.canRedo)
                Spacer()
                if store.isBusy || store.isInspectingSource { Button("Cancel") { store.cancelPreparation() } }
                if !store.isBusy { Button("Prepare data") { store.runPreparation() }.disabled(!store.validation.isValid) }
            }.buttonStyle(.borderless).font(.system(size: 11)).padding(12).background(MLTheme.panel)
            DataFlowCacheView(store: store)
            if let provenance = store.resultProvenance {
                Text(provenance).font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 12).padding(.vertical, 6)
            }
            DataFlowFailureView(store: store)
            Divider()
            DataFlowEditorView(store:store)
            HStack {
                Text(store.statusMessage).font(.system(size: 11)).lineLimit(2)
                Spacer()
                if let manifest = store.manifestURL { Button("Artifacts") { NSWorkspace.shared.activateFileViewerSelecting([manifest]) }.font(.caption) }
            }.padding(10).background(MLTheme.panel)
        }.background(MLTheme.background)
        .onAppear { store.inspectionClosed = false }
        .onDisappear { store.closeSourceInspection(); if store.isBusy { store.cancelPreparation() }; store.save() }
    }
}
