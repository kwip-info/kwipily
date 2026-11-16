import SwiftUI

struct CheckpointView:View {
    @ObservedObject var store:WorkspaceStore
    private var parent: CheckpointParent? { store.verifiedCheckpointParent }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            SectionLabel(title:"Continue a completed run")
            if let parent {
                Text("Parent \(parent.runID.prefix(8)) · epoch \(parent.completedEpoch) · \(parent.device.uppercased())").font(.caption).textSelection(.enabled)
                Text("Exact resume restores weights, Adam state, random state and history. Target total epochs must exceed the saved epoch. CPU and matching runtime, architecture, data and settings are required.").font(.caption).foregroundStyle(MLTheme.muted)
                if let reason=store.checkpointCompatibility(CheckpointContinuation(parent:parent,mode:.exactResume)) {Text(reason).font(.caption).foregroundStyle(.orange)}
                HStack {
                    Button("Resume exact to epoch \(store.epochCount)"){store.run(continuation:CheckpointContinuation(parent:parent,mode:.exactResume))}
                        .disabled(store.isBusy || store.checkpointCompatibility(CheckpointContinuation(parent:parent,mode:.exactResume)) != nil)
                    Button("Fine-tune for \(store.epochCount) epochs"){store.run(continuation:CheckpointContinuation(parent:parent,mode:.fineTune))}
                        .disabled(store.isBusy || store.checkpointCompatibility(CheckpointContinuation(parent:parent,mode:.fineTune)) != nil)
                    Button("Run settings"){store.activeSection="Settings"}
                }.buttonStyle(FlatButtonStyle())
                Text("Fine-tune restores only weights and starts a fresh optimizer, random state and epoch count. You may change data and training settings. Both actions create new artifacts and preserve this parent.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {Text("Complete a run to select its verified checkpoint. Imported checkpoints and older checkpoint formats are not supported here yet.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let request = store.activeTrainingRequest {
                Text(request.description).font(.caption).foregroundStyle(MLTheme.teal).textSelection(.enabled)
            }
            if let parent {
                Text("Last verified completed run: \(parent.lineageMode)" + (parent.parentRunID.map { " · parent \($0.prefix(8))" } ?? "")).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            }
            if let error = store.checkpointPresentationError { Text(error).font(.caption).foregroundStyle(.orange) }
        }.onAppear { store.refreshVerifiedCheckpointParent() }
    }
}
