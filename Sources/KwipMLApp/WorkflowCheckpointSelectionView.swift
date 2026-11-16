import SwiftUI
import KwipMLCore

struct WorkflowCheckpointSelectionView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var history:WorkflowHistoryStore
    var inlinePresentation=false
    @State private var choosing=false
    @State private var details:WorkflowWorkspaceSession.CheckpointDetails?
    @State private var detailsError:String?
    private var active:Bool {
        context.isCurrent && !context.isBusy && !host.isStagingCheckpoint && !host.isRetired
            && host.owner === context.authority.owner && history.isActive && history.sessionID == host.sessionID
    }
    private var selected:WorkflowCheckpointReference? {host.checkpointBindings.reference(for:context.authority.address)}
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack(spacing: 8) {
                if !inlinePresentation,let image = ComponentArtworkCatalog.checkpointLifecycleImage {
                    Image(nsImage: image).resizable().interpolation(.high).scaledToFit()
                        .frame(width: 24, height: 24).accessibilityHidden(true)
                }
                Text("Parent checkpoint").font(.caption.weight(.semibold))
            }
            if context.trainingMode != .fresh, let reason = host.checkpointRestoreUnavailableReason {
                Text(reason).font(.caption).foregroundStyle(MLTheme.rose)
            }
            if let selected {
                Text(inlinePresentation ? (context.trainingMode == .fresh ? "Retained · unused in fresh mode":"Selected checkpoint"):"Run \(selected.runID.prefix(12))").font(.caption).textSelection(.enabled)
                if let details {
                    Text("\(details.device.uppercased()) · \(details.completedEpoch) completed epochs").font(.caption)
                    if context.trainingMode == .exactResume {
                        Text(details.device == "cpu" ? "Choose CPU and more than \(details.completedEpoch) total epochs in Train’s Run settings.":"This parent cannot resume exactly. Choose a CPU checkpoint or use Fine-tune weights.")
                            .font(.caption2).foregroundStyle(details.device == "cpu" ? MLTheme.muted:MLTheme.rose)
                    }
                } else if let detailsError {
                    Text(detailsError).font(.caption2).foregroundStyle(MLTheme.rose)
                } else {Text("Verifying parent details…").font(.caption2).foregroundStyle(MLTheme.muted)}
                if !inlinePresentation {Text(context.trainingMode == .fresh ? "Retained selection; fresh training does not use this checkpoint.":"The parent will be verified again before training.").font(.caption2).foregroundStyle(MLTheme.muted)}
            } else {
                Text(inlinePresentation ? "No checkpoint selected":context.trainingMode == .fresh ? "Fresh training starts with new weights.":"Select a completed run before resuming or fine-tuning.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            HStack {
                Button("Choose checkpoint") {choosing=true;history.refresh()}.buttonStyle(FlatButtonStyle())
                if selected != nil {
                    Button("Clear checkpoint") {_=host.clearCheckpointParent(for:context.authority.address,lease:context.authority.token)}
                        .buttonStyle(FlatButtonStyle())
                }
            }.disabled(!active)
        }
        .task(id:selected) {
            details=nil;detailsError=nil
            guard let reference=selected else {return}
            do {
                let value=try await host.checkpointDetails(reference)
                guard !Task.isCancelled,selected == reference,context.isCurrent else {return}
                details=value
            } catch {
                guard !Task.isCancelled,selected == reference,context.isCurrent else {return}
                detailsError="Parent details unavailable: "+error.localizedDescription
            }
        }
        .sheet(isPresented:$choosing) {
            WorkflowHistoryPicker(title:"Choose parent checkpoint",
                explanation:"Only completed runs are listed. Selection verifies saved artifacts; compatibility with your current architecture and data is checked when running.",
                emptyMessage:"No matching completed runs. Finish a fresh training run to create a checkpoint.",
                history:history,active:active,completedOnly:true,status:host.statusMessage,onClose:{choosing=false}) {entry in
                    guard active else {return false}
                    return await host.selectCheckpointParent(reference:entry.reference,for:context.authority.address,lease:context.authority.token)
                }
        }
        .onChange(of:context.isCurrent) {_,current in if !current {choosing=false}}
    }
}
