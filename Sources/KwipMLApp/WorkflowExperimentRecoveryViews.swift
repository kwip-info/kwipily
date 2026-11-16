import SwiftUI

struct WorkflowExperimentRecoveryControls:View {
    let scope:WorkflowExperimentRecoveryStore.Scope
    @StateObject private var store:WorkflowExperimentRecoveryStore
    init(scope:WorkflowExperimentRecoveryStore.Scope,isCurrent:@escaping ()->Bool,onRefresh:@escaping ()->Void) {
        self.scope=scope
        _store=StateObject(wrappedValue:WorkflowExperimentRecoveryStore(
            isCurrent:{$0 == scope && isCurrent()},onRefresh:{value in if value == scope {onRefresh()}}))
    }
    var body:some View {
        VStack(alignment:.leading,spacing:6) {
            HStack(spacing:8) {
                Text(title).font(.caption.bold())
                Spacer(minLength:8)
                if store.isBusy {
                    ProgressView().controlSize(.small)
                    Button(store.state == .cancelling ? "Cancelling…":(store.state == .recovering ? "Cancel recovery":"Cancel check")) {store.cancel()}
                        .buttonStyle(FlatButtonStyle()).disabled(store.state == .cancelling)
                        .accessibilityLabel(store.state == .recovering ? "Cancel recovery":"Cancel recovery check")
                } else {
                    if store.state == .recoverable {
                        Button("Recover interrupted group") {store.recover()}.buttonStyle(FlatButtonStyle())
                    }
                    Button("Check again") {store.refresh()}.buttonStyle(FlatButtonStyle())
                        .accessibilityLabel("Check experiment recovery availability")
                }
            }
            Text(explanation).font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
        }.padding(10).background(MLTheme.panel)
            .task(id:scope) {store.select(scope:scope)}
            .onDisappear {store.invalidate()}
    }
    private var title:String {
        switch store.state {
        case .idle:return "Recovery check stopped"
        case .checking:return "Checking recovery availability…"
        case .inUse:return "In use · recovery unavailable"
        case .recoverable:return "Interrupted work can be reconciled"
        case .recovering:return "Reconciling saved experiment evidence…"
        case .cancelling:return "Waiting for the recovery check to finish…"
        case .recovered:return "Recovery recorded"
        case .terminal:return "No unfinished variants"
        case .unavailable:return "Recovery unavailable"
        }
    }
    private var explanation:String {
        switch store.state {
        case .inUse:return "An app operation or worker still holds this group. Check again after it releases ownership."
        case .recoverable:return "Recovery verifies saved outcomes, marks unverified work interrupted, and cancels unstarted variants. It preserves files and does not restart training."
        case .recovering,.cancelling:return "Recovery holds exclusive ownership while checking and recording results. Cancellation may follow a saved change; History will reload the actual record."
        case .recovered:return "The saved group was reloaded. Review each variant and its recovery details below."
        case .terminal:return "Every variant has a terminal recorded state. Review the independent evidence below before treating a model as completed."
        case .unavailable(let reason):return reason + " Recovery cannot modify this group. You can still inspect any independently verified runs below."
        case .idle:return "Check again to inspect ownership and saved evidence."
        case .checking:return "Checking saved protocol evidence and ownership. Opening this view does not change the group."
        }
    }
}

struct WorkflowExperimentRecoveryReceiptView:View {
    let receipt:WorkflowExperimentRecoveryReceipt
    var body:some View {
        DisclosureGroup("Recovery details") {
            VStack(alignment:.leading,spacing:10) {
                Text(Date(timeIntervalSince1970:receipt.timestampUnixSeconds),format:.dateTime.month().day().hour().minute().second())
                    .font(.caption2).foregroundStyle(MLTheme.muted)
                Text("Reconciled ledger revision \(receipt.priorRevision).").font(.caption2).foregroundStyle(MLTheme.muted)
                ForEach(receipt.variants,id:\.ordinal) {note in
                    VStack(alignment:.leading,spacing:4) {
                        Text("Variant \(note.ordinal) · \(note.priorStatus.rawValue) → \(note.status.rawValue)").font(.caption.bold())
                        Text(reason(note.reason)).font(.caption)
                        Text(note.detail).font(.caption2).textSelection(.enabled)
                        if note.reason == .terminalEvidenceUnavailable {
                            Text("Next: inspect the saved run and its evidence. A new run is a separate action; recovery did not resume this model.")
                                .font(.caption2).foregroundStyle(MLTheme.muted)
                        }
                    }.frame(maxWidth:.infinity,alignment:.leading)
                }
            }.padding(.top,8)
        }.font(.caption).padding(10).background(MLTheme.panel)
    }
    private func reason(_ value:WorkflowExperimentRecoveryReceipt.Reason)->String {
        switch value {
        case .pendingCancelled:return "Unstarted variant cancelled"
        case .verifiedOutcomeRecovered:return "Saved terminal outcome independently verified"
        case .terminalEvidenceUnavailable:return "No terminal outcome could be verified"
        }
    }
}
