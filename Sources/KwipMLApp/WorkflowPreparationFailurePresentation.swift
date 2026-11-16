import SwiftUI
import KwipMLCore

/// Captured worker evidence. Currentness is separate from attribution and raw activity.
struct WorkflowPreparationFailureEvidence: Equatable {
    let detail: DataFlowFailureDetail
    let runID: UUID
    let capturedRevision: UInt64
    let isCurrent: Bool
    var address: WorkflowNodeAddress { .init(domain: .data, localID: detail.nodeID) }
    func matches(_ address: WorkflowNodeAddress) -> Bool { self.address == address }
    func currentCause(for address: WorkflowNodeAddress) -> String? { isCurrent && matches(address) ? detail.cause : nil }
}

extension WorkflowWorkspaceSession {
    var attributedPreparationFailure: WorkflowPreparationFailureEvidence? {
        guard !isRetired else { return nil }
        if let failure=sourceProvider.apiFailureEvidence {return failure}
        if let session = currentDataPreparation, session.phase == .failed, let detail = session.preparationFailure {
            return .init(detail: detail, runID: session.runID, capturedRevision: session.document.revision,
                         isCurrent: session.preparationFailureIsCurrent)
        }
        if let session = currentExecution, session.phase == .failed, let detail = session.preparationFailure {
            return .init(detail: detail, runID: session.runID, capturedRevision: session.request.document.revision,
                         isCurrent: session.preparationFailureIsCurrent)
        }
        if let session = currentBackendFit, session.phase == .failed, let detail = session.preparationFailure {
            return .init(detail:detail,runID:session.runID,capturedRevision:session.document.revision,
                         isCurrent:session.preparationFailureIsCurrent)
        }
        return nil
    }
}

struct WorkflowPreparationFailureView: View {
    let evidence: WorkflowPreparationFailureEvidence
    var onSelect: ((WorkflowNodeAddress) -> Void)? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.circle.fill").foregroundStyle(evidence.isCurrent ? Color.red : Color.orange)
                Text(evidence.detail.nodeLabel ?? evidence.detail.componentTitle).fontWeight(.semibold)
                Text(evidence.isCurrent ? "Run failed" : "Previous run · inputs or workflow changed").font(.caption).foregroundStyle(MLTheme.muted)
                Spacer(minLength: 4)
                if let onSelect { Button("Show component") { onSelect(evidence.address) }.buttonStyle(FlatButtonStyle()) }
            }
            Text(evidence.detail.cause).font(.system(size: 12)).textSelection(.enabled)
            HStack(spacing: 14) {
                Label(evidence.detail.componentTitle, systemImage: "square.stack.3d.up")
                if let count = evidence.detail.inputRowCount { Text("\(count) input rows") }
                Text("Run \(evidence.runID.uuidString.prefix(8)) · revision \(evidence.capturedRevision)")
            }.font(.caption2).foregroundStyle(MLTheme.muted)
            VStack(alignment: .leading, spacing: 4) {
                Text("Check next").font(.caption.weight(.semibold))
                Text(evidence.detail.suggestion).font(.caption).textSelection(.enabled)
            }
        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
            .background(MLTheme.background)
            .overlay(Rectangle().stroke((evidence.isCurrent ? Color.red : Color.orange).opacity(0.45)))
    }
}
