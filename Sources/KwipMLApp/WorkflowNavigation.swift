import SwiftUI
import KwipMLCore

extension WorkspaceStore {
    var projectSections: [String] {
        if isLearningLessonActive { return [] }
        return workflowSession == nil ? ["Data", "Build", "Run", "Code", "Experiments", "Settings"] : ["Build", "History", "Code", "Settings"]
    }
    var currentRunConfigurationFinding:WorkflowModelValidationIssue? {workflowSession?.canvas.currentRunConfigurationFinding}
    var isCheckingRunConfiguration:Bool {
        guard let host=workflowSession else {return false}
        return host.canvas.isInferringTensors && host.owner.document.nodes.contains {$0.address.domain == .model}
    }
    @discardableResult func revealRunConfigurationFinding()->Bool {
        guard let host=workflowSession,let finding=currentRunConfigurationFinding,!isBusy,
              let id=host.canvas.projection.displayIDByAddress[finding.address] else {return false}
        host.canvas.selectNode(id,extending:false);host.canvas.revealSelection()
        activeSection="Build";showInspector=true
        return true
    }
    /// Cheap presentation readiness only. The selected terminal compiler remains
    /// authoritative when Run is pressed, including incomplete connection errors.
    var canRunCurrentFlow: Bool {
        guard !isBusy, activeProjectID != nil, !isLearningLessonActive else { return false }
        guard let host = workflowSession else { return validation.isValid }
        return runtime != nil && currentRunConfigurationFinding == nil && !isCheckingRunConfiguration && !host.isRetired && host.owner.projectID == activeProjectID
            && !host.owner.isBusy && !host.owner.hasPendingInspectorEdits
            && host.owner.document.drafts.isEmpty && host.owner.document.nodes.contains { node in
                switch node.payload { case .gameSession, .train, .estimator, .dataOperator, .saveModel, .evaluateTest, .generateText: return true; default: return false }
            }
    }
}

struct WorkflowStatusView: View {
    @ObservedObject var host: WorkflowWorkspaceSession
    var body: some View {
        HStack(spacing: 9) {
            Circle().fill(MLTheme.muted).frame(width: 6, height: 6)
            Text("Workflow · \(host.canvas.graph.nodes.count) components")
            Divider().frame(height: 12)
            Text(host.displayedStatusMessage).foregroundStyle(MLTheme.muted).lineLimit(1)
            Spacer()
            Text(host.owner.document.nodes.contains { if case .estimator = $0.payload { return true }; return false }
                 ? "Local · estimator backend" : "Local · device configured in Train").foregroundStyle(MLTheme.muted)
        }
    }
}
