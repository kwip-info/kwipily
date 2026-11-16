import SwiftUI
import KwipMLCore

/// Live operation evidence and configuration errors are independent of cached
/// output labels. A busy canvas never implies every displayed node is executing.
struct CanvasNodeFeedback: Equatable {
    let issueMessages: [String]
    let isExecuting: Bool
    let executionFailure: String?
    init(nodeID: String, issues: [ValidationIssue], isExecuting: Bool, executionFailure: String? = nil) {
        issueMessages = issues.filter { $0.nodeID == nodeID }.map(\.message)
        self.isExecuting = isExecuting; self.executionFailure = executionFailure
    }
    var isMisconfigured: Bool { !issueMessages.isEmpty }
    var accessibilityDescription: String {
        var parts: [String] = []
        if isExecuting { parts.append("Executing this component's operation") }
        if isMisconfigured { parts.append("Configuration needs attention"); parts += issueMessages }
        if let executionFailure { parts.append("Run failed"); parts.append(executionFailure) }
        return parts.joined(separator: ". ")
    }
    func shouldAnimate(reduceMotion: Bool) -> Bool { isExecuting && !reduceMotion }
    static func connectionIssueMessages(port: WorkflowResolvedPort, inboundCount: Int) -> [String] {
        if port.required && inboundCount == 0 { return ["Connect required input \(port.id)."] }
        if let maximum = port.maximumConnections, inboundCount > maximum {
            return ["Input \(port.id) accepts at most \(maximum) connection\(maximum == 1 ? "" : "s")."]
        }
        return []
    }
    /// Reuse the projection's already resolved typed contracts. Output ports do
    /// not require consumers; optional inputs and unused configured sources are
    /// valid authoring states. No mixed-domain GraphValidator is invoked.
    static func connectionIssues(nodeID: String, document: WorkflowDocument, projection: WorkflowCanvasProjection) -> [ValidationIssue] {
        guard let address = projection.addressByDisplayID[nodeID],
              let node = projection.graph.nodes.first(where: { $0.id == nodeID }),
              let definition = projection.registry.component(type: node.type) else { return [] }
        let inbound = (document.edges + document.inlineInternalEdges).filter { $0.to.address == address }
        let boundary: CompositeDefinition? = (try? WorkflowInlineContainer.parse(in: document, address: address))
            .flatMap { try? WorkflowInlineContainer.definition(in: document, root: $0.root) }
        let childID = (try? WorkflowInlineContainer.parse(in: document, address: address))?.childID
        return definition.inputs.flatMap { portID -> [ValidationIssue] in
            guard let port = projection.portDescriptor(address: address, port: portID, direction: .input)?.contract else { return [] }
            let exposed = boundary?.inputs.contains { input in
                input.targets.contains { $0.componentID == childID && $0.portID == portID }
            } == true
            let count = inbound.filter { $0.to.port == portID }.count + (exposed ? 1 : 0)
            return connectionIssueMessages(port: port, inboundCount: count).enumerated().map { index, message in
                .init(id: "canvas-input-\(nodeID)-\(portID)-\(index)", message: message, nodeID: nodeID)
            }
        }
    }
    static func isExecuting(phase: WorkflowExecutionPhase, matchesTerminal: Bool, isCurrent: Bool) -> Bool {
        phase == .training && matchesTerminal && isCurrent
    }
}

extension CanvasWorkspace {
    /// Actual capture/preparation boundaries drive activity; model layers require
    /// their own forward-operation telemetry rather than an invented sweep.
    func canvasNodeFeedback(for nodeID: String) -> CanvasNodeFeedback {
        var executing = false
        var issues = validation.issues
        if let adapter = self as? WorkflowCanvasAdapter {
            issues += adapter.connectionFeedbackIssues(for:nodeID)
        }
        if let adapter = self as? WorkflowCanvasAdapter,
           let host = adapter.outputProvider as? WorkflowWorkspaceSession,
           host.canvas === adapter, !host.isRetired,
           host.projectID == adapter.owner.projectID, host.sessionID == adapter.owner.sessionID,
           adapter.owner.isCurrent(adapter.owner.lease()),
           let address = adapter.projection.addressByDisplayID[nodeID] {
            executing = host.modelLoader.isBusy && host.modelLoader.activeAddress==address
            if let execution = host.execution {
                executing = WorkflowProgressPresentation.isActive(address, progress:execution.canvasProgress, isCurrent:execution.request.isCurrent(in:adapter.owner))
                if execution.request.isCurrent(in:adapter.owner),address.domain == .model {
                    executing = executing || execution.modelOperationProgress?.activeNode(in:[]) == address.localID
                }
            }
            if let capture=host.currentSourceCaptureProgress {
                executing = executing || WorkflowProgressPresentation.isActive(address,progress:capture,isCurrent:true)
            }
            if let preparation=host.predictionPreparation,preparation.isRunning {executing = executing || WorkflowProgressPresentation.isActive(address,progress:preparation.canvasProgress,isCurrent:preparation.document==adapter.owner.document)}
            if let fit = host.backendFit {
                executing = executing || CanvasNodeFeedback.isExecuting(phase: fit.phase,
                    matchesTerminal: fit.plan.estimator == address,
                    isCurrent: fit.document == adapter.owner.document && fit.bindingRevision == host.bindings.revision)
            }
        }
        let failure: String?
        if let adapter = self as? WorkflowCanvasAdapter, let address = adapter.projection.addressByDisplayID[nodeID],
           let host = adapter.outputProvider as? WorkflowWorkspaceSession, host.canvas === adapter {
            failure = host.modelLoader.failure(address) ?? host.attributedPreparationFailure?.currentCause(for: address)
        } else { failure = nil }
        return .init(nodeID: nodeID, issues: issues, isExecuting: executing, executionFailure: failure)
    }
}

/// Does not receive input, allocate a task per node, or repaint idle graphs.
/// Reduced motion retains a static blue perimeter rather than removing status.
struct CanvasNodeFeedbackOverlay: View {
    let feedback: CanvasNodeFeedback
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        ZStack(alignment: .topTrailing) {
            if feedback.isExecuting {
                RoundedRectangle(cornerRadius:10).stroke(Color.blue.opacity(0.6), lineWidth: 2)
                if feedback.shouldAnimate(reduceMotion: reduceMotion) {
                    TimelineView(.animation(minimumInterval: 1.0 / 30)) { context in
                        let phase = context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.6) / 1.6
                        RoundedRectangle(cornerRadius:10).stroke(Color.blue, style: StrokeStyle(lineWidth: 2, dash: [12, 7], dashPhase: CGFloat(phase * 19)))
                    }
                }
            }
            if feedback.isMisconfigured || feedback.executionFailure != nil {
                Image(systemName: "exclamationmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 15, height: 15)
                    .background(Circle().fill(Color.red))
                    .offset(x: 5, y: -5)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        .allowsHitTesting(false)
        // The containing node exposes status and issues once through its label/help.
        .accessibilityHidden(true)
    }
}
