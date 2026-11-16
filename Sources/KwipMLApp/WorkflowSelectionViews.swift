import SwiftUI
import Combine
import KwipMLCore

@MainActor private func selectionNode(_ address: WorkflowNodeAddress, canvas: WorkflowCanvasAdapter) -> GraphNode? {
    guard let id = canvas.projection.displayIDByAddress[address] else { return nil }
    return canvas.graph.nodes.first { $0.id == id }
}
@MainActor private func selectionTitle(_ address: WorkflowNodeAddress, canvas: WorkflowCanvasAdapter) -> String {
    guard let node = selectionNode(address, canvas: canvas) else { return address.domain.rawValue + ":" + address.localID }
    return node.name.isEmpty ? canvas.registry.component(type: node.type)?.title ?? node.type : node.name
}

struct WorkflowSelectionInspectorView: View {
    @ObservedObject var owner: WorkflowAuthoringStore
    @ObservedObject var canvas: WorkflowCanvasAdapter
    var workspace: WorkspaceStore?
    var inspectComponent: ((WorkflowNodeAddress) -> Void)?
    @State private var title = ""
    @State private var error: String?
    private var global: Bool { owner.selection.nodeAddresses.isEmpty && owner.selection.edgeIDs.isEmpty }
    private var selectedNodes: [WorkflowNode] { owner.document.nodes.filter { owner.selection.nodeAddresses.contains($0.address) } }
    private var internalCount: Int { owner.document.edges.filter { owner.selection.nodeAddresses.contains($0.from.address) && owner.selection.nodeAddresses.contains($0.to.address) }.count }
    var body: some View {
        ThemedScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Label(global ? "Workflow" : "\(selectedNodes.count + owner.selection.edgeIDs.count) selected", systemImage: global ? "point.3.connected.trianglepath.dotted" : "square.on.square")
                    .font(.system(size: 16, weight: .semibold)).accessibilityAddTraits(.isHeader)
                if global {
                    TextField("Workflow name", text: $title).flatField().onSubmit(rename)
                        .disabled(canvas.isBusy || owner.hasPendingInspectorEdits).accessibilityLabel("Workflow name")
                    if title != owner.document.name { Button("Rename workflow", action: rename).buttonStyle(FlatButtonStyle()).disabled(canvas.isBusy || owner.hasPendingInspectorEdits) }
                    Text("\(owner.document.nodes.count) components · \(owner.document.edges.count) \(owner.document.edges.count == 1 ? "connection" : "connections")").font(.caption).foregroundStyle(MLTheme.muted)
                    if let workspace { WorkflowGlobalWorkingDataView(workspace: workspace) }
                    InspectorSection("Navigate", symbol: "scope", identifier: "workflow.global.navigate") {
                        ForEach(owner.document.nodes.filter(isNavigationNode)) { node in nodeButton(node) }
                        if owner.document.nodes.isEmpty { Text("Add a component to begin.").font(.caption).foregroundStyle(MLTheme.muted) }
                    }
                    Button { canvas.fitWorkflow() } label: { Label("Fit workflow", systemImage: "arrow.up.left.and.arrow.down.right") }
                        .buttonStyle(FlatButtonStyle()).disabled(owner.document.nodes.isEmpty)
                    if let workspace { Button("Workflow settings") { workspace.activeSection = "Settings" }.buttonStyle(FlatButtonStyle()) }
                } else {
                    Text(selectionCounts).font(.caption).foregroundStyle(MLTheme.muted)
                    HStack(spacing: 8) { selectionActions }
                    InspectorSection("Selected components", symbol: "square.stack", identifier: "workflow.selection.components") {
                        ForEach(selectedNodes) { node in nodeButton(node) }
                        if selectedNodes.isEmpty { Text("No components selected.").font(.caption).foregroundStyle(MLTheme.muted) }
                    }
                    if !owner.selection.edgeIDs.isEmpty {
                        InspectorSection("Selected connections", symbol: "link", identifier: "workflow.selection.connections") {
                            ForEach(owner.document.edges.filter { owner.selection.edgeIDs.contains($0.id) }) { edge in
                                Text(selectionTitle(edge.from.address, canvas: canvas) + " · " + edge.from.port + " → " + selectionTitle(edge.to.address, canvas: canvas) + " · " + edge.to.port)
                                    .font(.caption).textSelection(.enabled)
                            }
                        }
                    }
                }
                if let error { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
                InspectorInformationSection {
                    Text(global ? "Working data is a source-order development sample. View, file outputs and training request complete upstream data within source settings and resource limits. File Input samples files before extraction; one file may emit several rows. Percentage can require a source count. Canvas navigation and naming do not recompute data." : "Inspect one selected component for its own configuration and output. Duplicate copies the selected components and their internal connections. Delete removes only the explicit selection and incident connections; Undo restores the edit. Selection Results keeps each component's captured output separate.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
            }.padding(14)
        }
        .onAppear { title = owner.document.name }
        .onChange(of: owner.document.name) { _, name in title = name }
    }
    private func rename() {
        guard !canvas.isBusy, !owner.hasPendingInspectorEdits else { return }
        if owner.rename(title, lease: owner.lease()) { error = nil } else { error = owner.error }
    }
    private func isNavigationNode(_ node: WorkflowNode) -> Bool {
        switch node.payload {
        case .dataOperator(let id, _): return DataFlowSourceOperators.contains(id) || id == "kwip.data.view" || id == "kwip.data.outputDelimited"
        case .runSetup, .train, .estimator, .modelDefinition, .evaluateTest: return true
        default: return false
        }
    }
    private func nodeButton(_ node: WorkflowNode) -> some View {
        Button {
            guard let id = canvas.projection.displayIDByAddress[node.address] else { return }
            if let inspectComponent { inspectComponent(node.address) } else { canvas.selectNode(id, extending: false) }; canvas.revealSelection()
        } label: {
            HStack(spacing: 8) {
                if let projected = selectionNode(node.address, canvas: canvas) {
                    ComponentIcon(type: canvas.canvasIconType(nodeID: projected.id), size: 24)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(selectionTitle(node.address, canvas: canvas)).lineLimit(1)
                        Text(canvas.registry.component(type: projected.type)?.title ?? projected.type).font(.caption2).foregroundStyle(MLTheme.muted).lineLimit(1)
                    }
                }
                Spacer(minLength: 4); Image(systemName: "chevron.right").font(.caption2)
            }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
        }.buttonStyle(.plain).disabled(canvas.isBusy).accessibilityLabel("Select " + selectionTitle(node.address, canvas: canvas))
    }
    private var selectionCounts: String {
        var values = ["\(selectedNodes.count) \(selectedNodes.count == 1 ? "component" : "components")"]
        if internalCount > 0 { values.append("\(internalCount) internal \(internalCount == 1 ? "connection" : "connections")") }
        if !owner.selection.edgeIDs.isEmpty { values.append("\(owner.selection.edgeIDs.count) selected \(owner.selection.edgeIDs.count == 1 ? "connection" : "connections")") }
        return values.joined(separator: " · ")
    }
    private func actionTile(_ title: String, symbol: String, color: Color = .primary) -> some View {
        VStack(spacing: 7) {
            Image(systemName: symbol).font(.system(size: 17))
            Text(title).font(.system(size: 10, weight: .medium)).lineLimit(1)
        }.foregroundStyle(color).frame(maxWidth: .infinity, minHeight: 56)
            .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
            .contentShape(Rectangle())
    }
    @ViewBuilder private var selectionActions: some View {
        Button { canvas.fitSelection() } label: { actionTile("Fit selection", symbol: "scope") }
            .buttonStyle(.plain).disabled(selectedNodes.isEmpty && owner.selection.edgeIDs.isEmpty)
        Button { canvas.duplicateSelection() } label: { actionTile("Duplicate", symbol: "doc.on.doc") }
            .buttonStyle(.plain).disabled(canvas.isBusy || owner.hasPendingInspectorEdits || selectedNodes.isEmpty)
        Button(role: .destructive) { canvas.deleteSelection() } label: { actionTile("Delete", symbol: "trash", color: MLTheme.rose) }
            .buttonStyle(.plain).disabled(canvas.isBusy || owner.hasPendingInspectorEdits)
    }
}

private struct WorkflowGlobalWorkingDataView: View {
    @ObservedObject var workspace: WorkspaceStore
    var body: some View {
        InspectorSection("Working data", symbol: "tablecells", identifier: "workflow.global.sampling") {
            PreparationSamplingControl(store: workspace, compact: true)
        }
    }
}

struct WorkflowMultipleResultsView: View {
    @ObservedObject var owner: WorkflowAuthoringStore
    @ObservedObject var canvas: WorkflowCanvasAdapter
    var outputs: (any WorkflowDataOutputs)?
    var host: WorkflowWorkspaceSession?
    let tab: String
    let inspectComponent: (WorkflowNodeAddress) -> Void
    let inspectConnection: (WorkflowEdge) -> Void
    @State private var outputRevision = 0
    private var nodes: [WorkflowNode] { owner.document.nodes.filter { owner.selection.nodeAddresses.contains($0.address) } }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if tab == "connections" { connections }
            else if tab == "diagnostics" { diagnostics }
            else { components }
        }.font(.system(size: 11))
        .animation(nil, value: outputRevision)
        .onReceive((outputs?.changes ?? Empty<Void, Never>().eraseToAnyPublisher()).throttle(for: .milliseconds(100), scheduler: RunLoop.main, latest: true)) { _ in outputRevision &+= 1 }
    }
    private func snapshots() -> [WorkflowNodeAddress: WorkflowCapturedDataOutput] {
        guard let outputs, outputs.projectID == owner.projectID, outputs.sessionID == owner.sessionID else { return [:] }
        return outputs.snapshots(for: owner.selection.nodeAddresses).filter { address, snapshot in
            WorkflowSelectionPresentation.owned(snapshot, projectID: owner.projectID, sessionID: owner.sessionID, address: address) != nil
        }
    }
    private func captureColor(_ snapshot: WorkflowCapturedDataOutput?) -> Color {
        switch WorkflowSelectionPresentation.captureState(snapshot) {
        case "Current": return ComponentStyle.tint("kwip.data.view")
        case "Stale", "Partial capture": return .orange
        default: return MLTheme.muted
        }
    }
    private var components: some View {
        let captures = snapshots()
        let projectedNodes = Dictionary(uniqueKeysWithValues: canvas.graph.nodes.map { ($0.id, $0) })
        return LazyVStack(spacing: 0) {
            HStack { Text("Component").frame(maxWidth: .infinity, alignment: .leading); Text("Tool").frame(maxWidth: .infinity, alignment: .leading); Text("Captured output").frame(maxWidth: .infinity, alignment: .leading); Text("State").frame(width: 100, alignment: .leading); Text("Action").frame(width: 66) }.fontWeight(.semibold).padding(8).background(MLTheme.background)
            ForEach(nodes) { node in
                let captured = captures[node.address]
                HStack(spacing: 8) {
                    if let id = canvas.projection.displayIDByAddress[node.address], let projected = projectedNodes[id] {
                        ComponentIcon(type: canvas.canvasIconType(nodeID: projected.id), size: 20)
                        Text(projected.name.isEmpty ? canvas.registry.component(type: projected.type)?.title ?? projected.type : projected.name).frame(maxWidth: .infinity, alignment: .leading).lineLimit(1)
                        Text(canvas.registry.component(type: projected.type)?.title ?? projected.type).frame(maxWidth: .infinity, alignment: .leading).lineLimit(1)
                    }
                    Text(WorkflowSelectionPresentation.outputDescription(captured)).frame(maxWidth: .infinity, alignment: .leading).lineLimit(1).help(WorkflowSelectionPresentation.outputDescription(captured))
                    Label(WorkflowSelectionPresentation.captureState(captured), systemImage: WorkflowSelectionPresentation.captureState(captured) == "Current" ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(captureColor(captured)).frame(width: 100, alignment: .leading)
                    Button("Inspect") { inspectComponent(node.address) }.buttonStyle(FlatButtonStyle()).frame(width: 66).disabled(canvas.isBusy).accessibilityLabel("Inspect selected " + (node.name.isEmpty ? node.address.localID : node.name))
                }.padding(.horizontal, 8).padding(.vertical, 5)
                Divider()
            }
            if nodes.isEmpty { Text("No components in this selection.").foregroundStyle(MLTheme.muted).padding(10) }
        }.background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private var connections: some View {
        VStack(spacing: 0) {
            HStack { Text("From").frame(maxWidth: .infinity, alignment: .leading); Text("To").frame(maxWidth: .infinity, alignment: .leading); Text("Kind").frame(width: 85, alignment: .leading); Text("Scope").frame(width: 85, alignment: .leading); Text("Action").frame(width: 66) }.fontWeight(.semibold).padding(8).background(MLTheme.background)
            ForEach(WorkflowSelectionPresentation.connections(in: owner.document, selection: owner.selection)) { edge in
                HStack(spacing: 8) {
                    Text(selectionTitle(edge.from.address, canvas: canvas) + " · " + edge.from.port).frame(maxWidth: .infinity, alignment: .leading).lineLimit(1)
                    Text(selectionTitle(edge.to.address, canvas: canvas) + " · " + edge.to.port).frame(maxWidth: .infinity, alignment: .leading).lineLimit(1)
                    Text(edge.kind.rawValue).frame(width: 85, alignment: .leading)
                    Text(WorkflowSelectionPresentation.relation(edge, selection: owner.selection)).frame(width: 85, alignment: .leading)
                    Button("Inspect") { inspectConnection(edge) }.buttonStyle(FlatButtonStyle()).frame(width: 66).disabled(canvas.isBusy).accessibilityLabel("Inspect selected connection " + edge.id)
                }.padding(.horizontal, 8).padding(.vertical, 5); Divider()
            }
        }.background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
    }
    @ViewBuilder private var diagnostics: some View {
        let ids = Set(nodes.compactMap { canvas.projection.displayIDByAddress[$0.address] })
        let issues = WorkflowSelectionPresentation.configurationIssues(canvas.validation.issues, displayIDs: ids)
        if !issues.isEmpty {
            Text("Current configuration").fontWeight(.semibold)
            ForEach(issues) { issue in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle").foregroundStyle(Color.orange)
                    Text(canvas.graph.nodes.first { $0.id == issue.nodeID }.map { $0.name.isEmpty ? $0.type : $0.name } ?? "Component").frame(width: 160, alignment: .leading)
                    Text(issue.message).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }.padding(8).overlay(Rectangle().stroke(MLTheme.rule))
            }
        }
        let failure = host?.attributedPreparationFailure
        let hasFailure = failure.map { evidence in nodes.contains { evidence.matches($0.address) } } ?? false
        if hasFailure, let failure {
            WorkflowPreparationFailureView(evidence: failure, onSelect: inspectComponent)
        }
        if let execution = host?.currentExecution {
            WorkflowSelectionDiagnosticsView(console: execution.console, selection: owner.selection, canvas: canvas, showsEmptyState: issues.isEmpty && !hasFailure).id(execution.runID)
        } else if issues.isEmpty && !hasFailure { Text("No attributed diagnostic evidence for this selection.").foregroundStyle(MLTheme.muted) }
    }
}

private struct WorkflowSelectionDiagnosticsView: View {
    @ObservedObject var console: WorkflowRunConsole
    let selection: WorkflowSelection
    let canvas: WorkflowCanvasAdapter
    var showsEmptyState = true
    var body: some View {
        let entries = WorkflowSelectionPresentation.diagnostics(console.entries, selection: selection)
        VStack(alignment: .leading, spacing: 8) {
            if entries.isEmpty && showsEmptyState { Text("No attributed diagnostic evidence for this selection.").foregroundStyle(MLTheme.muted) }
            ForEach(entries) { entry in
                DisclosureGroup {
                    Text(entry.message).font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                } label: {
                    HStack {
                        Text(entry.component.map { selectionTitle($0, canvas: canvas) } ?? "").frame(width: 160, alignment: .leading).lineLimit(1)
                        Text(entry.level.rawValue.capitalized).foregroundStyle(entry.level == .error ? MLTheme.rose : Color.orange).frame(width: 70, alignment: .leading)
                        Text(entry.message).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }.padding(8).overlay(Rectangle().stroke(MLTheme.rule))
            }
            DisclosureGroup("Information") { Text("Only warnings and errors explicitly attributed to these components are listed. Run-wide findings remain in Global results. Missing attributed evidence does not establish component health.").foregroundStyle(MLTheme.muted) }
        }
    }
}
