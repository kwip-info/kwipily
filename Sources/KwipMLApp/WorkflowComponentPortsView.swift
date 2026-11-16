import SwiftUI
import KwipMLCore

/// A menu action retains the owner lease and selection that opened it.
@MainActor struct WorkflowResultsPortActions {
    let owner:WorkflowAuthoringStore
    let canvas:WorkflowCanvasAdapter
    let scope:WorkflowResultsScope
    let lease:WorkflowEditorLease
    private let selection:WorkflowSelection
    init(owner:WorkflowAuthoringStore,canvas:WorkflowCanvasAdapter,scope:WorkflowResultsScope) {
        self.owner=owner;self.canvas=canvas;self.scope=scope;lease=owner.lease();selection=owner.selection
    }
    var isCurrent:Bool {
        canvas.owner === owner && owner.isCurrent(lease) && owner.selection == selection &&
        WorkflowResultsScope.resolve(selection:owner.selection,document:owner.document) == scope
    }
    func canEditConnection(_ edge:WorkflowEdge)->Bool {canEdit && belongs(edge)}
    var canEdit:Bool {isCurrent && !canvas.isBusy && !owner.hasPendingInspectorEdits}
    private func refuse()->Bool {
        canvas.statusMessage="Connection not changed. Select its component or connection in the current idle canvas, and apply or discard pending edits."
        return false
    }
    private func belongs(_ edge:WorkflowEdge)->Bool {
        guard canvas.inlineEditableEdges.contains(edge) else {return false}
        switch scope {
        case .component(let address): return edge.from.address == address || edge.to.address == address
        case .connection(let selected): return selected == edge
        default:return false
        }
    }
    private func declared(_ endpoint:WorkflowEndpoint,direction:PortDirection)->Bool {
        guard let id=canvas.projection.displayIDByAddress[endpoint.address],
              let node=canvas.graph.nodes.first(where:{$0.id == id}),
              let definition=canvas.registry.component(type:node.type),
              (direction == .input ? definition.inputs:definition.outputs).contains(endpoint.port) else {return false}
        return canvas.projection.portDescriptor(address:endpoint.address,port:endpoint.port,direction:direction)?.contract != nil
    }
    func canInspect(_ endpoint:WorkflowEndpoint,direction:PortDirection)->Bool {
        guard isCurrent, declared(endpoint,direction:direction) else {return false}
        let included:Bool
        switch scope {
        case .component(let address):included=endpoint.address == address
        case .connection(let edge):included=direction == .input ? endpoint == edge.to:endpoint == edge.from
        default:included=false
        }
        return included
    }
    @discardableResult func inspect(_ endpoint:WorkflowEndpoint,direction:PortDirection)->Bool {
        guard canInspect(endpoint,direction:direction),let id=canvas.projection.displayIDByAddress[endpoint.address] else {return false}
        return canvas.inspectPort(nodeID:id,port:endpoint.port,direction:direction)
    }
    @discardableResult func connect(from:WorkflowEndpoint,to:WorkflowEndpoint)->Bool {
        guard canEdit,case .component(let address)=scope,from.address == address || to.address == address else {return refuse()}
        guard declared(from,direction:.output),declared(to,direction:.input) else {
            canvas.statusMessage="Connection not changed. Choose currently declared input and output ports; saved repair endpoints are not connection choices.";return false
        }
        return canvas.apply(.connect(from:from,to:to),lease:lease)
    }
    @discardableResult func reconnect(_ edge:WorkflowEdge,from:WorkflowEndpoint,to:WorkflowEndpoint)->Bool {
        guard canEdit,belongs(edge) else {return refuse()}
        guard declared(from,direction:.output),declared(to,direction:.input) else {
            canvas.statusMessage="Connection not changed. Both replacement endpoints must be currently declared ports.";return false
        }
        return canvas.apply(.reconnect(edgeID:edge.id,from:from,to:to),lease:lease)
    }
    @discardableResult func remove(_ edge:WorkflowEdge)->Bool {
        guard canEdit,belongs(edge) else {return refuse()}
        return canvas.apply(.delete(.init(edgeIDs:[edge.id])),lease:lease)
    }
}

enum WorkflowPortPresentation {
    static func title(type:String,port:String,direction:PortDirection)->String {
        if type == "workflow.evaluateTest",direction == .input,port == "test" {return "Data"}
        if type == "workflow.train",direction == .output,port == "model" {return "Trained model"}
        return port
    }
}

private struct WorkflowPortChoice:Identifiable {
    let endpoint:WorkflowEndpoint
    let title:String
    var id:WorkflowEndpoint {endpoint}
}

/// Uses the per-instance configured projection, never primitive registry defaults.
@MainActor private func workflowPortChoices(_ canvas:WorkflowCanvasAdapter,direction:PortDirection)->[WorkflowPortChoice] {
    canvas.graph.nodes.flatMap { node in
        guard let address=canvas.projection.addressByDisplayID[node.id],let definition=canvas.registry.component(type:node.type) else {return [WorkflowPortChoice]()}
        let name=node.name.isEmpty ? definition.title:node.name
        return (direction == .input ? definition.inputs:definition.outputs).compactMap {port in
            guard let descriptor=canvas.projection.portDescriptor(address:address,port:port,direction:direction),descriptor.contract != nil else {return nil}
            return WorkflowPortChoice(endpoint:.init(address:address,port:port),title:"\(name) · \(WorkflowPortPresentation.title(type:node.type,port:port,direction:direction)) · \(descriptor.semanticKind?.rawValue ?? "unknown") [\(address.domain.rawValue):\(address.localID)]")
        }
    }
}

@MainActor private func workflowEndpointTitle(_ endpoint:WorkflowEndpoint,canvas:WorkflowCanvasAdapter,direction:PortDirection = .input)->String {
    let id=canvas.projection.displayIDByAddress[endpoint.address]
    let node=canvas.graph.nodes.first {$0.id == id}
    let name=node.flatMap {$0.name.isEmpty ? canvas.registry.component(type:$0.type)?.title:$0.name} ?? endpoint.address.localID
    return name+" · "+WorkflowPortPresentation.title(type:node?.type ?? "",port:endpoint.port,direction:direction)
}

/// One row per saved edge, plus actual declared ports with no edge in that direction.
struct WorkflowPortRow: Identifiable {
    let edge: WorkflowEdge?
    let endpoint: WorkflowEndpoint
    let direction: PortDirection
    var id: String { edge.map { "edge:" + $0.id } ?? "port:\(direction.rawValue):\(endpoint.port)" }
    static func project(address: WorkflowNodeAddress, inputs: [String], outputs: [String], edges: [WorkflowEdge]) -> [Self] {
        let attached = edges.filter { $0.from.address == address || $0.to.address == address }.sorted { $0.id < $1.id }
        var rows = attached.map { edge in Self(edge: edge, endpoint: edge.to.address == address ? edge.to : edge.from, direction: edge.to.address == address ? .input : .output) }
        for (ports, direction) in [(inputs, PortDirection.input), (outputs, PortDirection.output)] {
            for port in ports {
                let endpoint = WorkflowEndpoint(address: address, port: port)
                if !attached.contains(where: { direction == .input ? $0.to == endpoint : $0.from == endpoint }) {
                    rows.append(Self(edge: nil, endpoint: endpoint, direction: direction))
                }
            }
        }
        return rows
    }
}

struct WorkflowComponentPortsView: View {
    @ObservedObject var owner: WorkflowAuthoringStore
    @ObservedObject var canvas: WorkflowCanvasAdapter
    let address: WorkflowNodeAddress
    var body: some View {
        if let id = canvas.projection.displayIDByAddress[address],
           let node = canvas.graph.nodes.first(where: { $0.id == id }),
           let definition = canvas.registry.component(type: node.type) {
            WorkflowPortsTable(canvas: canvas, rows: WorkflowPortRow.project(address: address, inputs: definition.inputs, outputs: definition.outputs, edges: canvas.inlineInspectorEdges), actions: .init(owner: owner, canvas: canvas, scope: .component(address)))
        } else { Text("This component is no longer available.").font(.caption) }
    }
}

struct WorkflowConnectionResultsView: View {
    @ObservedObject var owner: WorkflowAuthoringStore
    @ObservedObject var canvas: WorkflowCanvasAdapter
    let edge: WorkflowEdge
    var body: some View {
        WorkflowPortsTable(canvas: canvas, rows: [.init(edge: edge, endpoint: edge.from, direction: .output)], actions: .init(owner: owner, canvas: canvas, scope: .connection(edge)))
    }
}

private struct WorkflowPortsTable: View {
    @ObservedObject var canvas: WorkflowCanvasAdapter
    let rows: [WorkflowPortRow]
    let actions: WorkflowResultsPortActions
    @State private var error = ""
    var body: some View {
        let captures = capturedSources()
        VStack(alignment: .leading, spacing: 8) {
            WorkflowResultsTableViewport(minimumWidth: 760) {
                VStack(spacing: 0) {
                    HStack(spacing: 12) {
                        Text("From").frame(maxWidth: .infinity, alignment: .leading)
                        Text("To").frame(maxWidth: .infinity, alignment: .leading)
                        Text("Kind").frame(width: 120, alignment: .leading)
                        Text("Capture").frame(width: 105, alignment: .leading)
                        Text("Actions").frame(width: 135, alignment: .leading)
                    }.font(.caption.weight(.semibold)).foregroundStyle(MLTheme.muted).padding(8).background(MLTheme.panel)
                    ForEach(rows) { row in
                        HStack(spacing: 12) {
                            endpointLabel(row.edge?.from ?? (row.direction == .output ? row.endpoint : nil),direction:.output)
                            endpointLabel(row.edge?.to ?? (row.direction == .input ? row.endpoint : nil),direction:.input)
                            let descriptor = descriptor(row)
                            HStack(spacing: 5) { CanvasPortGlyphView(glyph: descriptor.glyph); Text(descriptor.shortLabel) }
                                .font(.caption).frame(width: 120, alignment: .leading).help(descriptor.accessibilityDescription)
                            Text(capture(row, captures: captures)).font(.caption).foregroundStyle(capture(row, captures: captures) == "Stale" ? MLTheme.rose : MLTheme.muted)
                                .frame(width: 105, alignment: .leading)
                            HStack(spacing: 8) {
                                Button("Inspect") { _ = actions.inspect(row.endpoint, direction: row.direction) }.buttonStyle(FlatButtonStyle())
                                    .disabled(!actions.canInspect(row.endpoint, direction: row.direction))
                                    .help(actions.canInspect(row.endpoint, direction: row.direction) ? "Inspect this port" : "This saved endpoint is no longer declared. Reconnect or remove its connection.")
                                    .accessibilityLabel("Inspect \(row.direction.rawValue) \(workflowEndpointTitle(row.endpoint, canvas: canvas,direction:row.direction))")
                                editMenu(row)
                            }.frame(width: 135, alignment: .leading)
                        }.padding(8)
                        Rectangle().fill(MLTheme.rule).frame(height: 1)
                    }
                }.overlay(Rectangle().stroke(MLTheme.rule))
            }
            if rows.isEmpty { Text("This component has no declared ports or connections.").font(.caption).foregroundStyle(MLTheme.muted) }
            if !error.isEmpty { Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled) }
            if !actions.canEdit { Text("Editing unavailable · canvas busy or edits pending").font(.caption2).foregroundStyle(MLTheme.muted) }
            DisclosureGroup("Information") {
                Text("Rows show saved connections and actual unconnected ports. Kind describes the declared contract; it is not measured output. Capture describes the source port's retained evidence, without rereading source files. Inspect is read-only. Connection changes are validated and can be undone; apply or discard pending edits before changing endpoints.")
                    .font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 4)
            }.font(.caption).tint(MLTheme.teal)
        }.onChange(of: actions.lease) { _ in error = "" }
    }
    private func endpointLabel(_ endpoint: WorkflowEndpoint?,direction:PortDirection) -> some View {
        Text(endpoint.map { workflowEndpointTitle($0, canvas: canvas,direction:direction) } ?? "Not connected")
            .font(.caption).lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
            .help(endpoint.map { "\(workflowEndpointTitle($0, canvas: canvas,direction:direction)) [\($0.address.domain.rawValue):\($0.address.localID)]" } ?? "Not connected")
    }
    private func descriptor(_ row: WorkflowPortRow) -> CanvasPortDescriptor {
        guard let id = canvas.projection.displayIDByAddress[row.endpoint.address] else { return .unknown(direction: row.direction, reason: "Endpoint unavailable") }
        return canvas.canvasPortDescriptor(nodeID: id, port: row.endpoint.port, direction: row.direction)
    }
    private func capturedSources() -> [WorkflowNodeAddress: WorkflowCapturedDataOutput] {
        guard let provider = canvas.outputProvider, provider.projectID == actions.owner.projectID, provider.sessionID == actions.owner.sessionID else { return [:] }
        let addresses = Set(rows.compactMap { row in row.edge?.from.address ?? (row.direction == .output ? row.endpoint.address : nil) })
        return provider.snapshots(for: addresses)
    }
    private func capture(_ row: WorkflowPortRow, captures: [WorkflowNodeAddress: WorkflowCapturedDataOutput]) -> String {
        guard let source = row.edge?.from ?? (row.direction == .output ? row.endpoint : nil),
              let snapshot = captures[source.address],
              snapshot.projectID == actions.owner.projectID, snapshot.sessionID == actions.owner.sessionID,
              snapshot.address == source.address, snapshot.ports[source.port] != nil else { return "Not captured" }
        if !snapshot.matchesCurrentConstruction || !snapshot.matchesCurrentBindings { return "Stale" }
        return snapshot.phase == .completed ? "Captured" : "Partial capture"
    }
    private func perform(_ action: () -> Bool) {
        error = action() ? "" : canvas.statusMessage
    }
    private func editMenu(_ row: WorkflowPortRow) -> some View {
        Menu {
            if let edge = row.edge {
                if row.direction == .output, case .component = actions.scope {
                    Menu("Connect another destination") {
                        ForEach(workflowPortChoices(canvas, direction: .input).filter { $0.endpoint.address != row.endpoint.address }) { choice in
                            Button(choice.title) { perform { actions.connect(from: row.endpoint, to: choice.endpoint) } }
                        }
                    }
                }
                Menu("Reconnect source") {
                    ForEach(workflowPortChoices(canvas, direction: .output)) { choice in
                        Button(choice.title) { perform { actions.reconnect(edge, from: choice.endpoint, to: edge.to) } }
                    }
                }
                Menu("Reconnect destination") {
                    ForEach(workflowPortChoices(canvas, direction: .input)) { choice in
                        Button(choice.title) { perform { actions.reconnect(edge, from: edge.from, to: choice.endpoint) } }
                    }
                }
                Divider()
                Button("Remove connection", role: .destructive) { perform { actions.remove(edge) } }
            } else {
                let choices = workflowPortChoices(canvas, direction: row.direction == .input ? .output : .input).filter { $0.endpoint.address != row.endpoint.address }
                ForEach(choices) { choice in
                    Button(choice.title) { perform { actions.connect(from: row.direction == .input ? choice.endpoint : row.endpoint, to: row.direction == .input ? row.endpoint : choice.endpoint) } }
                }
                if choices.isEmpty { Text("No declared endpoints available") }
            }
        } label: { Image(systemName: row.edge == nil ? "plus" : "ellipsis") }
        .menuStyle(.borderlessButton).fixedSize().disabled(!actions.canEdit || (row.edge.map {!actions.canEditConnection($0)} ?? false))
        .accessibilityLabel("\(row.edge == nil ? "Connect" : "Reconnect or remove") \(workflowEndpointTitle(row.endpoint, canvas: canvas,direction:row.direction))")
    }
}
