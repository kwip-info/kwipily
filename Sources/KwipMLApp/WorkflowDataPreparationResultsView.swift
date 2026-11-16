import SwiftUI
import KwipMLCore

/// The data-only run uses the existing bottom Results dock. Counts and messages
/// come from its captured session; it never substitutes training metrics.
struct WorkflowDataPreparationResultsView: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    enum Page { case activity, summary, artifacts }
    @ObservedObject var session: WorkflowDataPreparationSession
    let page: Page
    @State private var selectedPort:String?
    @State private var portDetailsExpanded=false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            status
            switch page {
            case .activity: activity
            case .summary: summary
            case .artifacts: artifacts
            }
        }
    }
    private var status: some View {
        WorkflowResultsStatusStrip(title:"Data preparation",phase:session.phase.rawValue,
            message:session.statusMessage,isRunning:session.isRunning,detail:"\(session.preparationOutputs.count) components")
    }
    private var activity: some View {
        WorkflowMessageActivityView(messages:session.messages)
    }
    private var summary: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !session.preparationOutputs.isEmpty {
                WorkflowResultsTableViewport {
                    VStack(spacing: 0) {
                        HStack(spacing: 12) {
                            Text("Component").frame(maxWidth: .infinity, alignment: .leading)
                            Text("Tool").frame(maxWidth: .infinity, alignment: .leading)
                            Text("Captured output").frame(maxWidth: .infinity, alignment: .leading)
                            Text("State").frame(width: 110, alignment: .leading)
                        }.font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted)
                            .padding(8).background(MLTheme.background)
                        ForEach(session.preparationOutputs.keys.sorted(), id: \.self) { id in
                            let type = session.graph.nodes.first { $0.id == id }?.operatorID ?? ""
                            let ports = WorkflowGlobalResultsPresentation.capturedPorts([id: session.preparationOutputs[id] ?? [:]], completed: true)
                            HStack(spacing: 12) {
                                HStack(spacing: 8) {
                                    ComponentIcon(type: type, size: 22)
                                    Text(componentName(id)).lineLimit(1)
                                }.frame(maxWidth: .infinity, alignment: .leading)
                                Text(ComponentStyle.shortTitle(type, fallback: type)).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                                Text(ports.map { port in
                                    let count = port.rowCount.map { "\($0) records" } ?? port.kind
                                    return ports.count == 1 ? count : "\(port.port) · \(count)"
                                }.joined(separator: "; ")).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                                Label(session.phase == .completed ? "Captured" : "Partial", systemImage: session.phase == .completed ? "checkmark.circle.fill" : "circle.lefthalf.filled")
                                    .foregroundStyle(session.phase == .completed ? appearanceTheme.accentColor : Color.orange)
                                    .frame(width: 110, alignment: .leading)
                            }.font(.system(size: 11)).padding(8)
                            Divider()
                        }
                    }.background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
            DisclosureGroup("Information") {
                VStack(alignment:.leading,spacing:8) {
                    metadata("Run",session.runID.uuidString)
                    metadata("Captured revision",String(session.document.revision))
                    metadata("Configured data components",String(session.graph.nodes.count))
                    if let current = session.currentObservationIdentity, let ledger = session.observations {
                        Text(ledger.disclosure(current: current))
                    }
                    Text("Inspect a component or connector to explore its captured table. Failed and cancelled runs retain reached-stage observations only.")
                    Text("A data preparation run does not imply a trained model, prediction or evaluation. History owns archived output verification.")
                }.font(.caption2).foregroundStyle(MLTheme.muted).padding(.vertical,6)
            }.font(.caption)

        }
    }
    @ViewBuilder private var artifacts: some View {
        if session.phase == .completed {
            let inventory=WorkflowGlobalResultsPresentation.capturedPorts(session.preparationOutputs,completed:true)
            Text("Recorded captured ports").font(.system(size:12,weight:.semibold))
            if inventory.isEmpty {
                Text("No captured ports were recorded.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
                capturedPortTable(inventory)
            }
            DisclosureGroup("Information") {
                Text("Recorded ports describe captured observations from this completed run; counts apply to the captured working data. They are not a fresh file integrity check. Inspect a component to explore its table. History owns archived artifact verification and export. Delimited write receipts below describe separately requested destination files, which may have changed since completion.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            }.font(.caption)
            if session.savedOutputs.isEmpty {
                Text("No delimited writes requested.").font(.caption2).foregroundStyle(MLTheme.muted)
            } else {
                Text("Recorded delimited writes").font(.system(size:11,weight:.semibold))
                ScrollView(.horizontal) {
                    VStack(spacing: 0) {
                        outputRow(["Component", "Recorded destination", "Rows", "Recorded write status"], header: true)
                        ForEach(Array(session.savedOutputs.enumerated()), id: \.offset) { _, receipt in
                            outputRow([componentName(text(receipt["nodeID"])), text(receipt["path"]), text(receipt["rowCount"]), text(receipt["status"])])
                        }
                    }.background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
        } else {
            Text("No completed artifact inventory is offered for this run. In-flight files and partial observations are not completed outputs.")
                .font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func capturedPortTable(_ rows:[WorkflowGlobalResultsPresentation.CapturedPort])->some View {
        VStack(alignment:.leading,spacing:8) {
            WorkflowResultsTableViewport(minimumWidth: 630) {
                VStack(spacing:0) {
                    capturedPortRow(["Component","Port","Kind","Records","State"],header:true)
                    LazyVStack(spacing:0) {
                        ForEach(rows) { row in
                            Button {selectedPort=row.id;portDetailsExpanded=true} label: {
                                capturedPortRow([componentName(row.nodeID),row.port,row.kind,row.rowCount.map(String.init) ?? "—",row.state])
                                    .background(selectedPort == row.id ? appearanceTheme.accentColor.opacity(0.08):Color.clear)
                            }.buttonStyle(.plain).help("Show captured port details")
                        }
                    }
                }.background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
            }
            if let row=rows.first(where:{$0.id == selectedPort}) {
                DisclosureGroup("Captured port details",isExpanded:$portDetailsExpanded) {
                    VStack(alignment:.leading,spacing:6) {
                        metadata("Component",componentName(row.nodeID))
                        metadata("Port",row.port)
                        metadata("Kind",row.kind)
                        metadata("Captured records",row.rowCount.map(String.init) ?? "Not recorded")
                        if let version=row.dataVersion {metadata("Captured data version",version)}
                    }.padding(.vertical,6)
                }.font(.caption)
            }
        }
    }
    private func capturedPortRow(_ values:[String],header:Bool=false)->some View {
        HStack(spacing:0) {
            ForEach(Array(values.enumerated()),id:\.offset) {index,value in
                Text(value).font(.system(size:header ? 10:11,weight:header ? .semibold:.regular))
                    .lineLimit(1)
                    .frame(width: index == 0 ? nil : [170.0,110.0,95.0,75.0,100.0][index], alignment:.leading)
                    .frame(maxWidth: index == 0 ? .infinity : nil, alignment: .leading)
                    .foregroundStyle(!header && index == 4 ? appearanceTheme.accentColor:header ? MLTheme.muted:Color.primary)
                    .padding(.horizontal,8).padding(.vertical,7)
            }
        }.background(header ? MLTheme.panel:Color.clear)
            .overlay(alignment:.bottom){Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    private func outputRow(_ values: [String], header: Bool = false) -> some View {
        HStack(spacing: 0) {
            ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                Text(value).font(.system(size: header ? 10 : 11, weight: header ? .semibold : .regular))
                    .textSelection(.enabled).lineLimit(header ? 1 : 2)
                    .frame(width: [140.0, 380.0, 70.0, 100.0][index], alignment: .leading).padding(.horizontal, 10).padding(.vertical, 8)
            }
        }.background(header ? MLTheme.panel : MLTheme.background)
            .overlay(alignment: .bottom) { Rectangle().fill(MLTheme.rule).frame(height: 1) }
    }
    private func metadata(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label.uppercased()).font(.system(size: 9, weight: .semibold)).foregroundStyle(MLTheme.muted)
            Text(value).font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    private func componentName(_ id: String) -> String {
        let node = session.document.nodes.first(where: { $0.address.domain == .data && $0.address.localID == id })
        if let name = node?.name, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return name }
        guard let type = session.graph.nodes.first(where: { $0.id == id })?.operatorID else { return "Component" }
        return ComponentStyle.shortTitle(type, fallback: type)
    }
    private func text(_ value: JSONValue?) -> String {
        switch value { case .string(let text): return text; case .integer(let value): return String(value); default: return "Not recorded" }
    }
}
