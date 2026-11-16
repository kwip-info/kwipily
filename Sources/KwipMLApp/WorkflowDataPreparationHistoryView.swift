import SwiftUI
import AppKit
import KwipMLCore

/// Historical preparation is verified asynchronously; opening this view never
/// reads the current source or repeats a write to the user's destination.
struct WorkflowDataPreparationHistoryView: View {
    @ObservedObject var history: WorkflowDataPreparationHistory
    let entry: WorkflowDataPreparationHistory.Entry
    let isActive: () -> Bool
    var tableViewSession: ((WorkflowDataPreparationHistory.Entry, TableViewQuerySession.Artifact) -> TableViewQuerySession?)? = nil
    @State private var imageResolver: FileImageArtifactResolver?
    @State private var tableViewSessions: [String: TableViewQuerySession] = [:]
    @State private var verified: WorkflowDataPreparationHistory.Entry?
    @State private var outputs: [WorkflowNodeAddress: [String: JSONValue]] = [:]
    @State private var selectedNodeID = ""
    @State private var page = "tables"
    @State private var generation = 0
    @State private var loading = true
    @State private var databaseRecords:[DatabaseOutputLedger.Record]=[]
    @State private var message = "Verifying saved preparation…"

    private var nodes: [WorkflowNode] {
        (verified?.document.nodes ?? []).filter { outputs[$0.address] != nil }
    }
    private var selectedNode: WorkflowNode? { nodes.first { $0.address.localID == selectedNodeID } ?? nodes.first }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Label("Saved preparation", systemImage: "tablecells").font(.system(size: 13, weight: .semibold))
                Spacer()
                if loading { ProgressView().controlSize(.small) }
                if let verified {
                    Button("Show preparation files") {NSWorkspace.shared.activateFileViewerSelecting([verified.artifactDirectory])}
                        .buttonStyle(FlatButtonStyle()).disabled(!isActive())
                }
                Button("Verify again") { generation += 1 }.buttonStyle(FlatButtonStyle()).disabled(loading || !isActive())
            }
            if verified==nil {Text(message).font(.system(size:11)).foregroundStyle(loading ? MLTheme.muted:MLTheme.rose).textSelection(.enabled)}
            if let verified {
                HStack(spacing: 8) {
                    pageButton("Tables", id: "tables")
                    pageButton("Saved files", id: "files")
                    if !databaseRecords.isEmpty {pageButton(databaseRecords.contains {$0.configuration["kind"] == .string("api")} ? "External writes":"Database writes",id:"databases")}
                    pageButton("Information", id: "provenance")
                    Spacer()
                    Text("\(outputs.count) captured data components").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                }
                Rectangle().fill(MLTheme.rule).frame(height: 1)
                ThemedScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        if page == "tables" { tables }
                        else if page == "files" { savedFiles(verified) }
                        else if page == "databases" {ForEach(databaseRecords) {record in if record.configuration["kind"] == .string("api") {APIWriteReceiptView(record:record)} else {DatabaseOutputReceiptView(record:record)}}}
                        else { provenance(verified) }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(.trailing, 8)
                }
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    Text(loading ? "Checking captured evidence" : "Saved evidence unavailable").font(.system(size: 12, weight: .semibold))
                    Text("Tables appear after the captured request, graph and complete artifact inventory pass verification. Current canvas edits are not substituted.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }.padding(12).frame(maxWidth: .infinity, alignment: .leading).overlay(Rectangle().stroke(MLTheme.rule))
                Spacer()
            }
        }.task(id: "\(entry.id):\(generation)") { await load() }
            .onDisappear { tableViewSessions.values.forEach { $0.invalidate() } }
    }
    private func pageButton(_ title: String, id: String) -> some View {
        Button(title) { page = id }.buttonStyle(FlatButtonStyle(prominent: page == id))
    }
    @ViewBuilder private var tables: some View {
        if let node = selectedNode {
            FlatChoiceControl(title: "Captured component", selection: node.address.localID,
                options: nodes.map { .init(id: $0.address.localID, title: $0.name.isEmpty ? $0.address.localID : $0.name) }) { selectedNodeID = $0 }
            if let session = tableViewSessions[node.address.localID] {
                TableViewOutputView(session: session, imageResolver: imageResolver).frame(height: 360)
            } else if let ports = outputs[node.address] {
                Text(verified?.coverage?.isComplete(node.address.localID) == false
                    ? "Showing an archived source-order sample. Row counts describe this sampled output."
                    : "Archived sample")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                ForEach(ports.keys.sorted(), id: \.self) { port in
                    VStack(alignment: .leading, spacing: 8) {
                        DataPortPreviewView(port: port, output: WorkflowCapturedDataOutput.render(ports[port]!) as? [String: Any] ?? [:], imageResolver: imageResolver)
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading).overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
        } else {
            Text("This archive has no table or fitted-state previews.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func savedFiles(_ value: WorkflowDataPreparationHistory.Entry) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Files recorded during this run").font(.system(size: 12, weight: .semibold))
            Text("These receipts describe the completed write at execution time. Destination files may have changed since; opening History does not write them again or revalidate their current contents.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if value.savedOutputs.isEmpty {
                Text("No delimited output files were recorded.").font(.caption).padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading).overlay(Rectangle().stroke(MLTheme.rule))
            } else {
                ScrollView(.horizontal) {
                    VStack(spacing: 0) {
                        fileRow(["Component", "Recorded destination", "Rows", "Write status"], header: true)
                        ForEach(Array(value.savedOutputs.enumerated()), id: \.offset) { _, output in
                            fileRow([componentName(text(output["nodeID"]), in: value.document), text(output["path"]), text(output["rowCount"]), text(output["status"])])
                        }
                    }.overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
        }
    }
    private func fileRow(_ values: [String], header: Bool = false) -> some View {
        HStack(spacing: 0) {
            ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                Text(value).font(.system(size: header ? 10 : 11, weight: header ? .semibold : .regular))
                    .textSelection(.enabled).lineLimit(header ? 1 : 2)
                    .frame(width: [140.0, 380.0, 70.0, 100.0][index], alignment: .leading).padding(.horizontal, 10).padding(.vertical, 8)
            }
        }.background(header ? MLTheme.panel : MLTheme.background)
            .overlay(alignment: .bottom) { Rectangle().fill(MLTheme.rule).frame(height: 1) }
    }
    private func provenance(_ value: WorkflowDataPreparationHistory.Entry) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            provenanceRow("Verification", value.coverage?.isSampled == true ? "Verified preparation · sampled branches recorded separately" : "Complete captured preparation · no trained model implied")
            if let coverage = value.coverage {
                provenanceRow("Complete branches", String(coverage.completeNodeIDs.count))
                provenanceRow("Sampled branches", String(coverage.sampledNodeIDs.count))
            }
            provenanceRow("Operation", value.id)
            provenanceRow("Captured revision", String(value.document.revision))
            provenanceRow("Recorded", value.completedAt.formatted(date: .abbreviated, time: .standard))
            provenanceRow("Request", value.requestIdentity)
            provenanceRow("Manifest", value.manifestSHA256)
            provenanceRow("Sources", "Archived snapshots checked; current files and permissions were not revalidated")
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func provenanceRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(label).font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted).frame(width: 120, alignment: .leading)
            Text(value).font(.system(size: 11, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
        }.padding(10).overlay(alignment: .bottom) { Rectangle().fill(MLTheme.rule).frame(height: 1) }
    }
    private func componentName(_ id: String, in document: WorkflowDocument) -> String {
        guard let node = document.nodes.first(where: { $0.address.domain == .data && $0.address.localID == id }), !node.name.isEmpty else { return id }
        return node.name
    }
    private func text(_ value: JSONValue?) -> String {
        switch value {
        case .string(let text): return text
        case .integer(let number): return String(number)
        default: return "Not recorded"
        }
    }
    @MainActor private func load() async {
        tableViewSessions.values.forEach { $0.invalidate() }; tableViewSessions = [:]
        loading = true; verified = nil; outputs = [:]; imageResolver = nil; message = "Verifying saved preparation…"
        do {
            let checked = try await history.verify(entry)
            let captured = try await history.outputs(for: checked)
            databaseRecords = (try? await history.databaseRecords(for:checked)) ?? []
            guard !Task.isCancelled, isActive() else { return }
            imageResolver = try FileImageArtifactResolver(directory: checked.artifactDirectory, verifiedArtifacts: checked.artifacts)
            verified = checked; outputs = captured
            if let factory = tableViewSession {
                tableViewSessions = checked.tableViews.reduce(into: [:]) { result, item in
                    if let session = factory(checked, item.value) { result[item.key] = session }
                }
            }
            message = "Verified archived preparation. Current sources and destination files were not revalidated."
        } catch {
            guard !Task.isCancelled, isActive() else { return }
            message = "Saved preparation was not loaded: " + error.localizedDescription
        }
        loading = false
    }
}
