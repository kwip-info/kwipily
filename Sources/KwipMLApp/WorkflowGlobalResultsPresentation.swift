import SwiftUI
import KwipMLCore

/// Presentation only: no inferred operation, duration, health or artifact verification.
enum WorkflowGlobalResultsPresentation {
    enum Tone: Equatable { case neutral, active, complete, attention, problem }
    struct Artifact: Identifiable, Equatable {
        var id: String { name }
        let name: String
        let kind: String
        let state: String
        let hash: String
    }
    struct CapturedPort:Identifiable,Equatable {
        var id:String {"\(nodeID.utf8.count):\(nodeID)\(port)"}
        let nodeID:String
        let port:String
        let kind:String
        let rowCount:Int64?
        let state:String
        let dataVersion:String?
    }
    static func capturedPorts(_ outputs:[String:[String:JSONValue]],completed:Bool)->[CapturedPort] {
        guard completed else {return []}
        return outputs.keys.sorted().flatMap { node in
            (outputs[node] ?? [:]).keys.sorted().map { port in
                let value=outputs[node]?[port]
                let object:[String:JSONValue] = {if case .object(let object)=value {return object};return [:]}()
                let kind:String = {if case .string(let kind)=object["kind"] {return kind};return "Not recorded"}()
                let count:Int64? = {if case .object(let preview)=object["preview"],case .integer(let count)=preview["rowCount"],count>=0 {return count};return nil}()
                let version:String? = {if case .object(let lineage)=object["lineage"],case .string(let version)=lineage["dataVersion"] {return version};return nil}()
                return .init(nodeID:node,port:port,kind:kind,rowCount:count,state:"Recorded",dataVersion:version)
            }
        }
    }
    struct Message: Identifiable, Equatable {
        var id: Int { sequence }
        let sequence: Int
        let message: String
    }
    static func tone(_ phase: String) -> Tone {
        switch phase.lowercased() {
        case "completed": return .complete
        case "failed": return .problem
        case "cancelled": return .attention
        case "preparing", "training": return .active
        default: return .neutral
        }
    }
    static func artifacts(hashes: [String:String], completed: Bool) -> [Artifact] {
        guard completed else { return [] }
        return hashes.keys.sorted().map { .init(name:$0,kind:"Captured file",state:"Recorded",hash:hashes[$0] ?? "") }
    }
    static func retainedSelection(_ selected:Message?,in messages:[String])->Message? {
        guard let selected else {return nil}
        return self.messages(messages).first {$0 == selected}
    }
    static func messages(_ messages: [String], query: String = "") -> [Message] {
        messages.enumerated().suffix(200).compactMap { index, message in
            guard query.isEmpty || message.localizedCaseInsensitiveContains(query) else { return nil }
            return .init(sequence:index+1,message:message)
        }
    }
}

enum WorkflowResultsStyle {
    static func color(_ tone: WorkflowGlobalResultsPresentation.Tone, accent: Color) -> Color {
        switch tone {
        case .neutral: return MLTheme.muted
        case .active, .complete: return accent
        case .attention: return .orange
        case .problem: return MLTheme.rose
        }
    }
}

struct WorkflowResultsStatusStrip: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    let title: String
    let phase: String
    let message: String
    var isRunning = false
    var detail: String? = nil
    var body: some View {
        let tone = WorkflowGlobalResultsPresentation.tone(phase)
        HStack(spacing: 16) {
            HStack(spacing: 8) {
                if isRunning { ProgressView().controlSize(.small) }
                else { Image(systemName: tone == .complete ? "checkmark.circle.fill" : tone == .problem ? "exclamationmark.circle.fill" : tone == .attention ? "exclamationmark.triangle.fill" : "circle") }
                Text(phase.capitalized).fontWeight(.semibold)
            }.foregroundStyle(WorkflowResultsStyle.color(tone, accent: appearanceTheme.accentColor))
            Rectangle().fill(WorkflowResultsStyle.color(tone, accent: appearanceTheme.accentColor).opacity(0.35)).frame(width: 1, height: 18)
            Label(title, systemImage: "externaldrive")
            if !message.isEmpty && phase.lowercased() != "completed" {
                Text(message).lineLimit(2).textSelection(.enabled)
            }
            Spacer(minLength: 0)
            if let detail { Text(detail).foregroundStyle(MLTheme.muted) }
        }.font(.system(size: 11)).padding(.horizontal, 12).padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(WorkflowResultsStyle.color(tone, accent: appearanceTheme.accentColor).opacity(0.10))
            .overlay(Rectangle().stroke(WorkflowResultsStyle.color(tone, accent: appearanceTheme.accentColor).opacity(0.25)))
            .help(message)

    }
}

/// Used only when a session exposes bounded raw messages, not structured console events.
struct WorkflowMessageActivityView: View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    let messages: [String]
    @State private var query = ""
    @State private var pausedMessages: [String]?
    @State private var selected: WorkflowGlobalResultsPresentation.Message?
    @State private var detailsExpanded = false
    private var displayedSelection:WorkflowGlobalResultsPresentation.Message? {
        WorkflowGlobalResultsPresentation.retainedSelection(selected,in:pausedMessages ?? messages)
    }
    var body: some View {
        VStack(alignment:.leading,spacing:8) {
            HStack(spacing:12) {
                TextField("Search captured events",text:$query).flatField()
                Button(pausedMessages == nil ? "Pause display" : "Follow live") {
                    if pausedMessages == nil { pausedMessages = messages } else { pausedMessages = nil }
                }.buttonStyle(FlatButtonStyle())
            }
            Text(pausedMessages == nil ? "Live · latest 200 messages" : "Display paused · execution continues")
                .font(.caption2).foregroundStyle(MLTheme.muted)
            let rows = WorkflowGlobalResultsPresentation.messages(pausedMessages ?? messages,query:query)
            WorkflowResultTable(columns:WorkflowResultTableData.columns(["Captured event"]),rows:rows.map {WorkflowResultTableData.row($0.sequence-1,[$0.message])},selectedRow:Binding(get:{displayedSelection.map {$0.sequence-1}},set:{index in
                selected=rows.first {$0.sequence-1==index};detailsExpanded=selected != nil
            }))
            if let selected=displayedSelection {
                DisclosureGroup("Captured message · \(selected.sequence)",isExpanded:$detailsExpanded) {
                    Text(selected.message).font(.system(size:11,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading).padding(.vertical,6)
                }.font(.caption)
            }
            DisclosureGroup("Information") {
                Text("Messages retain their captured text. Sequence is relative to the retained message window; no event time, component attribution or level is inferred. Pausing freezes this display only. History owns saved evidence and verification.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            }.font(.caption)
        }
        .onChange(of:messages) { _, values in
            guard pausedMessages == nil else {return}
            selected=WorkflowGlobalResultsPresentation.retainedSelection(selected,in:values)
            if selected == nil {detailsExpanded=false}
        }
    }
}

/// Fill the available dock width while retaining horizontal scrolling at narrow sizes.
/// Measures its own container only; canvas camera changes do not drive this layout.
struct WorkflowResultsTableViewport<Content: View>: View {
    var minimumWidth: CGFloat = 720
    @ViewBuilder let content: () -> Content
    @State private var width: CGFloat = 0
    var body: some View {
        ThemedScrollView(.horizontal) {
            content().frame(width: max(minimumWidth, width), alignment: .topLeading)
        }.fixedSize(horizontal: false, vertical: true)
            .background(GeometryReader { proxy in
                Color.clear.preference(key: WorkflowResultsTableWidth.self, value: proxy.size.width)
            })
            .onPreferenceChange(WorkflowResultsTableWidth.self) { width = $0 }
    }
}
private struct WorkflowResultsTableWidth: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}
