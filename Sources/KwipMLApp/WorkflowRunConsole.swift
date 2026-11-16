import Foundation
import Combine
import SwiftUI
import KwipMLCore

/// Session-local observations, separate from verified archived results and raw data.
/// Independently observed so console appends do not invalidate the canvas host.
@MainActor final class WorkflowRunConsole:ObservableObject {
    enum Level:String {case info,warning,error}
    struct Entry:Identifiable {
        let id=UUID()
        let time=Date()
        let phase:WorkflowExecutionPhase
        let level:Level
        let message:String
        let component:WorkflowNodeAddress?
    }
    @Published private(set) var entries:[Entry]=[]
    @Published private(set) var omittedCount=0
    func append(phase:WorkflowExecutionPhase,level:Level = .info,message:String,component:WorkflowNodeAddress? = nil) {
        if entries.count == 500 {entries.removeFirst();omittedCount += 1}
        entries.append(.init(phase:phase,level:level,message:String(message.prefix(4000)),component:component))
    }
}

/// Pure presentation selection; the console's bounded source capture stays unchanged.
enum WorkflowConsolePresentation {
    static func visibleEntries(_ entries:[WorkflowRunConsole.Entry], component:WorkflowNodeAddress? = nil,
                               problemsOnly:Bool = false, query:String = "") -> [WorkflowRunConsole.Entry] {
        entries.filter {
            (component == nil || $0.component == component) &&
            (!problemsOnly || $0.level != .info) &&
            (query.isEmpty || $0.message.localizedCaseInsensitiveContains(query) ||
             ($0.component?.localID.localizedCaseInsensitiveContains(query) ?? false))
        }
    }
}

struct WorkflowRunConsoleView:View {
    @Environment(\.appearanceTheme) private var appearanceTheme
    @ObservedObject var console:WorkflowRunConsole
    var component:WorkflowNodeAddress? = nil
    var componentNames:[WorkflowNodeAddress:String]=[:]
    @State private var problemsOnly=false
    @State private var query=""
    @State private var pausedEntries:[WorkflowRunConsole.Entry]?
    @State private var pausedOmitted=0
    @State private var selectedEntry:WorkflowRunConsole.Entry?
    @State private var detailsExpanded=false
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            ViewThatFits(in:.horizontal) {
                HStack(spacing:12) {title;search.frame(width:240);controls}
                VStack(alignment:.leading,spacing:8) {title;search;controls}
            }
            Text(pausedEntries == nil ? "Live · latest 500 events" : "Display paused · execution continues")
                .font(.caption2).foregroundStyle(MLTheme.muted)
            let omitted=pausedEntries == nil ? console.omittedCount:pausedOmitted
            if omitted>0 {Text("\(omitted) earlier events omitted from the run’s activity capture.").font(.caption).foregroundStyle(.orange)}
            let visible=WorkflowConsolePresentation.visibleEntries(pausedEntries ?? console.entries,
                component:component,problemsOnly:problemsOnly,query:query)
            if visible.isEmpty {Text("No matching captured events.").font(.caption).foregroundStyle(MLTheme.muted)}
            WorkflowResultTable(columns:WorkflowResultTableData.columns(["Time","Stage","Level","Component","Event"]),rows:visible.enumerated().map {index,entry in
                WorkflowResultTableData.row(index,[entry.time.formatted(date:.omitted,time:.standard),entry.phase.rawValue,entry.level.rawValue,componentLabel(entry),entry.message])
            },selectedRow:Binding(get:{visible.firstIndex {$0.id==selectedEntry?.id}},set:{index in
                selectedEntry=index.flatMap {visible.indices.contains($0) ? visible[$0]:nil};detailsExpanded=selectedEntry != nil
            }))
            if let selectedEntry {
                DisclosureGroup("Captured event details",isExpanded:$detailsExpanded) {
                    VStack(alignment:.leading,spacing:5) {
                        Text("\(selectedEntry.phase.rawValue) · \(selectedEntry.level.rawValue) · \(componentLabel(selectedEntry))").foregroundStyle(MLTheme.muted)
                        Text(selectedEntry.message).textSelection(.enabled)
                    }.font(.system(size:11,design:.monospaced)).frame(maxWidth:.infinity,alignment:.leading).padding(.vertical,6)
                }.font(.caption)
            }
            DisclosureGroup("Information") {
                Text("Times record console receipt, not operation duration. Component labels appear only for explicitly attributed events. Pause freezes display only; execution and capture continue. Saved evidence and artifact verification remain in History.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
                if let component { Text("Exact component: \(component.domain.rawValue) · \(component.localID)").font(.caption2).textSelection(.enabled) }
            }.font(.caption)

        }
        .onChange(of:component) { _, _ in selectedEntry=nil;detailsExpanded=false }
        .onChange(of:ObjectIdentifier(console)) { _, _ in
            selectedEntry=nil;detailsExpanded=false;pausedEntries=nil;pausedOmitted=0
        }
    }
    private var title:some View {Text("Activity").font(.caption.weight(.semibold))}
    private var search:some View {
        TextField("Search captured events",text:$query).textFieldStyle(.plain)
            .padding(6).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private var controls:some View {
        HStack(spacing:12) {
            Toggle("Problems only",isOn:$problemsOnly).toggleStyle(.checkbox).font(.caption)
            Spacer(minLength:0)
            Button(pausedEntries == nil ? "Pause display" : "Follow live") {
                if pausedEntries == nil {pausedEntries=console.entries;pausedOmitted=console.omittedCount}
                else {pausedEntries=nil}
            }.buttonStyle(FlatButtonStyle())
        }.fixedSize(horizontal:true,vertical:false)
    }
    private func componentLabel(_ entry:WorkflowRunConsole.Entry)->String {
        entry.component.map {componentNames[$0] ?? "\($0.domain.rawValue) · \($0.localID)"} ?? "Run"
    }
    private func color(_ entry:WorkflowRunConsole.Entry)->Color {
        entry.level == .error ? MLTheme.rose : entry.level == .warning ? .orange : .primary
    }
}

struct WorkflowActivityStatus:View {
    @ObservedObject var execution:WorkflowExecutionSession
    var body:some View {
        WorkflowResultsStatusStrip(title:"Training",phase:execution.phase.rawValue,
            message:execution.statusMessage,isRunning:[.preparing,.training].contains(execution.phase),
            detail:"Request " + String(execution.request.requestIdentity.prefix(12)))
    }
}
