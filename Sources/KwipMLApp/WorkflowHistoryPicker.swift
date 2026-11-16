import SwiftUI

/// Shared local-history chooser. Listing is not verification; the selection action verifies evidence.
struct WorkflowHistoryPicker:View {
    let title:String
    let explanation:String
    let emptyMessage:String
    @ObservedObject var history:WorkflowHistoryStore
    let active:Bool
    var completedOnly=false
    let status:String
    let onClose:()->Void
    let onSelect:(WorkflowHistoryEntry) async -> Bool
    @State private var search=""
    @State private var selecting=false
    @State private var selectionTask:Task<Void,Never>?
    private var entries:[WorkflowHistoryEntry] {
        history.listing.entries.filter {entry in
            entry.verification == .recordedV2 && (!completedOnly || entry.status == .completed)
                && (search.isEmpty || [entry.runID ?? "",entry.reference.directoryID.uuidString,entry.finishedAt ?? "",entry.status?.rawValue ?? ""]
                    .contains {$0.localizedCaseInsensitiveContains(search)})
        }
    }
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {Text(title).font(.headline);Spacer();Button("Done",action:onClose).buttonStyle(FlatButtonStyle())}
            Text(explanation).font(.caption).foregroundStyle(MLTheme.muted)
            HStack {
                TextField("Search date, status or run ID",text:$search).flatField()
                Button("Refresh") {history.refresh()}.buttonStyle(FlatButtonStyle()).disabled(!active || selecting || history.isRefreshing)
            }
            if history.isRefreshing || selecting {ProgressView(selecting ? "Verifying saved result…":"Loading history…").controlSize(.small)}
            ThemedScrollView {
                LazyVStack(alignment:.leading,spacing:8) {
                    ForEach(entries,id:\.reference) {entry in
                        Button {
                            guard active,!selecting else {return}
                            selecting=true
                            selectionTask=Task {
                                let accepted=await onSelect(entry)
                                selecting=false
                                if accepted && !Task.isCancelled {onClose()}
                            }
                        } label: {
                            VStack(alignment:.leading,spacing:4) {
                                Text(entry.finishedAt ?? "Completion time unavailable").font(.caption)
                                Text("\(entry.status?.rawValue.capitalized ?? "Recorded") · \((entry.runID ?? entry.reference.directoryID.uuidString).prefix(12))")
                                    .font(.caption2).foregroundStyle(MLTheme.muted)
                            }.padding(10).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel)
                        }.buttonStyle(.plain).disabled(!active || selecting)
                            .accessibilityLabel("Select saved run \(entry.runID ?? entry.reference.directoryID.uuidString)")
                    }
                    if entries.isEmpty && !history.isRefreshing {Text(emptyMessage).font(.caption).foregroundStyle(MLTheme.muted)}
                    ForEach(Array(history.listing.issues.enumerated()),id:\.offset) {_,issue in Text(issue).font(.caption2).foregroundStyle(MLTheme.rose)}
                }
            }
            Text(history.statusMessage).font(.caption2).foregroundStyle(MLTheme.muted)
            Text(status).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
        }.padding(20).frame(width:520,height:450).background(MLTheme.background)
            .onDisappear {selectionTask?.cancel()}
    }
}
