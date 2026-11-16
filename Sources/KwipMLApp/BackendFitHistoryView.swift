import SwiftUI
import AppKit
import KwipMLCore

struct BackendFitHistoryView: View {
    @ObservedObject var history: BackendFitHistoryStore
    var selectedEntryID:String? = nil
    var historyPolicy=RunHistoryPolicy.all
    var tableViewSession:(BackendFitHistoryStore.Entry,TableViewQuerySession.Artifact)->TableViewQuerySession? = {_,_ in nil}
    @State private var predictionQuery:TableViewQuerySession?
    @State private var tableSelection:(entry:String,path:String)?
    @State private var openTask:Task<Void,Never>?
    @State private var actionMessage = ""
    private var entries:[BackendFitHistoryStore.Entry] {history.entries.filter {selectedEntryID == nil || $0.id == selectedEntryID}}
    var body: some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {Text("Backend models").font(.headline);Spacer();Button("Refresh backend history") {history.reload()}.buttonStyle(FlatButtonStyle())}
            if !history.status.isEmpty {Text(history.status).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
            if !actionMessage.isEmpty {Text(actionMessage).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
            if entries.isEmpty {Text("No retained backend record is available. Refresh History to recheck; current canvas output is never substituted.").font(.caption).foregroundStyle(MLTheme.muted)}
            ForEach(entries) {entry in
                if historyPolicy.componentOutputs,entry.artifacts.contains(where:{$0.path=="predictions.json"}) {
                    HStack {
                        Button("Output") {openPredictions(entry)}.buttonStyle(FlatButtonStyle(prominent:tableSelection?.entry==entry.id && tableSelection?.path=="predictions.json"))
                        if entry.artifacts.contains(where:{$0.path=="profiles.json"}) {Button("Profiles"){openPredictions(entry,path:"profiles.json")}.buttonStyle(FlatButtonStyle(prominent:tableSelection?.entry==entry.id && tableSelection?.path=="profiles.json"))}
                        if entry.artifacts.contains(where:{$0.path=="confusion.json"}) {Button("Confusion"){openPredictions(entry,path:"confusion.json")}.buttonStyle(FlatButtonStyle(prominent:tableSelection?.entry==entry.id && tableSelection?.path=="confusion.json"))}
                        Button("Metrics") {openTask?.cancel();predictionQuery?.invalidate();predictionQuery=nil;tableSelection=nil}.buttonStyle(FlatButtonStyle(prominent:predictionQuery == nil))
                    }
                }
            }
            if let predictionQuery,historyPolicy.componentOutputs {TableViewOutputView(session:predictionQuery)}
            else {ThemedScrollView {
                VStack(alignment:.leading,spacing:8) {ForEach(entries) {entry in
                    VStack(alignment:.leading,spacing:8) {
                        HStack {Text(entry.document.name).font(.caption.weight(.semibold));Text(entry.backendID).font(.caption).foregroundStyle(MLTheme.muted);Spacer();Text(entry.completedAt,format:.dateTime.month().day().hour().minute()).font(.caption2).foregroundStyle(MLTheme.muted)}
                        Text("Captured revision \(entry.document.revision)").font(.caption).foregroundStyle(MLTheme.muted)
                        if historyPolicy.metricsAndDiagnostics {
                            if entry.evaluation["task"] == .string("prebuilt") {PrebuiltResultsMetrics(manifest:["evaluation":.object(entry.evaluation)])}
                            else if entry.evaluation["confusionMatrix"] != nil || entry.evaluation["operation"] == .string("predict") {
                                DocumentClassificationResults(manifest:["operation":entry.evaluation["operation"] ?? .string("fit"),"evaluation":.object(entry.evaluation)])
                            } else {ForEach(entry.evaluation.keys.sorted(),id:\.self) {key in
                                HStack {Text(key).foregroundStyle(MLTheme.muted);Spacer();Text(render(entry.evaluation[key]))}.font(.system(size:11,design:.monospaced))
                            }}
                        }
                        HStack {if entry.artifacts.contains(where:{$0.path=="portable/state.json"}) {Button("Reveal portable package") {reveal(entry,portable:true)}.buttonStyle(FlatButtonStyle())};Button("Reveal run artifacts") {reveal(entry,portable:false)}.buttonStyle(FlatButtonStyle())}
                    }.padding(10).background(MLTheme.panel)
                }}
            }}
        }.task {await history.reloadAndWait()}
        .onChange(of:selectedEntryID) {_,_ in openTask?.cancel();predictionQuery?.invalidate();predictionQuery=nil;tableSelection=nil}
        .onChange(of:historyPolicy.componentOutputs) {_,allowed in if !allowed {openTask?.cancel();predictionQuery?.invalidate();predictionQuery=nil;tableSelection=nil}}
        .onDisappear {openTask?.cancel();predictionQuery?.invalidate();predictionQuery=nil;tableSelection=nil}
    }
    private func openPredictions(_ entry:BackendFitHistoryStore.Entry,path:String="predictions.json") {
        actionMessage="Verifying recorded prediction table…"
        openTask?.cancel()
        openTask=Task {
            do {
                let verified=try await history.verify(entry)
                try Task.checkCancellation()
                guard historyPolicy.componentOutputs,(selectedEntryID==nil || selectedEntryID==entry.id) else {return}
                guard selectedEntryID==nil || selectedEntryID==entry.id,let file=verified.artifacts.first(where:{$0.path==path}) else {throw WorkflowArchiveError("Recorded prediction table is unavailable.")}
                let artifact=TableViewQuerySession.Artifact(indexURL:verified.artifactDirectory.appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")
                predictionQuery?.invalidate();predictionQuery=tableViewSession(verified,artifact)
                tableSelection=predictionQuery == nil ? nil:(entry.id,path)
                actionMessage=predictionQuery == nil ? "Recorded table viewer is unavailable.":"Verified recorded prediction table. No fit or delivery was rerun."
            }catch {actionMessage=error.localizedDescription}
        }
    }
    private func reveal(_ entry:BackendFitHistoryStore.Entry,portable:Bool) {
        actionMessage="Verifying saved output…"
        Task {do {let verified=try await history.verify(entry);NSWorkspace.shared.activateFileViewerSelecting([portable ? verified.artifactDirectory.appendingPathComponent("portable"):verified.artifactDirectory]);actionMessage="Verified saved output."} catch {actionMessage=error.localizedDescription}}
    }
    private func render(_ value:JSONValue?)->String {
        guard let value else {return "—"}
        switch value {case .number(let n):return String(format:"%.6g",n);case .integer(let n):return String(n);case .string(let s):return s;default:return "Recorded structured metric"}
    }
}
