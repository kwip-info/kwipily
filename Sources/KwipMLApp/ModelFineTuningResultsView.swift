import SwiftUI
import KwipMLCore

struct ModelFineTuningMetricsView:View {
    @ObservedObject var store:ModelLoaderStore
    let address:WorkflowNodeAddress
    private var points:[TrainingProgressPoint] {store.trainingAddress==address ? store.trainingProgress.points:[]}
    private var epochs:[EpochMetric] {store.trainingAddress==address ? store.trainingMetrics:[]}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if store.trainingAddress != address,let reference=store.reference(address),let record=store.trainingRecords.first(where:{$0.id==reference.directoryID}) {
                ModelFineTuningHistoryView(record:record,root:store.root)
            } else {
            if store.trainingAddress==address {WorkflowResultsStatusStrip(title:"Fine-tuning",phase:store.trainingPhase,message:store.message(for:address),isRunning:store.isBusy && store.activeAddress==address)}
            TrainingMetricsChart(metrics:epochs,progress:points,observed:store.trainingProgress.observed,running:store.isBusy && store.activeAddress==address)
            }
        }
    }
}
struct ModelFineTuningHistoryView:View {
    let record:ModelFineTuningRecord
    let root:URL
    @State private var progress:[TrainingProgressPoint]=[]
    @State private var observed=0
    @State private var report:[String:JSONValue]?
    @State private var error:String?
    private var metrics:[EpochMetric] {
        guard case .array(let rows)=report?["metrics"] else {return []}
        return rows.compactMap {value in guard case .object(let row)=value,let epoch=row["epoch"]?.integerValue,let train=row["trainingLoss"]?.doubleValue,let valid=row["validationLoss"]?.doubleValue else {return nil};return .init(epoch:Int(epoch),trainLoss:train,validationLoss:valid,accuracy:nil)}
    }
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if let error {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
            else if let report {
                Text("Verified fine-tuning · "+String(record.operationID.prefix(8))).font(.caption.weight(.semibold))
                Text("Selected epoch \(report["selectedEpoch"]?.integerValue ?? 0) · completed epoch \(report["completedEpoch"]?.integerValue ?? 0)").font(.caption).foregroundStyle(MLTheme.muted)
                TrainingMetricsChart(metrics:metrics,progress:progress,observed:observed,recorded:true)
            } else {ProgressView("Verifying model and training evidence…")}
        }.task(id:record.id) {
            do {let value=try await Task.detached(priority:.utility) {()->([String:JSONValue],TrainingProgressArchive?) in
                let report=try record.report(root:root)
                let progress=try record.progressFile.map{try TrainingProgressArchive.decode(DiagnosticFileIO.read(root:record.directory(root),parts:WorkflowArchiveCodec.pathParts($0.path),limit:8_000_000))}
                return (report,progress)
            }.value;guard !Task.isCancelled else {return};report=value.0;progress=value.1?.points ?? [];observed=value.1?.observed ?? 0}
            catch {self.error=error.localizedDescription}
        }
    }
}

struct ModelOperationActivityView:View {
    @ObservedObject var store:ModelLoaderStore
    var body:some View {WorkflowMessageActivityView(messages:[store.status])}
}

struct ModelCausalEvaluationSummaryView:View {
    let summary:[String:JSONValue]
    var body:some View {VStack(alignment:.leading,spacing:10){
        if summary["kind"] == .string("typedDecisionEvaluation") {
            Text("Typed-decision evaluation · "+(summary["role"]?.stringValue ?? "").capitalized).font(.headline)
            Text("Distribution loss · \(summary["loss"]?.doubleValue ?? 0) · Brier · \(summary["brier"]?.doubleValue ?? 0) · Accuracy · \(summary["accuracy"]?.doubleValue ?? 0)").font(.system(.body,design:.monospaced))
            Text("\(summary["scoredItems"]?.integerValue ?? 0) scored decisions. Calibration/review status remains in the standard Predictions table.").font(.caption)
        } else {
        Text("Causal evaluation · "+(summary["role"]?.stringValue ?? "").capitalized).font(.headline)
        Text("Token-weighted loss · "+String(format:"%.6g",summary["loss"]?.doubleValue ?? 0)+" · Perplexity · "+(summary["perplexity"]?.doubleValue.map{String(format:"%.6g",$0)} ?? "Overflow / unavailable")).font(.system(.body,design:.monospaced)).textSelection(.enabled)
        Text("Scored tokens · \(summary["scoredTokens"]?.integerValue ?? 0). Predictions are a standard table output.").font(.caption)
        }
        Text("Local exposure tracking is advisory. Earlier history may be unknown. Prior inspections: \(summary["priorInspections"]?.integerValue ?? 0); overlapping observations: \(summary["priorOverlap"]?.integerValue ?? 0). Repeated evaluation is not a pristine holdout.").font(.caption).foregroundStyle(MLTheme.muted)
    }}
}
struct ModelCausalEvaluationHistoryView:View {
    let record:ModelCausalEvaluationRecord
    @ObservedObject var workspace:WorkspaceStore
    let isActive:()->Bool
    @State private var summary:[String:JSONValue]?
    @State private var error:String?
    @State private var query:TableViewQuerySession?
    @State private var showPredictions=false
    var body:some View {VStack(alignment:.leading,spacing:10){
        if let error{Text(error).foregroundStyle(MLTheme.rose)}else if let summary{
            HStack{Button("Metrics"){showPredictions=false}.buttonStyle(FlatButtonStyle(prominent:!showPredictions));Button("Predictions"){showPredictions=true}.buttonStyle(FlatButtonStyle(prominent:showPredictions));Spacer()}
            if showPredictions,let query{TableViewOutputView(session:query,executionBusy:workspace.isBusy)}else{ModelCausalEvaluationSummaryView(summary:summary)}
        }else{ProgressView("Verifying evaluation evidence…")}
    }.task(id:record.id){
        let root=workspace.workspaceDirectory,exposure=workspace.applicationDirectory
        do{let value=try await Task.detached(priority:.utility){try record.disclose(root:root,exposureRoot:exposure)}.value;guard !Task.isCancelled,isActive()else{return};summary=value
            if let file=record.files.first(where:{$0.path=="output/table.json"}){query=workspace.makeTableViewQuerySession(artifact:.init(indexURL:record.directory(root).appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table"),projectID:record.document.projectID,accepts:isActive)}
        }catch{self.error=error.localizedDescription}
    }.onDisappear{query?.cancel();query=nil}
    }
}
