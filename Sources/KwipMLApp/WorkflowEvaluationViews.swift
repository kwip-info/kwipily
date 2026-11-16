import SwiftUI
import KwipMLCore

struct WorkflowFrozenModelSelectionView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var history:WorkflowHistoryStore
    @State private var choosing=false
    private var selected:WorkflowFrozenModelReference? {host.frozenBindings.reference(for:context.authority.address)}
    private var active:Bool {context.isCurrent && !context.isBusy && !host.isRetired && !host.isStagingEvaluation && host.owner === context.authority.owner && history.isActive}
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            SectionLabel(title:"Saved run")
            if let selected {
                Text("Saved run · \(selected.runID.prefix(12))").font(.caption)
                Text("Checkpoint · \(selected.checkpointSHA256.prefix(16))").font(.system(size:10,design:.monospaced)).textSelection(.enabled)
                Text("Weights and training-fitted preparation are frozen. Evaluation uses the selected archived population, even if the original CSV is gone.").font(.caption2).foregroundStyle(MLTheme.muted)
                Button("Clear selection") {_=host.clearFrozenModel(for:context.authority.address,lease:context.authority.token)}.buttonStyle(FlatButtonStyle())
            } else {Text("Choose a completed prepared model from this project.").font(.caption).foregroundStyle(MLTheme.muted)}
            Button("Choose saved run…") {history.refresh();choosing=true}.buttonStyle(FlatButtonStyle()).disabled(!active)
        }.disabled(!active)
        .sheet(isPresented:$choosing) {
            WorkflowHistoryPicker(title:"Choose saved run",explanation:"Selection independently verifies the saved model and preparation. Connect the matching archived Validation or Test output to Evaluate.",emptyMessage:"Train a model with held-out partitions to create a selectable checkpoint.",history:history,active:active,completedOnly:true,status:host.statusMessage,onClose:{choosing=false}) {entry in
                guard active else {return false}
                return await host.selectFrozenModel(entry.reference,for:context.authority.address)
            }
        }
        .onChange(of:context.isCurrent) {_,value in if !value {choosing=false}}
    }
}

struct WorkflowEvaluationLaunchView:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @ObservedObject var host:WorkflowWorkspaceSession
    var onStarted:()->Void
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            Button("Evaluate") {
                if host.evaluate(address:context.authority.address) {onStarted()}
            }.buttonStyle(FlatButtonStyle(prominent:true))
                .disabled(!context.isCurrent || context.isBusy || host.isStagingEvaluation || host.isRetired)
                .accessibilityIdentifier("workflow.evaluation.run")
            Text("Evaluation preserves the captured weights and preparation. If its connected Train has no current output, Train runs first. Inspection is recorded across model revisions.").font(.caption2).foregroundStyle(MLTheme.muted)
            ResultsSection(title:"Evaluation guide",symbol:"info.circle") { EvaluationGuideView().padding(.top, 8) }
                .accessibilityIdentifier("workflow.evaluation.guide")
            Text(host.statusMessage).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
        }
    }
}

struct WorkflowEvaluationHistoryList:View {
    @ObservedObject var store:WorkflowEvaluationHistoryStore
    var onSelect:()->Void = {}
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {SectionLabel(title:"Evaluations");Spacer();Button("Refresh") {store.refresh()}.buttonStyle(.plain).disabled(store.isRefreshing || !store.isActive)}
            if store.listing.entries.isEmpty {Text("No saved evaluations loaded.").font(.caption).foregroundStyle(MLTheme.muted)}
            ForEach(store.listing.entries,id:\.reference) {entry in
                Button {onSelect();store.select(entry.reference)} label: {
                    VStack(alignment:.leading,spacing:4) {
                        HStack {
                            Text([entry.partitionRole?.capitalized,entry.status?.rawValue.capitalized ?? state(entry.verification)].compactMap {$0}.joined(separator:" · ")).font(.caption.weight(.medium))
                            Spacer()
                            if store.selectedReference==entry.reference {Image(systemName:"checkmark").foregroundStyle(MLTheme.rose)}
                        }
                        if let date=entry.finishedAt {Text(date.formatted(date:.abbreviated,time:.standard)).font(.caption2).foregroundStyle(MLTheme.muted)}
                        Text(String(entry.reference.evaluationID.uuidString.prefix(12))).font(.system(size:10,design:.monospaced))
                        Text(entry.message ?? "Select to verify and record inspection.").font(.caption2).foregroundStyle(MLTheme.muted)
                    }.frame(maxWidth:.infinity,alignment:.leading).padding(8)
                        .background(store.selectedReference==entry.reference ? MLTheme.panel:Color.clear)
                }.buttonStyle(.plain).disabled(!store.isActive)
            }
            ForEach(Array(store.listing.issues.enumerated()),id:\.offset) {_,message in Text(message).font(.caption2).foregroundStyle(MLTheme.rose)}
        }.onAppear {store.refresh()}
    }
    private func state(_ value:WorkflowEvaluationHistoryVerification)->String {
        switch value {case .recordedUnverified:return "Recorded";case .inUse:return "In use";case .incomplete:return "Incomplete";case .unavailable:return "Unavailable"}
    }
}

struct WorkflowEvaluationResultsView:View {
    @ObservedObject var store:WorkflowEvaluationHistoryStore
    var historyPolicy=RunHistoryPolicy.all
    @State private var page=0
    private let pageSize=25
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {
                Text("Evaluation results").font(.caption.weight(.semibold))
                Spacer()
                if store.disclosure != nil {
                    Button("Export evaluation…") {_=store.prepareExport()}.buttonStyle(FlatButtonStyle())
                        .disabled(store.isLoading || !store.isActive)
                        .accessibilityIdentifier("workflow.evaluation.export")
                }
                if store.isLoading {ProgressView().controlSize(.small);Button("Cancel loading") {store.cancelLoading()}.buttonStyle(.plain)}
            }
            Text(store.statusMessage).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            if let disclosed=store.disclosure {
                details(disclosed)
            } else if !store.isLoading {
                Text("Select a saved evaluation to inspect the selected archived population. Listing history does not reveal metrics.").font(.caption2).foregroundStyle(MLTheme.muted)
            }
        }.onChange(of:store.selectedReference) {_,_ in page=0}
            .accessibilityIdentifier("workflow.evaluation.results")
            .sheet(isPresented:Binding(get:{store.export != nil},set:{if !$0 {store.closeExport()}})) {
                if let export=store.export {WorkflowEvaluationExportView(export:export,onClose:{store.closeExport()})}
            }
    }
    private func number(_ value:Double)->String {String(format:"%.5g",value)}
    private func details(_ result:WorkflowEvaluationDisclosure.DisclosedEvaluation)->some View {
        let summary=result.value.summary,prior=result.priorExposure
        let count=result.value.predictions.count
        let start=min(page*pageSize,max(0,count-1)),end=min(start+pageSize,count)
        return VStack(alignment:.leading,spacing:10) {
            if historyPolicy.metricsAndDiagnostics {
            HStack(spacing:18) {
                metric(result.value.manifest.settings.partitionTitle+" rows",String(summary.evaluatedCount))
                if let accuracy=result.value.accuracy {metric("Accuracy",String(format:"%.2f%%",accuracy*100))}
                if let score=result.value.score {metric(result.value.objectiveEvaluation == nil ? "Mean loss":"Objective",number(score))}
                metric("Device",summary.device.uppercased())
                metric("Batch",String(summary.batchSize))
            }
            }
            Text("Saved \(result.outcome.finishedAt.formatted()) · Saved checkpoint \(result.value.manifest.parent.checkpointSHA256.prefix(12))")
                .font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            Text("Prior local inspections: \(prior.inspectionCount) of this population; \(prior.checkpointInspectionCount) with these weights. Previously disclosed overlapping rows: \(prior.sameSourceInspectedOverlap).")
                .font(.caption2).foregroundStyle(prior.inspectionCount>0 || prior.sameSourceInspectedOverlap>0 ? MLTheme.rose:MLTheme.muted)
            Text("Exposure before local tracking is unknown. Changed source files may contain previously viewed observations that cannot be matched.")
                .font(.caption2).foregroundStyle(MLTheme.muted)
            Divider()
            if historyPolicy.metricsAndDiagnostics && summary.schemaVersion == 1 {
            HStack(alignment:.top,spacing:24) {
                Grid(alignment:.leading,horizontalSpacing:14,verticalSpacing:5) {
                    GridRow {Text("Actual / predicted");Text("0");Text("1")}
                    ForEach(0..<2,id:\.self) {row in GridRow {Text("\(row)");Text("\(summary.confusionMatrix[row][0])");Text("\(summary.confusionMatrix[row][1])")}}
                }.font(.caption).accessibilityElement(children:.ignore).accessibilityLabel(WorkflowEvaluationPresentation.confusionDescription(summary.confusionMatrix,partitionTitle:result.value.manifest.settings.partitionTitle))
                VStack(alignment:.leading,spacing:4) {
                    ForEach(summary.perClass,id:\.classIndex) {value in
                        Text("Class \(value.classIndex) · precision \(value.precision.map(number) ?? "unavailable") · recall \(value.recall.map(number) ?? "unavailable") · support \(value.support)").font(.caption2)
                        ForEach(WorkflowEvaluationPresentation.explanations(value,partitionTitle:result.value.manifest.settings.partitionTitle),id:\.self) {reason in Text(reason).font(.caption2).foregroundStyle(MLTheme.muted)}
                    }
                }
            }
            }
            if historyPolicy.componentOutputs && summary.schemaVersion == 2 {
                Text("Full predictions are available in this component’s Predictions table output.").font(.caption2).foregroundStyle(MLTheme.muted)
            }
            if historyPolicy.componentOutputs && summary.schemaVersion == 1 {
            Divider()
            HStack {
                Text("Predictions · \(count==0 ? 0:start+1)–\(end) of \(count)").font(.caption.weight(.semibold))
                Spacer()
                Button("Previous") {page=max(0,page-1)}.buttonStyle(.plain).disabled(page==0)
                Button("Next") {page+=1}.buttonStyle(.plain).disabled(end>=count)
            }
            Grid(alignment:.leading,horizontalSpacing:16,verticalSpacing:5) {
                GridRow {Text("Observation");Text("Actual");Text("Predicted");Text("Logits")}.foregroundStyle(MLTheme.muted)
                ForEach(Array(result.value.predictions[start..<end]),id:\.observationID) {row in
                    GridRow {
                        Text(row.observationID).lineLimit(1).truncationMode(.middle).frame(maxWidth:300,alignment:.leading).help(row.observationID)
                        Text("\(row.target)");Text("\(row.predicted)")
                        Text(row.logits.map(number).joined(separator:", "))
                    }
                }
            }.font(.system(size:11,design:.monospaced)).textSelection(.enabled)
            }
        }
    }
    private func metric(_ title:String,_ value:String)->some View {
        VStack(alignment:.leading,spacing:3) {Text(title).font(.caption2).foregroundStyle(MLTheme.muted);Text(value).font(.caption.weight(.semibold))}
    }
}

struct WorkflowEvaluationDock:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var store:WorkflowEvaluationHistoryStore
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            if host.isStagingEvaluation {Text("Verifying trained model and archived evaluation data…").font(.caption)}
            if let session=host.evaluation {WorkflowEvaluationLiveStatus(session:session)}
            if host.evaluationOutcomes.first?.outcome.status == .completed {
                Button("Inspect latest evaluation") {selectLatest()}.buttonStyle(FlatButtonStyle())
                    .disabled(store.isLoading || !store.isActive)
            }
            WorkflowEvaluationResultsView(store:store)
            WorkflowEvaluationHistoryList(store:store)
        }
    }
    private func selectLatest() {
        guard let result=host.evaluationOutcomes.first,result.outcome.status == .completed,result.recordError==nil,
              let id=UUID(uuidString:result.outcome.evaluationID) else {return}
        let reference=WorkflowEvaluationHistoryReference(projectID:result.outcome.projectID,evaluationID:id)
        if store.selectedReference != reference {store.select(reference)}
    }
}
private struct WorkflowEvaluationLiveStatus:View {
    @ObservedObject var session:WorkflowEvaluationSession
    var body:some View {
        HStack {
            Text(session.statusMessage).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            Spacer()
            if session.totalCount>0 {Text("\(session.completedCount) / \(session.totalCount)").font(.caption.monospacedDigit())}
        }
    }
}
