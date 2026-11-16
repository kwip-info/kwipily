import SwiftUI
import KwipMLCore

/// Recorded history is explicitly loaded; live observations have a separate owner
/// and are never substituted for the selected archive's verified measurements.
struct WorkflowHistoryView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    @ObservedObject var history:WorkflowHistoryStore
    @ObservedObject var workspace:WorkspaceStore
    @State private var captureTab="metrics"
    @State private var canvasTab=true
    @State private var detail="result"
    @State private var showingEvaluation=false
    @State private var selectedGroup:String?
    @State private var groupRefresh=0
    @State private var selectedRecord:HistoryTableCatalog.Row?
    @State private var selectedBackendID:String?
    @State private var selectedPreparationID:String?
    @State private var selectedFineTuningID:String?
    @State private var selectedCausalEvaluationID:String?
    @State private var selectedSelfPlayID:String?
    @State private var selectedGameID:String?
    @State private var selectedGenerationID:String?
    @State private var queueExpanded=false
    @State private var selectedMetric:Int?
    var body:some View {
        VStack(spacing:0) {
            HStack(spacing:10) {
                Text("History").font(.system(size:18,weight:.semibold))
                Spacer()
                if history.isLoading || history.isRefreshing || host.evaluationHistory.isLoading {
                    ProgressView().controlSize(.small)
                    Button("Cancel loading") {history.cancelLoading();host.evaluationHistory.cancelLoading()}.buttonStyle(FlatButtonStyle())
                }
                Button {refreshHistory()} label: {Image(systemName:"arrow.clockwise").frame(width:20,height:20)}
                    .buttonStyle(FlatButtonStyle()).accessibilityLabel("Refresh history")
                    .disabled(!history.isActive || history.isRefreshing)
            }.padding(.horizontal,16).padding(.top,12).padding(.bottom,8)
            HStack {
                Button("Canvas versions") {canvasTab=true}.buttonStyle(FlatButtonStyle(prominent:canvasTab))
                Button("Recorded runs") {canvasTab=false}.buttonStyle(FlatButtonStyle(prominent:!canvasTab))
                Spacer()
            }.padding(.horizontal,16).padding(.bottom,8)
            if canvasTab {
                CanvasVersionHistoryView(owner:host.owner,workspace:workspace)
            } else {
            VSplitView {
                HistoryRecordsTable(history:history,evaluations:host.evaluationHistory,backends:host.backendHistory,preparations:host.dataPreparationHistory,
                    selfPlay:host.selfPlayRecords,games:host.gameRecords,generations:host.generationRecords,fineTuning:host.modelLoader.trainingRecords,causalEvaluations:host.modelLoader.evaluationRecords,root:workspace.workspaceDirectory,refreshKey:groupRefresh,selectedTarget:selectedRecord?.target,select:selectRecord)
                    .frame(minHeight:200,idealHeight:420,maxHeight:.infinity)
                VStack(alignment:.leading,spacing:0) {
                    HStack {
                        Text(selectedRecord?.title ?? "Record details").font(.system(size:14,weight:.semibold)).lineLimit(1)
                        Spacer()
                        if let baseline=history.comparisonBaseline {
                            Text("Baseline · "+String(baseline.outcome.runID.prefix(8))).font(.caption).foregroundStyle(MLTheme.muted)
                            Button("Clear baseline") {history.clearComparisonBaseline();detail="result"}.buttonStyle(FlatButtonStyle())
                        }
                    }.padding(.horizontal,16).padding(.vertical,10).background(MLTheme.panel)
                    Rectangle().fill(MLTheme.rule).frame(height:1)
                    selectedDetails.frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading).padding(12)
                }.frame(minHeight:180,idealHeight:300,maxHeight:.infinity)
            }
            }
            Rectangle().fill(MLTheme.rule).frame(height:1)
            DisclosureGroup("Session and queue",isExpanded:$queueExpanded) {
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:10) {
                        WorkflowQueueView(queue:host.executionQueue)
                        if let execution=host.execution {WorkflowLiveExecutionSummary(execution:execution)}
                    }.padding(.vertical,8)
                }.frame(maxHeight:200)
            }.font(.system(size:11)).padding(.horizontal,16).padding(.vertical,8)
        }
        .onAppear {refreshHistory()}
        .onChange(of:host.outcomes.count) {_,_ in history.refresh();groupRefresh+=1}
        .onChange(of:history.selectedReference,initial:true) {_,reference in
            guard let reference else {return}
            canvasTab=false;detail="result";selectedGroup=nil;showingEvaluation=false;selectedBackendID=nil;selectedPreparationID=nil
            if selectedRecord?.target != .training(reference) {
                selectedRecord=HistoryTableCatalog.rows(training:history.listing.entries,evaluation:[],experiments:[],backends:[]).first {$0.target == .training(reference)}
            }
        }
    }
    private func refreshHistory() {
        history.refresh();host.evaluationHistory.refresh();host.backendHistory.reload();host.dataPreparationHistory.reload();host.reloadGenerationRecords();host.reloadGameRecords();host.reloadSelfPlayRecords();host.modelLoader.reloadTraining();groupRefresh+=1
    }
    private func selectRecord(_ record:HistoryTableCatalog.Row) {
        guard history.isActive else {return}
        selectedRecord=record;detail="result";selectedGroup=nil;selectedBackendID=nil;selectedPreparationID=nil;selectedGenerationID=nil;selectedSelfPlayID=nil;selectedGameID=nil;selectedFineTuningID=nil;selectedCausalEvaluationID=nil;showingEvaluation=false
        switch record.target {
        case .training(let reference):host.evaluationHistory.clearSelection();history.select(reference)
        case .evaluation(let reference):history.cancelLoading();showingEvaluation=true;host.evaluationHistory.select(reference)
        case .experiment(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedGroup=id
        case .backend(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedBackendID=id
        case .causalEvaluation(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedCausalEvaluationID=id
        case .fineTuning(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedFineTuningID=id
        case .selfPlay(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedSelfPlayID=id
        case .gameSession(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedGameID=id
        case .generation(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedGenerationID=id
        case .preparation(let id):history.cancelLoading();host.evaluationHistory.clearSelection();selectedPreparationID=id
        }
    }
    @ViewBuilder private var selectedDetails:some View {
        if let selectedSelfPlayID,let record=host.selfPlayRecords.first(where:{$0.id==selectedSelfPlayID}) {SelfPlayHistoryView(record:record,workspace:workspace,isActive:{history.isActive && self.selectedSelfPlayID==record.id}).id(record.id)
        } else if let selectedGameID,let record=host.gameRecords.first(where:{$0.id==selectedGameID}) {GameEpisodeHistoryView(record:record,workspace:workspace,isActive:{history.isActive && self.selectedGameID==record.id}).id(record.id)
        } else if let selectedCausalEvaluationID,let record=host.modelLoader.evaluationRecords.first(where:{$0.id==selectedCausalEvaluationID}) {ModelCausalEvaluationHistoryView(record:record,workspace:workspace,isActive:{history.isActive && self.selectedCausalEvaluationID==record.id}).id(record.id)
        } else if let selectedFineTuningID,let record=host.modelLoader.trainingRecords.first(where:{$0.id==selectedFineTuningID}) {
            ModelFineTuningHistoryView(record:record,root:workspace.workspaceDirectory).id(record.id)
        } else if let selectedGenerationID,let record=host.generationRecords.first(where:{$0.id==selectedGenerationID}) {
            WorkflowGenerationHistoryView(record:record,workspace:workspace,isActive:{history.isActive && self.selectedGenerationID==record.id}).id(record.id)
        } else if let selectedPreparationID {
            if let entry = host.dataPreparationHistory.entries.first(where: { $0.id == selectedPreparationID }) {
                RunHistoryDetailsGate(root:workspace.workspaceDirectory,projectID:history.projectID,runID:entry.id) {policy in
                    if policy.componentOutputs {
                WorkflowDataPreparationHistoryView(history: host.dataPreparationHistory, entry: entry, isActive: { history.isActive },
                    tableViewSession: { captured, artifact in
                        workspace.makeTableViewQuerySession(artifact: artifact, projectID: captured.document.projectID,
                            accepts: { history.isActive && selectedPreparationID == captured.id })
                    })
                    .id(selectedPreparationID)
                    } else {Text("Component outputs not captured").font(.caption).foregroundStyle(MLTheme.muted)}
                }
            } else {
                notice("Preparation record unavailable", reason: "Refresh History to recheck the saved record. No current canvas output has been substituted.", error: nil)
            }
        } else if let selectedBackendID {
            RunHistoryDetailsGate(root:workspace.workspaceDirectory,projectID:history.projectID,runID:selectedBackendID) {policy in
                BackendFitHistoryView(history:host.backendHistory,selectedEntryID:selectedBackendID,historyPolicy:policy,tableViewSession:{entry,artifact in
                    workspace.makeTableViewQuerySession(artifact:artifact,projectID:entry.document.projectID,accepts:{history.isActive && self.selectedBackendID==entry.id})
                }).padding(12)
            }
        } else if showingEvaluation {
            if let reference=host.evaluationHistory.selectedReference {
                RunHistoryDetailsGate(root:workspace.workspaceDirectory,projectID:history.projectID,runID:reference.evaluationID.uuidString) {policy in
                    ThemedScrollView {WorkflowEvaluationResultsView(store:host.evaluationHistory,historyPolicy:policy).padding(12)}
                }
            }
        } else if let selectedGroup {
            WorkflowExperimentGroupDetail(root:workspace.workspaceDirectory,projectID:history.projectID,groupID:selectedGroup,
                sessionID:history.sessionID,isCurrent:{history.isActive},onRecoveryRefresh:{history.refresh();host.evaluationHistory.refresh();groupRefresh+=1}) {reference in
                self.selectedGroup=nil;showingEvaluation=false;selectedBackendID=nil;selectedPreparationID=nil;detail="result"
                selectedRecord=HistoryTableCatalog.rows(training:history.listing.entries,evaluation:[],experiments:[],backends:[]).first {$0.target == .training(reference)}
                history.select(reference)
            }.id(WorkflowExperimentRecoveryStore.Scope(workspace:workspace.workspaceDirectory,projectID:history.projectID,sessionID:history.sessionID,groupID:selectedGroup))
        } else if history.isLoading {
            VStack(alignment:.leading,spacing:10) {Text("Verifying selected archive…");Text("Captured source, registries, input snapshots and phase artifacts are checked before showing verified content.").font(.caption).foregroundStyle(MLTheme.muted)}.padding(12)
        } else if case .training = selectedRecord?.target,let selection=history.selection {
            switch selection {
            case .verified(let verified):verifiedDetails(verified)
            case .legacyUnverifiable(let legacy,let reason):
                notice("Legacy record · unverified",reason:reason,error:legacy.error)
            case .incomplete(let reason):notice("Incomplete archive",reason:reason,error:nil)
            case .unavailable(let reason):notice("Evidence unavailable",reason:reason,error:nil)
            }
        } else {notice("Select a history record",reason:"Selecting an entry verifies its archived evidence. Listing a receipt alone does not verify a completed model.",error:nil)}
    }
    private func notice(_ title:String,reason:String,error:String?)->some View {
        ThemedScrollView {
            VStack(alignment:.leading,spacing:12) {
                Text(title).font(.headline)
                Text(reason).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                if let error {Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            }.padding(12).frame(maxWidth:.infinity,alignment:.leading)
        }
    }
    private func verifiedDetails(_ verified:VerifiedWorkflowOutcome)->some View {
        VStack(alignment:.leading,spacing:12) {
            HStack(spacing:8) {
                Button("Captured result") {detail="result"}.buttonStyle(FlatButtonStyle(prominent:detail == "result"))
                if verified.modelContent != nil {Button("Saved model source") {detail="source"}.buttonStyle(FlatButtonStyle(prominent:detail == "source"))}
                if verified.outcome.preparationManifestSHA256 != nil {
                    Button("Saved preparation") {detail="preparation"}.buttonStyle(FlatButtonStyle(prominent:detail == "preparation"))
                }
                if verified.outcome.status == .completed, verified.modelContent != nil {
                    Button("Set as baseline") { _ = history.pinComparisonBaseline() }.buttonStyle(FlatButtonStyle())
                    if let baseline = history.comparisonBaseline, baseline.reference != verified.reference {
                        Button("Compare") { detail="compare" }.buttonStyle(FlatButtonStyle(prominent:detail == "compare"))
                    }
                }
                Spacer()
                Button("Verify again") {history.select(verified.reference)}.buttonStyle(FlatButtonStyle()).disabled(!history.isActive)
            }
            if detail == "support" {
                WorkflowSupportSummaryView(verified: verified, isActive: history.isActive).id(verified.reference)
            } else if detail == "compare", let baseline = history.comparisonBaseline,
               baseline.reference != verified.reference, verified.outcome.status == .completed, verified.modelContent != nil {
                WorkflowComparisonView(baseline:baseline,candidate:verified)
            } else if detail == "programExport",let export=history.programExport {
                WorkflowProgramExportView(export:export,onClose:{history.closeProgramExport();detail="source"})
            } else if detail == "export", let export = history.trainedExport {
                WorkflowTrainedExportView(export: export, workspace: workspace)
            } else if detail == "preparation" {
                WorkflowHistoricalPreparationView(history:history,workspace:workspace)
            } else if detail == "source",let content=verified.modelContent {
                HStack {
                    Text("PYTHON · VERIFIED ARCHIVED COPY").font(.system(size:9,weight:.medium)).foregroundStyle(MLTheme.muted)
                    Spacer()
                    Button("Export trained model…") {
                        if history.prepareTrainedExport() { detail="export" }
                    }.buttonStyle(FlatButtonStyle(prominent:true)).disabled(!history.isActive || workspace.isBusy)
                    Button("Export inference program…") {
                        if history.prepareProgramExport() {detail="programExport"}
                    }.buttonStyle(FlatButtonStyle()).disabled(!history.isActive || workspace.isBusy || !workspace.programExportRuntimeCapability.isAvailable || workspace.runtimeContractVersion == nil || verified.outcome.status != .completed || verified.plan.bagInput != nil || content.bagContext != nil)
                    Button("Copy") {guard history.isActive else {return};workspace.copyPythonCode(content.code)}.buttonStyle(FlatButtonStyle())
                    Button("Save .py…") {guard history.isActive else {return};workspace.savePythonCode(content.code)}.buttonStyle(FlatButtonStyle())
                }
                SyntaxHighlightedCode(source:content.code).overlay(Rectangle().stroke(MLTheme.rule))
                Text("This source defines the captured model. Trained weights are a separate artifact; current canvas edits do not rewrite this copy.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
                capturedResult(verified)
            }
        }.padding(.leading,12)
    }
    /// Presentation only: all content still belongs to this verified archive.
    private func capturedResult(_ verified:VerifiedWorkflowOutcome)->some View {
        RunHistoryDetailsGate(root:workspace.workspaceDirectory,projectID:history.projectID,runID:verified.outcome.runID) {policy in
            VStack(alignment:.leading,spacing:8) {
                HStack(spacing:8) {
                    Label("Verified",systemImage:"checkmark.shield").font(.caption).foregroundStyle(MLTheme.muted)
                    ForEach(["metrics","outputs","diagnostics","information"],id:\.self) {tab in
                        if tab == "information" || (tab == "outputs" ? policy.componentOutputs:policy.metricsAndDiagnostics) {
                            Button(tab.capitalized) {captureTab=tab}.buttonStyle(FlatButtonStyle(prominent:captureTab==tab))
                        }
                    }
                    Spacer()
                }
                if captureTab=="information" {
                    ThemedScrollView {
                        VStack(alignment:.leading,spacing:10) {
                            captureField("Started",verified.outcome.startedAt)
                            captureField("Finished",verified.outcome.finishedAt)
                            captureField("Revision",String(verified.capture.revision))
                            captureField("Run ID",verified.outcome.runID)
                            captureField("Request",verified.capture.requestIdentity)
                            Text("Verified archived evidence. Current source files and runtime are not revalidated. Required model, prepared-data and replay artifacts remain independent of optional History detail capture.").font(.caption).foregroundStyle(MLTheme.muted)
                            Button("Review support summary…") {detail="support"}.buttonStyle(FlatButtonStyle())
                        }.padding(8)
                    }
                } else if captureTab=="diagnostics",policy.metricsAndDiagnostics {
                    ThemedScrollView {VStack {partialCapture(verified);captureDiagnostics(verified)}}
                } else if captureTab=="outputs",policy.componentOutputs {
                    WorkflowHistoricalOutputsView(verified:verified,history:history,workspace:workspace)
                } else if policy.metricsAndDiagnostics,let content=verified.modelContent {
                    GeometryReader {viewport in
                        ThemedScrollView {TrainingMetricsChart(metrics:content.metrics,progress:content.trainingProgress?.points ?? [],observed:content.trainingProgress?.observed ?? 0,recorded:true,trainingControl:content.trainingControl)
                            .environment(\.trainingChartViewportHeight,viewport.size.height-8).padding(.bottom,8)}
                    }
                } else if policy.componentOutputs {
                    WorkflowHistoricalOutputsView(verified:verified,history:history,workspace:workspace)
                } else {Text("No completed metrics captured").font(.caption).foregroundStyle(MLTheme.muted);Spacer()}
            }
        }
    }
    private func captureField(_ title:String,_ value:String)->some View {
        HStack(alignment:.top,spacing:10) {
            Text(title).font(.system(size:11)).foregroundStyle(MLTheme.muted).frame(width:82,alignment:.leading)
            Text(value).font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                .frame(maxWidth:.infinity,alignment:.leading).fixedSize(horizontal:false,vertical:true)
        }
    }
    private func captureDiagnostics(_ verified:VerifiedWorkflowOutcome)->some View {
        MeasuredDiagnosticWorkbench(verified:verified,host:host,workspace:workspace,inspectSource:{detail="source"},inspectPreparation:{detail="preparation"})
            .id(verified.reference).padding(12).overlay(Rectangle().stroke(MLTheme.rule))
    }
    @ViewBuilder private func partialCapture(_ verified:VerifiedWorkflowOutcome)->some View {
        if let error=verified.outcome.error {Text(error).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        if let partial=verified.failedGradientEvidence {
            DisclosureGroup("Verified partial gradients · failed training") {
                let batches=partial.completedSummaries.flatMap(\.batches)+partial.interruption.batches
                Text("Captured after backward, before optimizer step. Training failed at epoch \(partial.interruption.epoch) during \(partial.interruption.stage). These are retained observations, not a completed model or diagnosis.").font(.caption).foregroundStyle(MLTheme.muted)
                row("Observed batches",String(batches.count))
                row("Finite element observations",String(batches.reduce(0){$0+$1.finiteElementCount}))
                row("Nonfinite element observations",String(batches.reduce(0){$0+$1.nonfiniteElementCount}))
                row("Missing element observations",String(batches.reduce(0){$0+$1.missingElementCount}))
                Text("Counts sum the captured batches; the same parameter can be observed more than once. Unsampled batches remain unknown.").font(.caption).foregroundStyle(MLTheme.muted)
            }
        } else if let notice=verified.outcome.failedGradientNotice {
            Text(notice).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
        } else if verified.outcome.status == .failed, verified.plan.settings.gradientCapture != nil {
            Text("No verified failed-gradient observations were retained for this run.").font(.caption).foregroundStyle(MLTheme.muted)
        }

    }
    private func row(_ title:String,_ value:String)->some View {
        VStack(alignment:.leading,spacing:3) {
            Text(title).font(.caption).foregroundStyle(MLTheme.muted)
            Text(value).font(.system(size:11,design:.monospaced)).textSelection(.enabled)
        }
    }
    private func verificationLabel(_ value:WorkflowHistoryVerification)->String {
        switch value {
        case .recordedV2:return "Recorded receipt · select to verify"
        case .legacyV1:return "Legacy evidence · not verified"
        case .incomplete:return "Interrupted or incomplete"
        case .unavailable:return "Unavailable or corrupt receipt"
        }
    }
}

struct WorkflowLiveExecutionSummary:View {
    @ObservedObject var execution:WorkflowExecutionSession
    var includesDiagnostics=true
    var compact = false
    var includesMetricHeader=true
    @ViewBuilder var body:some View {
        if compact {
            TrainingMetricsChart(metrics:execution.liveMetrics,progress:execution.liveTrainingProgress,observed:execution.trainingProgressObserved,running:execution.phase == .training,trainingControl:execution.trainingControlReport)
        } else {
        ResultsSection(title:"Current session · \(execution.phase.rawValue)",symbol:"waveform.path",initiallyExpanded:true) {
            VStack(alignment:.leading,spacing:10) {
                Text(execution.statusMessage).font(.caption).textSelection(.enabled)
                Text("Live observations belong to request \(execution.request.requestIdentity.prefix(12)). Select and verify a History entry to inspect its saved copy.").font(.caption).foregroundStyle(MLTheme.muted)
                TrainingMetricsChart(metrics:execution.liveMetrics,progress:execution.liveTrainingProgress,observed:execution.trainingProgressObserved,running:execution.phase == .training,trainingControl:execution.trainingControlReport)
            }.padding(.vertical,8)
        }.font(.system(size:12,weight:.medium))
        }
    }
}
