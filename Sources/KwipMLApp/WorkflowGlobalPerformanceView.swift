import SwiftUI
import KwipMLCore

struct WorkflowGlobalPerformanceView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                Label(host.performanceCaptureEnabled ? (host.generationIsLatest ? "Capture enabled for the next training run":"Capture enabled for the next run"):"Performance capture disabled",systemImage:host.performanceCaptureEnabled ? "record.circle":"speedometer")
                    .font(.system(size:12,weight:.semibold))
                Spacer()
                Button(host.performanceCaptureEnabled ? "Disable capture":"Enable capture") {_ = host.setPerformanceCaptureEnabled(!host.performanceCaptureEnabled)}
                    .buttonStyle(FlatButtonStyle()).disabled(host.canvas.isBusy)
            }
            if host.generationIsLatest {
                Text("Generation does not capture a Validation performance profile.").font(.caption).foregroundStyle(MLTheme.muted)
            } else if let execution=host.currentExecution,execution.phase == .completed,let capture=execution.performanceCapture {
                WorkflowPerformanceView(capture:capture,graph:execution.request.plan.model.graph)
            } else if let execution=host.currentExecution,[.failed,.cancelled].contains(execution.phase) {
                Text("This run did not complete; no completed performance profile is available.").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
                Text(host.currentExecution?.request.sampleCapture?.performanceMeasurements == true ? "Measurements will appear after this run completes and its artifacts are verified.":"No performance measurements were captured for the current run.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
            DisclosureGroup("Information") {
                Text("Capture measures one representative final Validation forward, up to 8 rows and 128 operations. It records host wall time (synchronized on MPS) and process resident memory. It does not measure whole-run throughput, GPU kernel time, peak memory or per-layer allocation. Enabling here also enables diagnostic/performance retention, independently of component snapshots. Capture changes apply to future runs; disabling it does not erase saved measurements.")
                    .font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
            }.font(.caption)
        }
    }
}

struct WorkflowComponentMetricsView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    let address:WorkflowNodeAddress
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if let backend=host.currentBackendFit,backend.plan.estimator==address,backend.plan.configuration.hasTableOutputs {
                if backend.phase == .completed,!host.predictionOutputIsCurrent(.init(address:address,port:"predictions")) {
                    Text("Metrics belong to an earlier configuration. Rerun this component or inspect its recorded metrics in History.").font(.caption).foregroundStyle(MLTheme.muted)
                } else {BackendFitResultsView(session:backend)}
            } else if host.owner.document.nodes.contains(where:{$0.address==address && FineTuningConfiguration.isExternal($0)}) {
                ModelFineTuningMetricsView(store:host.modelLoader,address:address)
            } else if let record=host.modelLoader.currentEvaluation,record.address==address,host.modelLoader.evaluationIsCurrent(record),let summary=host.modelLoader.evaluationSummary {
                ModelCausalEvaluationSummaryView(summary:summary)
            } else if let evaluation=host.evaluation,evaluation.request.plan.evaluate==address,let result=evaluation.verifiedResult,evaluation.phase == .completed,evaluation.predictionTableArtifact != nil {
                metrics([("Population",evaluation.request.plan.settings.partitionTitle),("Rows",String(result.summary.evaluatedCount))] + (result.score.map {[(result.objectiveEvaluation == nil ? "Mean loss":"Objective",number($0))]} ?? []) + (result.accuracy.map {[("Accuracy",percent($0))]} ?? []))
            } else if let execution=host.currentExecution,execution.request.plan.train==address {
                if execution.phase == .completed,!host.predictionOutputIsCurrent(.init(address:address,port:"predictions")) {
                    Text("Metrics belong to an earlier configuration. Rerun this component or inspect its saved run in History.").font(.caption).foregroundStyle(MLTheme.muted)
                } else {
                WorkflowLiveExecutionSummary(execution:execution,includesDiagnostics:false,compact:true,includesMetricHeader:false)
                }
            } else if host.owner.document.nodes.contains(where:{node in
                guard node.address==address else {return false};if case .train=node.payload {return true};return false
            }) {TrainingMetricsChart(metrics:[])}
            else {Text("No captured metrics for this component.").font(.caption).foregroundStyle(MLTheme.muted)}
        }
    }
    private func percent(_ value:Double)->String {String(format:"%.1f%%",value*100)}
    private func number(_ value:Double)->String {String(format:"%.6g",value)}
    private func metrics(_ values:[(String,String)])->some View {
        ThemedScrollView(.horizontal) {
            HStack(spacing:0) {ForEach(values.indices,id:\.self) {index in
                VStack(alignment:.leading,spacing:5) {
                    Text(values[index].0).font(.system(size:10,weight:.semibold)).foregroundStyle(MLTheme.muted)
                    Text(values[index].1).font(.system(size:15,weight:.semibold,design:.monospaced)).textSelection(.enabled)
                }.padding(12).frame(minWidth:120,alignment:.leading).overlay(Rectangle().stroke(MLTheme.rule))
            }}}
    }
}
