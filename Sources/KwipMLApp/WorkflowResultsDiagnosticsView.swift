import SwiftUI
import KwipMLCore

/// Current loaded-observation rule evaluation; does not replace an archived report.
struct WorkflowResultsDiagnosticsView: View {
    @ObservedObject var host: WorkflowWorkspaceSession
    var onSelectFailure: ((WorkflowNodeAddress) -> Void)? = nil
    var body: some View {
        if host.isRetired {
            Text("This file session is no longer active.").font(.caption).foregroundStyle(MLTheme.muted)
        } else {
            if let repair = host.owner.retiredToolRepairMessage {
                Text(repair).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
            }
            ForEach(host.owner.document.nodes.filter {host.canvas.configurationIssue(for:$0.address) != nil},id:\.address) {node in
                VStack(alignment:.leading,spacing:5) {
                    Text(node.name.isEmpty ? "Component configuration":node.name).font(.caption.weight(.semibold))
                    Text(host.canvas.configurationIssue(for:node.address) ?? "").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                    if let onSelectFailure {Button("Show component") {onSelectFailure(node.address)}.buttonStyle(FlatButtonStyle())}
                }
            }
            ForEach(host.owner.document.nodes.filter {host.modelLoader.failure($0.address) != nil},id:\.address) {node in
                VStack(alignment:.leading,spacing:5) {
                    Text(node.name.isEmpty ? WorkflowCanvasAdapter.icon(node.payload):node.name).font(.caption.weight(.semibold))
                    Text(host.modelLoader.failure(node.address) ?? "").font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                    if let onSelectFailure {Button("Show component") {onSelectFailure(node.address)}.buttonStyle(FlatButtonStyle())}
                }
            }
            if let failure = host.attributedPreparationFailure {
                WorkflowPreparationFailureView(evidence: failure, onSelect: onSelectFailure)
            }
            if let generation=host.generation,!generation.isBusy,generation.artifact==nil {
                Text("Generate text · "+generation.statusMessage).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                if let onSelectFailure {Button("Show component") {onSelectFailure(generation.address)}.buttonStyle(FlatButtonStyle())}
            }
            if let evaluation=host.evaluation,evaluation.phase == .failed {
                Text("Evaluate · "+evaluation.statusMessage).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
            }
            if let preparation=host.predictionPreparation,preparation.phase == .failed {
                Text("Prediction preparation · "+preparation.statusMessage).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
            }
            if let execution = host.currentExecution {
                WorkflowCurrentFindingsTable(execution: execution,
                    error: host.outcomes.first(where: { $0.runID == execution.runID })?.error)
                    .id(execution.runID)
            } else if host.attributedPreparationFailure == nil {
                if let preparation = host.currentDataPreparation, preparation.phase == .failed {
                    Text(preparation.statusMessage).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)
                } else {
                    Text("No run diagnostic evidence is loaded for this session.").font(.caption).foregroundStyle(MLTheme.muted)
                }
            }
        }
    }
}

private struct WorkflowCurrentFindingsTable: View {
    @ObservedObject var execution: WorkflowExecutionSession
    let error: String?
    @State private var selectedID: String?
    private var result: Result<DiagnosticReport, Error> {
        Result { try RunDiagnostics.report(execution.liveMetrics, error: error,
            context: DiagnosticInputContext(runID: execution.runID.uuidString, sampling: .sampled)) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Findings · current loaded observations").font(.system(size: 12, weight: .semibold))
            Text("Evaluated from the bounded live snapshot and this run’s terminal error, when present. Rules may withhold findings when observations are missing or insufficient. This is not the original archived diagnostic report.")
                .font(.caption2).foregroundStyle(MLTheme.muted)
            switch result {
            case .failure(let failure):
                Text("Diagnostic evaluation unavailable: \(failure.localizedDescription)").font(.caption).foregroundStyle(MLTheme.rose)
            case .success(let report):
                if report.findings.isEmpty {
                    Text("No findings from the available evidence. This does not establish model health.").font(.caption).foregroundStyle(MLTheme.muted)
                } else {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 16) {
                            findings(report).frame(minWidth: 220, maxWidth: 280)
                            selectedFinding(report).frame(minWidth: 320, maxWidth: .infinity)
                        }
                        VStack(alignment: .leading, spacing: 12) {
                            findings(report)
                            selectedFinding(report)
                        }
                    }
                }
                ResultsSection(title:"Rule coverage, evidence and limits",symbol:"checklist") {
                    Text("Ruleset version \(report.registryVersion) · \(report.observedInputCount) supplied observations · \(report.context.sampling.rawValue) observations")
                        .font(.caption2).foregroundStyle(MLTheme.muted)
                    ForEach(report.evaluations,id: \.ruleID) { evaluation in
                        if let rule = DiagnosticRuleRegistry.builtIn.rules.first(where: {$0.id == evaluation.ruleID}) {
                            ResultsSection(title:rule.title,symbol:"checkmark.shield") {
                                detail("Evaluation status",evaluation.status.rawValue)
                                detail("Evidence outcome",evaluation.reason.rawValue)
                                detail("Requires",rule.prerequisites.joined(separator:"; "))
                                detail("Limits",rule.confounders.joined(separator:"; "))
                                detail("Missing evidence",rule.missingEvidence)
                            }
                        }
                    }
                    ForEach(report.limitations,id: \.self) { Text($0).font(.caption).foregroundStyle(MLTheme.muted) }
                    detail("Input SHA-256",report.inputSHA256)
                }
            }
            Text("For saved measurement provenance, traces, suppression annotations and guarded repair proposals, select and verify the run in History.")
                .font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    @ViewBuilder private func selectedFinding(_ report: DiagnosticReport) -> some View {
        if let finding = report.findings.first(where: {$0.id == selectedID}) ?? report.findings.first {
            let guide = DiagnosticAudienceGuide.make(ruleID:finding.id,observedFact:finding.evidence)
            VStack(alignment:.leading,spacing:12) {
                Text(finding.title).font(.system(size:13,weight:.semibold))
                Text("Run-wide finding · Severity not recorded").font(.caption2).foregroundStyle(MLTheme.muted)
                detail("Observed",finding.evidence)
                detail("Meaning",guide.meaning)
                detail("Try next",finding.suggestion)
                Text("A suggestion is a proposed investigation. No canvas change or run is applied automatically.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
                ResultsSection(title:"Finding evidence and limits",symbol:"doc.text") {
                    detail("Other explanations",guide.alternatives)
                    detail("Limits",guide.limits)
                    detail("Rule",finding.id)
                    Text("This report has no component attribution. Its evidence belongs to the run as a whole.")
                        .font(.caption2).foregroundStyle(MLTheme.muted)
                }
            }.frame(maxWidth:.infinity,alignment:.leading)
        }
    }
    private func findings(_ report: DiagnosticReport) -> some View {
        VStack(alignment:.leading,spacing:0) {
            Text("Findings · \(report.findings.count)").font(.system(size:10,weight:.semibold))
                .padding(8).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel)
            ForEach(report.findings) { finding in
                Button { selectedID = finding.id } label: {
                    VStack(alignment:.leading,spacing:5) {
                        Text(finding.title).font(.system(size:11,weight:.medium))
                        Text(finding.id).font(.system(size:10,design:.monospaced)).foregroundStyle(MLTheme.muted)
                    }.padding(10).frame(maxWidth:.infinity,alignment:.leading)
                        .background((selectedID ?? report.findings.first?.id) == finding.id ? MLTheme.rose.opacity(0.12) : MLTheme.background)
                }.buttonStyle(.plain)
                Rectangle().fill(MLTheme.rule).frame(height:1)
            }
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func detail(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased()).font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted)
            Text(text).font(.caption).textSelection(.enabled)
        }.frame(maxWidth: .infinity, alignment: .leading).padding(10)
            .overlay(Rectangle().stroke(MLTheme.rule))
    }
}
