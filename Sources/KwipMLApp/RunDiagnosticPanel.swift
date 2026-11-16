import SwiftUI

/// Read-only explanations share the registry and evaluated report with export consumers.
struct RunDiagnosticPanel: View {
    let metrics: [EpochMetric]
    let error: String?
    let sampling: DiagnosticSampling
    private var report: Result<DiagnosticReport, Error> {
        Result { try RunDiagnostics.report(metrics, error: error, context: DiagnosticInputContext(sampling: sampling)) }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Current rule evaluation").font(.caption.bold())
            Text("Evaluated from the observations currently loaded. Original saved diagnostics are available in History when recorded.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            DiagnosticReportPanel(result: report)
        }
    }
}

/// Shared rendering for a current evaluation or a verified immutable historical report.
struct DiagnosticReportPanel: View {
    let result: Result<DiagnosticReport, Error>
    var hiddenLegacyRuleIDs: Set<String> = []
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Run diagnostics")
            switch result {
            case .failure(let error):
                Text("Diagnostic evaluation unavailable: \(error.localizedDescription)").font(.caption).foregroundStyle(MLTheme.rose)
                Text("No diagnostic conclusion is shown when report verification fails.").font(.caption)
            case .success(let report):
                ForEach(report.findings.filter { !hiddenLegacyRuleIDs.contains($0.id) }) { finding in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(finding.title).font(.headline)
                        DiagnosticExplanationView(guide: .make(ruleID: finding.id, observedFact: finding.evidence))
                        ResultsSection(title:"Original captured guidance",symbol:"doc.text") {
                            Text(finding.suggestion).font(.caption).textSelection(.enabled)
                            Text("Preserved with this report. Earlier reports may use previous navigation names; use the current canvas actions above.").font(.caption2).foregroundStyle(MLTheme.muted)
                        }
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                }
                if report.findings.isEmpty {
                    Text("No findings from the available evidence. This does not establish model health.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                if report.findings.contains(where: { hiddenLegacyRuleIDs.contains($0.id) }) {
                    Text("The matching legacy warning is grouped with the measured divergence check. Its original evaluation remains in rule coverage.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                ResultsSection(title:"Evidence & rule coverage · \(report.evaluations.count) checks",symbol:"checklist") {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("Ruleset version \(report.registryVersion) · \(report.observedInputCount) supplied observations · \(samplingLabel(report.context.sampling))")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                        ForEach(report.evaluations, id: \.ruleID) { evaluation in
                            if let rule = DiagnosticRuleRegistry.builtIn.rules.first(where: { $0.id == evaluation.ruleID }) {
                                ResultsSection(title:rule.title,symbol:"checkmark.shield") {
                                    diagnosticField("Status",explanation(evaluation.reason))
                                    diagnosticField("Requires",rule.prerequisites.joined(separator:"; "))
                                    diagnosticField("Limits",rule.confounders.joined(separator:"; "))
                                    diagnosticField("Missing evidence",rule.missingEvidence)
                                }
                            }
                        }
                        ForEach(report.limitations, id: \.self) { Text($0).font(.caption).foregroundStyle(MLTheme.muted) }
                    }.padding(.top, 8)
                }
            }
        }
    }
    private func diagnosticField(_ title:String,_ value:String)->some View {
        HStack(alignment:.top,spacing:16) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(MLTheme.muted).frame(width:120,alignment:.leading)
            Text(value).font(.caption).frame(maxWidth:.infinity,alignment:.leading).textSelection(.enabled)
        }.padding(.vertical,4)
    }
    private func samplingLabel(_ sampling: DiagnosticSampling) -> String {
        switch sampling {
        case .sampled: return "sampled observations"
        case .complete: return "complete observations (declared by producer)"
        case .unspecified: return "sampling completeness unspecified"
        }
    }
    private func explanation(_ reason: DiagnosticEvaluationReason) -> String {
        switch reason {
        case .matched: return "Finding reported from the supplied evidence."
        case .thresholdNotMet: return "Checked; the rule did not trigger."
        case .noObservations: return "Not evaluated: no metric observations yet."
        case .missingError: return "Not evaluated: no runtime error was supplied."
        case .earlierErrorClassifier: return "Withheld: an earlier error category matched this message."
        case .insufficientObservations: return "Withheld: at least 10 distinct positive-epoch observations are required."
        case .nonfiniteMeasurements: return "Withheld: the observations include a non-finite value."
        case .invalidMetricDomain: return "Withheld: loss or accuracy lies outside the expected domain."
        case .nonfiniteDerivedSignals: return "Withheld: derived statistics could not be represented as finite values."
        }
    }
}
