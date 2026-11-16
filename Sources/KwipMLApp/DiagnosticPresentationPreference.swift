import Foundation

/// Only current, context-qualified findings group duplicate live cards. Stored
/// reports remain unchanged and ordinary archive rendering has no exclusions.
enum DiagnosticPresentationPreference {
    static func hiddenLegacyRuleIDs(measured: MeasuredDiagnosticReport?) -> Set<String> {
        guard let measured, measured.registryVersion == 2 else { return [] }
        do { try measured.validate() } catch { return [] }
        return measured.findings.contains { $0.ruleID == "generalization.divergence.v1" }
            ? ["trend.generalization-gap.v1"] : []
    }
}
