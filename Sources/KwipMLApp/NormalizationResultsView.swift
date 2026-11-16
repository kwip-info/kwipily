import SwiftUI
import KwipMLCore

struct NormalizationResultsView: View {
    @ObservedObject var lesson: NormalizationLearningStore
    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 28) {
                summary.frame(width: 340, alignment: .leading)
                measurements.frame(minWidth: 400, maxWidth: .infinity, alignment: .leading)
            }
            VStack(alignment: .leading, spacing: 20) { summary; Divider(); measurements }
        }
    }
    private var measurements: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let projection = lesson.formativeProjection {
                NormalizationFormativeResultsView(projection: projection).id(projection.submissionDigest)
            } else {
                Text("Guided and Practice checks reveal measured values for the current construction. Assessment fixture values stay private, including after switching modes.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private var summary: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionLabel(title: "Latest measured result")
            if let grade = lesson.latestGrade {
                Text(lesson.latestAttempt == nil ? "No verified receipt for this result" : (lesson.latestResultIsCurrent ? "Matches current construction" : "Previous construction · check again after edits"))
                    .font(.caption).foregroundStyle(lesson.latestAttempt != nil && lesson.latestResultIsCurrent ? MLTheme.teal : MLTheme.rose)
                if let attempt = lesson.latestAttempt {
                    Text("\(attempt.mode.rawValue.capitalized) · \(attempt.timestamp.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    Text("Content \(version(attempt.contentVersion)) · assessment \(version(attempt.assessmentVersion))")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                    if !lesson.latestAssessmentIsCurrent {
                        Text("Historical contract. This receipt is verified against its recorded version; it does not establish completion of the current course version.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }
                }
                Text(grade.status.rawValue).font(.headline)
                LessonRichText(text: grade.summary)
                ForEach(Array(grade.outcomes.enumerated()), id: \.offset) { _, outcome in
                    VStack(alignment: .leading, spacing: 5) {
                        Label(outcome.criterionID, systemImage: outcome.passed ? "checkmark.circle" : "exclamationmark.circle").font(.caption.bold())
                        LessonRichText(text: outcome.evidence)
                        if !outcome.passed { LessonRichText(text: outcome.nextStep).foregroundStyle(MLTheme.muted) }
                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
                        .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                }
                Text(lesson.resultSaved ? "Verified attempt saved" : "No saved completion from this result")
                    .font(.caption).foregroundStyle(lesson.resultSaved ? MLTheme.teal : MLTheme.rose)
            } else { Text("Build your flow and check its measured behavior.").font(.caption).foregroundStyle(MLTheme.muted) }
            Divider()
            Text(lesson.hasCurrentCompletion ? "A passing assessment is saved for the current course version." : "No verified passing assessment found for the current course version.").font(.caption)
            Text("A recorded pass describes that construction and contract. It does not certify subsequent edits or general mastery.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            SectionLabel(title: "Recent saved attempts")
            ForEach(Array(lesson.attempts.filter { $0.objectiveID == NormalizationLessonTask.objectiveID }.suffix(12).reversed())) { attempt in
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(attempt.mode.rawValue.capitalized) · \(attempt.grade.status.rawValue)").font(.caption.bold())
                    Text(attempt.timestamp.formatted(date: .abbreviated, time: .shortened)).font(.caption).foregroundStyle(MLTheme.muted)
                    Text(attempt.grade.summary).font(.caption).lineLimit(3)
                }
            }
        }
    }
    private func version(_ value: ContractVersion) -> String { "\(value.major).\(value.minor).\(value.patch)" }
}

/// Only accepts the verified formative projection; private plans never enter this view.
struct NormalizationFormativeResultsView: View {
    let projection: NormalizationFormativeProjection
    @State private var selectedFixtureID = ""
    @State private var selectedOutput = "mean"
    private var fixture: NormalizationFormativeFixture? { projection.fixtures.first { $0.id == selectedFixtureID } ?? projection.fixtures.first }
    private func title(_ key: String) -> String {
        switch key {
        case "mean": return "Fitted column means"
        case "scale": return "Fitted column scales"
        case "training": return "Transformed training"
        case "validation": return "Transformed held-out data"
        default: return key
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Inspect measured values")
            Text("Rows are observations; columns are x and y. These are measured fixture results, not the public worked example.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title: "Measured normalization fixture", selection: fixture?.id ?? "", options: projection.fixtures.enumerated().map {
                FlatChoiceOption(id: $0.element.id, title: "Fixture \($0.offset + 1)")
            }) { selectedFixtureID = $0 }
            if let fixture {
                LearningTensorTable(tensor: fixture.training, title: "Input training")
                LearningTensorTable(tensor: fixture.validation, title: "Input held out")
                let keys = fixture.measured.keys.sorted()
                let key = keys.contains(selectedOutput) ? selectedOutput : (keys.first ?? "")
                FlatChoiceControl(title: "Measured normalization output", selection: key, options: keys.map { FlatChoiceOption(id: $0, title: title($0)) }) { selectedOutput = $0 }
                if let measured = fixture.measured[key], let expected = fixture.expected[key] {
                    LearningTensorTable(tensor: measured, title: title(key) + " · measured").id("measured-" + key + fixture.id)
                    LearningTensorTable(tensor: expected, title: title(key) + " · expected").id("expected-" + key + fixture.id)
                    let differences = zip(measured.values, expected.values).map { $0 - $1 }
                    if measured.shape == expected.shape, differences.allSatisfy(\.isFinite) {
                        LearningTensorTable(tensor: LearningTensor(shape: measured.shape, values: differences), title: "Measured − expected", signed: true).id("difference-" + key + fixture.id)
                    }
                }
                Text("Values are rounded for display; hover for full precision. Signed differences show the direction of error. The criterion evidence above uses the assessment's numerical tolerances.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
            }
        }
    }
}
