import SwiftUI
import KwipMLCore

/// Public observations only. No recovered parameters or held-out assessment fixtures enter this view.
struct AffineRecoveryReferenceView: View {
    @ObservedObject var lesson: AffineLearningStore
    let task: AffineLessonTask

    private var nextProbeID: String? {
        task.probes.first { !(lesson.draft?.hintIDs.contains("probe-\($0.id)") ?? false) }?.id
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if lesson.mode == .assessment {
                LessonTextPanel(title:AffineRecoveryLessonCopy.title("reference.probes"),text:AffineRecoveryLessonCopy.text("reference.probes"))
            } else {
            LessonRichText(text: "Use the public examples below to discover how each input feature contributes to the output. In Guided mode, predict before revealing each example. Build your proposed multiply-and-add calculation with the allowed blocks and enter its weights and fixed offsets in Parameters.")
            }
            Text("Public examples · two input features, two output features")
                .font(.system(size: 10)).foregroundStyle(MLTheme.teal)
            let originatingProjectID = lesson.draft?.projectID
            ForEach(task.probes, id: \.id) { probe in
                let revealed = lesson.mode != .guided || lesson.draft?.hintIDs.contains("probe-\(probe.id)") == true
                if revealed || probe.id == nextProbeID {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(title(probe.id)).font(.system(size: 11, weight: .semibold))
                        HStack(alignment: .top, spacing: 12) {
                            vector(probe.input, title: "Input values", observation: probe.id)
                            if revealed { vector(probe.output, title: "Observed values", observation: probe.id) }
                        }
                        if !revealed {
                            Text(prompt(probe.id)).font(.caption).foregroundStyle(MLTheme.muted)
                            Text("Write your prediction in Prediction & reasoning below, then reveal the observation.")
                                .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                            Button("Reveal observed output") {
                                guard lesson.draft?.projectID == originatingProjectID, lesson.lessonTask?.identity == task.identity else { return }
                                lesson.revealHint("probe-\(probe.id)")
                            }
                                .buttonStyle(FlatButtonStyle()).disabled(lesson.isBusy)
                                .accessibilityLabel("Reveal \(title(probe.id)) observed output")
                        }
                    }.padding(.vertical, 3)
                }
            }
            if lesson.mode != .assessment {
                Text("These examples describe the task. Assessment checks additional private inputs.")
                    .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func title(_ id: String) -> String {
        switch id {
        case "zero": return "Zero input"
        case "basis-0": return "Only feature 1 switched on"
        case "basis-1": return "Only feature 2 switched on"
        default: return "Public observation"
        }
    }
    private func prompt(_ id: String) -> String {
        id == "zero" ? "An affine map multiplies inputs by fixed weights and adds a fixed offset. Predict what can remain when both inputs are zero." : "Predict what changes from the all-zero example when only this input feature becomes one."
    }
    private func vector(_ values: [Double], title: String, observation: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 10, weight: .medium))
            VStack(spacing: 0) {
                ForEach(Array(values.enumerated()), id: \.offset) { row, value in
                    Text(value.formatted(.number.precision(.fractionLength(0...6))))
                        .font(.system(size: 12, design: .monospaced)).monospacedDigit()
                        .frame(minWidth: 62, minHeight: 26).padding(.horizontal, 4)
                        .background(MLTheme.background)
                        .accessibilityLabel("\(observation), \(title), row \(row), column 0, \(value)")
                }
            }.overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1)).textSelection(.enabled)
        }
    }
}
