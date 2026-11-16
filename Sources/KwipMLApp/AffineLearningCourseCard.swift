import SwiftUI
import KwipMLCore

struct AffineLearningCourseCard: View {
    @ObservedObject var lesson: AffineLearningStore
    @ObservedObject var workspace: WorkspaceStore
    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            card(identity: .composition, title: "Compose affine maps", description: "Build a graph and enter its weights and biases. Compare a chain of transformations with one equivalent map, then test your construction on held-out inputs.")
            Divider()
            card(identity: .recovery(taskSeed: 0), title: "Recover an affine map", description: "Infer an affine transformation from public zero-input and basis-input observations. Predict, inspect, and construct a map that also works on unseen inputs.")
            Divider()
            card(identity: .backwardChain, title: "Build the backward map", description: "Construct the input gradient of two affine layers. Reverse the path, reason about transposes and bias, and test your map on unseen upstream gradients.")
            Divider()
            card(identity: .frozenStandardization, title: "Carry fitted statistics into inference", description: "Turn saved feature means and scales into an affine graph. Apply the same transformation to unseen rows and explain why new batch companions must not change an observation's output.")
            Divider()
            card(identity: .relu, title: "Transform values with ReLU", description: "Replace negative numbers with zero and keep positive numbers unchanged. Preserve each row and feature, compare measured values, and see why repeating this block gives the same result.")
            Divider()
            card(identity: .tanh, title: "Transform values with Tanh", description: "Preserve the sign of each number while keeping outputs between minus one and one. Compare measured values, preserve feature order, and see why large inputs change their outputs very little.")
            Divider()
            card(identity: .sigmoid, title: "Transform values with Sigmoid", description: "Build a smooth transformation of each number into a value between zero and one. Preserve feature order, inspect measured outputs, and compare new rows. These values do not automatically tell you how likely an outcome is.")
            Divider()
            card(identity: .geluExact, title: "Construct exact GELU", description: "Build a Gaussian-gated activation. Predict negative and near-zero outputs, then check your graph against the exact formula.")
            Divider()
            card(identity: .geluTanh, title: "Explore approximate GELU", description: "Configure the tanh approximation and test where it differs from exact GELU. Practice separating an approximation from a different activation.")
        }
    }
    private func card(identity: AffineLessonIdentity, title: String, description: String) -> some View {
        HStack(alignment: .top, spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                SectionLabel(title: (identity.geluApproximateTanh != nil || [.sigmoid, .tanh, .relu].contains(identity)) ? "Activations · measured construction" : "Linear algebra · measured construction")
                Text(title).font(.title3.weight(.semibold))
                Text(description).font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                Text("Guided exploration · independent practice · assessed outputs")
                    .font(.system(size: 11)).foregroundStyle(MLTheme.teal)
                if identity.assessmentID == AffineLessonIdentity.recovery(taskSeed: 0).assessmentID { Text("Each new project gets its own fixed observations.").font(.caption).foregroundStyle(MLTheme.muted) }
            }
            Spacer()
            if lesson.isActive && lesson.lessonTask?.identity.assessmentID == identity.assessmentID {
                Button("Resume lesson") { workspace.activeSection = "Learn" }
                    .accessibilityLabel("Resume \(title)")
            } else {
                Button("Start lesson") {
                    let selectedIdentity: AffineLessonIdentity = identity.assessmentID == AffineLessonIdentity.recovery(taskSeed: 0).assessmentID ? .recovery(taskSeed: UInt64.random(in: UInt64.min...UInt64.max)) : identity
                    lesson.start(identity: selectedIdentity, mode: .guided)
                    if lesson.isActive { workspace.activeSection = "Learn" }
                }
                .disabled(workspace.isBusy || lesson.isActive)
                .accessibilityLabel("Start \(title)")
                .help(lesson.isActive ? "Leave the active lesson before starting another project." : "Start a new managed lesson project")
            }
        }
    }
}
