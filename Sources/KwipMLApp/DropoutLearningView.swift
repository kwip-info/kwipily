import SwiftUI
import KwipMLCore

/// Presentation and original task launch only; assessment mathematics and identities stay unchanged.
enum DropoutLearningContent {
    static let lessonID="dropout"
    static let componentID="dropout"
    static let tasks:[RemainingActivationLearningTask]=[.dropoutTraining,.dropoutEvaluation]
}

struct DropoutLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    @ObservedObject var tensorLearning:TensorLearningStore
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail {
            Text("Dropout · training and evaluation").font(.title3.weight(.semibold))
            Text("Temporarily remove individual measurements while training. In Configuration, choose Drop probability; the run or lesson owns its seed and mode.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            LessonTextPanel(title:"Training",text:"Probability is a drop chance, not a measured percentage. For a probability below 1, retained values multiply by 1 ÷ (1 − probability). Probability 0 keeps every value; probability 1 makes every training output zero.")
            LessonTextPanel(title:"Evaluation",text:"Every value passes through unchanged, with no new random mask. A mask records the positions kept or removed. This elementwise operation adds no learned weights and preserves tensor shape.")
            LessonTextPanel(title:"Repeatability",text:"New training calls normally draw new masks. The same seed, runtime, device and call order can repeat a pattern. CPU and Apple GPU masks need not match; changing call order can change later masks.")
            }
            Text("Measured construction lessons").font(.system(size:13,weight:.semibold))
            ForEach(DropoutLearningContent.tasks,id:\.rawValue) {task in
                Button(task.title) {
                    ApplicationNavigationWindows.shared.openLessonOnCanvas(entryID:"remaining-activations",operation:task.rawValue)
                }.disabled(store.isBusy)
                    .accessibilityLabel("Start "+task.title)
                    .accessibilityIdentifier("dropout.lesson."+task.rawValue)
            }
            Text("The separate Construction assessments measure values and input influence automatically. Opening this practice or its example does not award their completion.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            Button("Open Dropout classifier example") {store.loadComponentWorkflowExample(DropoutLearningContent.componentID)}
                .disabled(store.isBusy)
                .accessibilityIdentifier("dropout.example")
        }
    }
}
