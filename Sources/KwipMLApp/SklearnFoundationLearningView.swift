import SwiftUI
import KwipMLCore

struct SklearnFoundationLearningView: View {
    @ObservedObject var store: WorkspaceStore
    private let lesson = try? SklearnFoundationLesson.bundled()
    @Environment(\.lessonRailPresented) private var inRail
    var body:some View {
        if inRail {Button("Build logistic regression example") {_ = store.loadBackendWorkflowExample(backendID:"sklearn")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)} else {referenceBody}
    }
    private var referenceBody:some View {
        VStack(alignment: .leading, spacing: 12) {
            if let lesson {
                HStack(alignment: .top, spacing: 20) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("scikit-learn · A linear boundary").font(.title3.weight(.semibold))
                        Text("Use two measurements to predict one of two answer categories. Build, run and inspect the example without writing code.")
                            .font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                    }
                    Spacer(minLength: 12)
                    Button("Build logistic regression example") { _ = store.loadBackendWorkflowExample(backendID: "sklearn") }
                        .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                }
                InspectorSection("Before you start", symbol: "info.circle", identifier: "learn.sklearn.start") {
                    LessonRichText(text: """
                    **Logistic regression** is a classifier: it chooses between answer categories. It combines the input measurements using learned weights (how strongly each input counts) and an optional offset, then turns that score into a probability. A probability is a share between 0 and 1; it expresses the model's estimate, not a guarantee.

                    **Features** are input columns. The **target** is the answer column. This example uses signal and context as features and label as the target. **Validation** means checking rows kept out of training.
                    """)
                }
                InspectorSection("Build, run and inspect", symbol: "list.number", identifier: "learn.sklearn.steps") {
                    LessonRichText(text: """
                    1. Choose **Build logistic regression example**. Follow Table Input, Columns, Partition and Preprocess. Select Partition to see validation/test percentages and its seed. A seed is a number used to repeat a random split with the same data and settings.

                    2. Follow Partition’s three outputs into one Preprocess. Standard scaling learns the training average and spread, then uses those same rules on training, validation and test rows. Checking rows must not help fit these rules.

                    3. Select **Logistic regression**. In Configuration, check the First feature column is signal, the Second feature column is context, and the Target column is label. If you edit these fields, choose **Apply column mapping** before changing other settings. Keep this order when using the saved model later.

                    4. Choose **Run Flow**. Select a data component, then open **Results → Output** to see its captured data. For a specific port, use **Results → Ports & connections → Inspect**. Open History for the completed run. **Accuracy** is the fraction of correct answers; **log loss** also measures how confidently the model was wrong. Compare both, rather than treating one good score as proof.

                    5. Return to the Logistic regression component and find **Inverse regularization C**. Regularization discourages overly large learned weights; a smaller C strengthens that restraint. Change only C, run again and compare validation results with the same split. Watch any convergence warning: it means the solver did not meet its stopping goal within its allowed work.
                    """)
                }
                InspectorSection("Try a controlled comparison", symbol: "arrow.left.arrow.right", identifier: "learn.sklearn.practice") {
                    LessonRichText(text: """
                    Keep the data and split unchanged. Compare C = 1 with C = 0.1. Record how validation accuracy and log loss change; a smaller C need not give a better result.

                    This solver fits all training rows together. It has no batch-size or model-initialization seed setting. To change the random partition, edit the seed on Partition. Changing that seed is a different experiment from changing C.

                    This is practice with a runnable example. It does not award an automatically scored mastery receipt.
                    """)
                }
                InspectorSection("Saved results and current support", symbol: "externaldrive", identifier: "learn.sklearn.support") {
                    LessonRichText(text: """
                    A saved result contains learned coefficients (weights) and the optional offset in state.json, with portable prediction code. Prediction expects already-prepared feature values in the same order; loading the saved state does not automatically apply the scaler.

                    This foundation supports two answer categories using a dense table (ordinary values represented in every feature position) on CPU. It does not support GPU, partial fitting, warm restart or shared automatic gradients with a neural model. Loading a result for prediction is not resuming its training. Other scikit-learn model families remain outside this foundation.
                    """)
                    Text(lesson.assessment).font(.caption).foregroundStyle(MLTheme.muted)
                }
                InspectorSection("Optional technical reference", symbol: "function", identifier: "learn.sklearn.technical", initiallyExpanded: false) {
                    LessonRichText(text: lesson.objective)
                    Text(lesson.formula).font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                    ForEach(Array(lesson.steps.enumerated()), id: \.offset) { _, step in LessonRichText(text: step) }
                    LessonRichText(text: lesson.exercise)
                    LessonRichText(text: lesson.limits).foregroundStyle(MLTheme.muted)
                }
                Text("Bundled source v\(lesson.contentVersion) · Reviewed \(lesson.reviewedAt)")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            } else { Text("scikit-learn starter learning material is unavailable in this build.").foregroundStyle(MLTheme.muted) }
        }
    }
}
