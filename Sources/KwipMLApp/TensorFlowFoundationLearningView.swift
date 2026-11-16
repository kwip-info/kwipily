import SwiftUI
import KwipMLCore

struct TensorFlowFoundationLearningView: View {
    @ObservedObject var store: WorkspaceStore
    private let lesson = try? TensorFlowFoundationLesson.bundled()
    @Environment(\.lessonRailPresented) private var inRail
    var body:some View {
        if inRail {Button("Build TensorFlow example") {_ = store.loadBackendWorkflowExample(backendID:"tensorflow")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)} else {referenceBody}
    }
    private var referenceBody:some View {
        VStack(alignment: .leading, spacing: 12) {
            if let lesson {
                HStack(alignment: .top, spacing: 20) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("TensorFlow · One classifier, two learned layers").font(.title3.weight(.semibold))
                        Text("Follow a small model from prepared rows to class scores, then inspect its training results and saved weights. No code is required for this example.")
                            .font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                    }
                    Spacer(minLength: 12)
                    Button("Build TensorFlow example") { store.loadBackendWorkflowExample(backendID: "tensorflow") }
                        .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                }
                InspectorSection("Before you start", symbol: "info.circle", identifier: "learn.tensorflow.start") {
                    LessonRichText(text: """
                    **TensorFlow** runs the model; **Keras** is the model-building interface it uses. A **Dense** layer combines every input feature using learned weights and an optional offset. On this canvas, those layers appear as **Linear** components.

                    **Features** are the input measurements; the **target** is the answer category. A **tensor** is an array of numbers, such as a table. This example receives two features and produces two raw class scores, also called **logits**. Scores are not probabilities: probabilities are shares between 0 and 1.
                    """)
                }
                InspectorSection("Build, run and inspect", symbol: "list.number", identifier: "learn.tensorflow.steps") {
                    LessonRichText(text: """
                    1. Choose **Build TensorFlow example**. Follow Table Input, Columns, Partition and Preprocess. Training rows teach the model; validation rows check it; test rows stay reserved. Preprocess learns the average and spread from training only, then applies them unchanged to every population.

                    2. Follow the model path: **Linear → Activation (ReLU) → Linear → Model definition**. Select Model definition to confirm two input features. The first Linear creates eight learned features. **ReLU** keeps positive values and replaces negative values with zero. The final Linear produces two class scores per row.

                    3. Select the **Keras classifier** component. In Configuration, keep signal then context as feature columns and label as the target. Choose **Apply column mapping** if you edit these fields. Follow its model connection back to Model definition, which identifies the connected architecture rather than carrying data rows.

                    4. Review its training settings. An **epoch** is one pass through the training rows. A **batch** is a group of rows used for one update. The **learning rate** controls the size of weight adjustments. A **seed** helps repeat the starting random choices; Partition has its own seed for choosing rows.

                    5. Choose **Run Flow**. Clear the canvas selection to watch **Results → Activity**. Select a data component and open **Results → Output** to inspect its captured table. In History, compare **loss** (how wrong predictions are under the training rule) and **accuracy** (the fraction of correct answers). Validation results matter because those rows did not teach the model. A better training score alone does not show that new data will work better.
                    """)
                }
                InspectorSection("Try it and explain the result", symbol: "arrow.left.arrow.right", identifier: "learn.tensorflow.practice") {
                    LessonRichText(text: """
                    Keep the data and split fixed. Change only the classifier's seed or learning rate, run again, and compare the two records in History. Predict what may change before running. Repeatedly choosing settings to improve validation can also make the model too specific to those checking rows.

                    Here is a small arithmetic check: the first layer has 2 × 8 weights plus 8 offsets, or 24 learned values. The second has 8 × 2 weights plus 2 offsets, or 18. Together they learn 42 values. If a batch has 16 rows, the first layer produces 16 rows of 8 values.

                    This is reading, visual practice and manual checking. There is no automatically scored assessment receipt.
                    """)
                }
                InspectorSection("Saved weights and current support", symbol: "externaldrive", identifier: "learn.tensorflow.support") {
                    LessonRichText(text: """
                    The result includes portable prediction code, the model graph and Keras HDF5 weights. A **checkpoint** is saved model state; restoring weights for prediction is different from continuing the exact training process, which also needs the optimizer's update state.

                    Prediction expects already-prepared float32 rows with two features in the same order. float32 is a decimal-number storage type. The run separately retains preparation and source records; the prediction entry point does not silently prepare new raw rows.

                    This foundation supports two input features and two answer categories on CPU. Its reviewed components are configured model input, Linear, ReLU, Tanh, Add, Concatenate and the model boundary. Tanh squeezes values between -1 and 1; Add sums matching values; Concatenate joins feature sets. GPU, optimizer-state resume, arbitrary layers and custom code are unsupported. This export restores weights for prediction; it is not a resumable training checkpoint.
                    """)
                }
                InspectorSection("Optional technical reference", symbol: "function", identifier: "learn.tensorflow.technical", initiallyExpanded: false) {
                    ForEach(Array(lesson.steps.enumerated()), id: \.offset) { _, step in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(step.title).font(.system(size: 12, weight: .semibold))
                            LessonRichText(text: step.body)
                            if let formula = step.formula { Text(formula).font(.system(size: 12, design: .monospaced)).textSelection(.enabled) }
                        }
                    }
                    LessonRichText(text: lesson.limitations).foregroundStyle(MLTheme.muted)
                }
                InspectorSection("Optional manual questions and answers", symbol: "questionmark.circle", identifier: "learn.tensorflow.questions", initiallyExpanded: false) {
                    ForEach(Array(lesson.checks.enumerated()), id: \.offset) { index, check in
                        InspectorSection(check.question, symbol: "checkmark.circle", identifier: "learn.tensorflow.question.\(index)", initiallyExpanded: false) {
                            LessonRichText(text: check.expected)
                        }
                    }
                }
                Text("Bundled source v\(lesson.contentVersion) · Reviewed \(lesson.reviewedAt)")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            } else { Text("TensorFlow starter learning material is unavailable in this build.").foregroundStyle(MLTheme.muted) }
        }
    }
}
