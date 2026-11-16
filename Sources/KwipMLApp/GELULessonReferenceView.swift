import SwiftUI
import KwipMLCore

struct GELULessonReferenceView: View {
    let approximateTanh: Bool
    private var reference: GELULessonReference { LearningExamples.geluReference(approximateTanh: approximateTanh) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(approximateTanh ? "Use the approximate GELU calculation" : "Use the exact GELU calculation").font(.headline)
            LessonRichText(text: "Set every **GELU** block’s **Tanh approximation** switch **\(approximateTanh ? "on":"off")**. Include at least one GELU. Each value is transformed separately; do not mix the two feature columns or change the number of observations.")
            LessonRichText(text: "This lesson’s runner also requires a **Linear** block. Enter its weights and bias (fixed offsets) in **Parameters**. Those surrounding blocks must preserve the requested GELU behavior; the runner will not learn their numbers for you. Different valid arrangements are accepted.")
            LessonRichText(text: "Predict the result for zero, a small negative value and a large positive value. Write your reasoning, connect the blocks, then use **Check construction** in Guided or Practice to inspect measured values. Assessment tests additional private examples.")
            InspectorSection("Optional · how small changes affect learning", symbol: "chart.xyaxis.line", identifier: "lesson.gelu.sensitivity", initiallyExpanded: false) {
                LessonRichText(text: "A **gradient** describes how a very small input change affects the output. Around zero, GELU passes about half of that change. A large positive input passes almost all of it; a large negative input changes its output very little. Both calculation modes have these behaviors.")
                LessonRichText(text: "For some negative inputs, increasing the input can briefly make the output more negative. GELU is a smooth gate, not simply a smaller ReLU. These small-change relationships help training decide how to adjust the earlier layer’s weights. GELU has no weights of its own, and applying it does not guarantee that a model will learn well.")
                LessonRichText(text: "Try nearby input values in Guided or Practice and compare their measured outputs. This is an explanation of sensitivity, not an additional gradient-construction assessment.")
            }
            InspectorSection("Optional · the GELU equation", symbol: "function", identifier: "lesson.gelu.formula", initiallyExpanded: false) {
                LessonRichText(text: "x is one input value. GELU(x) is its output. √ means square root; π is pi, approximately 3.14; x³ means x multiplied by itself three times. erf is the error function, which supplies a smooth Gaussian probability gate. tanh is a smooth curve between −1 and 1 used by the approximate calculation. ≈ means approximately equal. The gate multiplies x; it is not a standalone Tanh transformation.")
                Text(reference.formula).font(.system(size: 12, design: .monospaced)).foregroundStyle(MLTheme.teal).textSelection(.enabled)
                LessonRichText(text: "GELU has no trainable numbers of its own. In shape notation, [batch, 2] is an array with a group of observations in rows and two features in columns; input and output keep that shape.")
            }
        }
    }
}
