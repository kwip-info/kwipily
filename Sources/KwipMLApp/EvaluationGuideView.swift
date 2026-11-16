import SwiftUI

/// Shared reference for the learning catalog and the evaluation inspector.
struct EvaluationGuideView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            section("Build the evaluation path", "Train the holdout example first. Connect Train’s Trained model output to Evaluate’s Model input, or select Saved checkpoint and choose a verified completed run in Configuration. Connect that model to Model and the matching Partition output to Data. Choose Validation for development or Test for final evaluation. The checkpoint and training-fitted transforms stay fixed. Evaluation uses the exact selected population captured by that training run, even if the original CSV changes or disappears.")
            section("Keep model selection separate", "Use training rows to fit weights and preparation, and validation rows to choose settings. Inspect test results after those choices. Tuning after seeing test results makes that population part of model selection; a new seed or model revision does not make already-seen observations untouched again. Local inspection history records known exposure, not proof that no exposure occurred elsewhere.")
            section("Read the mathematics", "For a batch X with shape [B, F], an affine layer computes Z = XWᵀ + b. A binary classifier ends with logits [B, 2]. The predicted class is the larger logit; an exact tie chooses class 0. Logits are scores, not probabilities.")
            Text("p(c) = exp(z[c] − max(z)) / Σⱼ exp(z[j] − max(z))\nloss = −log p(actual class)")
                .font(.system(size: 12, design: .monospaced)).foregroundStyle(MLTheme.teal).textSelection(.enabled)
            section("Interpret the result", "Mean loss averages all evaluated rows, including the final partial batch. Accuracy counts correct classes. In the confusion table, rows are actual classes and columns are predicted classes. Precision divides correct predictions of a class by all predictions of that class; recall divides them by all actual observations of that class. An empty denominator is unavailable, not zero. High accuracy alone can hide failure on a rare class.")
            section("Control iteration", "Batch size controls memory, not the fraction of selected data evaluated. Preview rows affect display only. Every row of the selected archived Validation or Test population is evaluated; weights and fitted preparation are never updated. CPU, MPS and automatic device selection are configurable. Small device rounding differences can occur; replay checks both numerical tolerance and class decisions.")
            section("Take the model with you", "Open the saved result in History and choose Export evaluation. The default package includes code, weights, frozen preparation, an input contract and summary, without source rows or predictions. Learned statistics and vocabularies can still reveal source information. The optional replay package includes every column at the selected evaluation endpoint, including columns the model does not use. Review the columns before saving. Supplied external observations are labeled external input; the package does not certify an untouched holdout.")
            section("Check your understanding", "If you halve batch size, should the number of evaluated rows halve? No. If every prediction is class 0, can accuracy still be high? Yes, on an imbalanced population; inspect recall for class 1. If test accuracy improves after tuning to its mistakes, is that an independent final estimate? No. Use a genuinely new evaluation population for a new independent estimate.")
        }.font(.caption).textSelection(.enabled)
            .frame(maxWidth: 780, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
    private func section(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption.weight(.semibold))
            Text(detail).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
        }
    }
}
