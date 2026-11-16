import SwiftUI
import KwipMLCore

/// Displays only the supplied maps, never their combined matrix or bias.
struct AffineTaskReferenceView: View {
    let reference: AffineCompositionReference
    var instructions: String? = nil
    var conclusion:String? = nil
    var showsCombinedEquation=true
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            LessonRichText(text: instructions ?? "Apply the first public transformation, then the second. Each weight-table row produces one output feature; each column receives one input feature. The offset table adds a fixed number to each output. These are reference values, not your completed answer.")
            referenceMap(reference.first, weightLabel: "First weights", biasLabel: "First offsets")
            referenceMap(reference.second, weightLabel: "Second weights", biasLabel: "Second offsets")
            if let conclusion {LessonRichText(text:conclusion)}
            if showsCombinedEquation {
            InspectorSection("Optional · combined equation", symbol: "function", identifier: "lesson.affine.referenceEquation", initiallyExpanded: false) {
                LessonRichText(text: "x is one input row’s values and y is the final output. W₁ and W₂ are the first and second weight tables; b₁ and b₂ are their offsets. Start inside the parentheses: apply the first map, then the second. This equation describes the public forward task, not the combined weights you must enter or its backward answer.")
                Text("y = W₂(W₁x + b₁) + b₂").font(.system(size: 12, design: .monospaced)).foregroundStyle(MLTheme.teal).textSelection(.enabled)
            }
            }
        }
    }
    private func referenceMap(_ map: AffineReferenceMap, weightLabel: String, biasLabel: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(map.name).font(.system(size: 11, weight: .semibold))
                Spacer()
                Text("\(map.inputDimension) input → \(map.outputDimension) output features").font(.system(size: 10, design: .monospaced)).foregroundStyle(MLTheme.muted)
            }
            HStack(alignment: .top, spacing: 14) {
                tensor(map.weight, label: weightLabel)
                tensor(map.bias, label: biasLabel)
                Spacer(minLength: 0)
            }
        }
    }
    private func tensor(_ tensor: LearningTensor, label: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label).font(.system(size: 12, weight: .medium, design: .monospaced))
            if tensor.wellFormed, tensor.shape.count == 1 || tensor.shape.count == 2 {
                let rows = tensor.shape[0]
                let columns = tensor.shape.count == 2 ? tensor.shape[1] : 1
                Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                    ForEach(0..<rows, id: \.self) { row in
                        GridRow {
                            ForEach(0..<columns, id: \.self) { column in
                                Text(tensor.values[row * columns + column].formatted(.number.precision(.fractionLength(0...6))))
                                    .font(.system(size: 12, design: .monospaced)).monospacedDigit()
                                    .frame(minWidth: 38, minHeight: 26).padding(.horizontal, 3)
                                    .background(MLTheme.background)
                                    .accessibilityLabel("\(label), row \(row), column \(column), \(tensor.values[row * columns + column])")
                            }
                        }
                    }
                }.overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1)).textSelection(.enabled)
                Text(tensor.shape.map(String.init).joined(separator: " × ")).font(.system(size: 9, design: .monospaced)).foregroundStyle(MLTheme.muted)
            }
        }
    }
}
