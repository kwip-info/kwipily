import SwiftUI
import KwipMLCore

/// Accepts the store's verified formative projection, never a private assessment plan.
struct AffineFormativeResultsView: View {
    let projection: AffineFormativeProjection
    var backward = false
    @State private var selectedFixtureID = ""
    private var selected: AffineFormativeFixture? {
        projection.fixtures.first { $0.id == selectedFixtureID } ?? projection.fixtures.first
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionLabel(title: "Measured outputs")
            Text("Formative feedback · does not award assessment completion").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title: "Measured fixture", selection: selected?.id ?? "", options: projection.fixtures.enumerated().map {
                FlatChoiceOption(id: $0.element.id, title: "Input batch \($0.offset + 1)")
            }) { selectedFixtureID = $0 }
            if let fixture = selected {
                Text("Each row is one input example; columns are feature coordinates.").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                matrixPair(fixture.input, firstTitle: backward ? "Upstream gradient g" : "Input x", fixture.measured, secondTitle: backward ? "Measured input gradient" : "Measured output")
                matrixPair(fixture.expected, firstTitle: backward ? "Expected input gradient" : "Expected output", fixture.difference, secondTitle: "Measured − expected", signedSecond: true)
                Text("Differences retain their sign. Small values use scientific notation; displayed values are rounded. Hover a value for full precision.").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                Text("Numerical check: |measured − expected| ≤ \(String(format: "%.6g", projection.absoluteTolerance)) + \(String(format: "%.6g", projection.relativeTolerance)) × |expected|. Passing depends on both tolerances, not display rounding.")
                    .font(.system(size: 10)).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            }
        }
    }
    private func matrixPair(_ first: LearningTensor, firstTitle: String, _ second: LearningTensor, secondTitle: String, signedSecond: Bool = false) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 12) {
                matrix(first, title: firstTitle)
                matrix(second, title: secondTitle, signed: signedSecond)
            }.fixedSize(horizontal: true, vertical: false)
            VStack(alignment: .leading, spacing: 12) {
                matrix(first, title: firstTitle)
                matrix(second, title: secondTitle, signed: signedSecond)
            }
        }
    }
    private func matrix(_ tensor: LearningTensor, title: String, signed: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 10, weight: .semibold)).fixedSize(horizontal: false, vertical: true)
            if tensor.wellFormed, tensor.shape == [2, 2] {
                Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                    ForEach(0..<2, id: \.self) { row in
                        GridRow {
                            ForEach(0..<2, id: \.self) { column in
                                let value = tensor.values[row * 2 + column]
                                Text(String(format: signed ? "%+.7g" : "%.7g", value))
                                    .font(.system(size: 10, design: .monospaced)).monospacedDigit()
                                    .frame(minWidth: 55, minHeight: 27).padding(.horizontal, 3)
                                    .background(MLTheme.background)
                                    .help(String(format: "%.17g", value))
                                    .accessibilityLabel("\(title), row \(row), column \(column), \(String(format: "%.17g", value))")
                            }
                        }
                    }
                }.overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1)).textSelection(.enabled)
            }
        }
    }
}
