import SwiftUI
import KwipMLCore

/// A bounded read-only tensor view with explicit leading-coordinate slices. Never accepts a lesson plan.
struct LearningTensorTable: View {
    let tensor: LearningTensor
    let title: String
    var signed = false
    @State private var rowPage = 0
    @State private var columnPage = 0
    @State private var sliceIndex=0
    private var slice:LearningTensorMatrixSlice? {tensor.matrixSlice(at:min(sliceIndex,max(0,tensor.matrixSliceCount-1)))}
    private var matrix:LearningTensor {slice?.matrix ?? tensor}
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 11, weight: .semibold))
            Text("Shape [\(tensor.shape.map(String.init).joined(separator: ", "))]")
                .font(.system(size: 10, design: .monospaced)).foregroundStyle(MLTheme.muted)
            if let slice,!slice.leadingCoordinates.isEmpty {
                Text("Fixed coordinates · "+slice.leadingCoordinates.enumerated().map {"axis \($0.offset)=\($0.element)"}.joined(separator:", "))
                    .font(.system(size:10,design:.monospaced)).textSelection(.enabled)
                HStack {
                    Button("Previous slice") {sliceIndex=max(0,sliceIndex-1);rowPage=0;columnPage=0}.disabled(sliceIndex==0)
                    Button("Next slice") {sliceIndex+=1;rowPage=0;columnPage=0}.disabled(sliceIndex>=tensor.matrixSliceCount-1)
                    Text("\(sliceIndex+1) of \(tensor.matrixSliceCount)")
                }.font(.caption).buttonStyle(.borderless)
            }
            if let page = MatrixPage(tensor: matrix, rowPage: rowPage, columnPage: columnPage) {
                ThemedScrollView(.horizontal) {
                    Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                        ForEach(Array(page.rows), id: \.self) { row in
                            GridRow {
                                ForEach(Array(page.columns), id: \.self) { column in
                                    if let value = page.value(row: row, column: column) {
                                        Text(String(format: signed ? "%+.7g" : "%.7g", value))
                                            .font(.system(size: 11, design: .monospaced)).monospacedDigit()
                                            .frame(minWidth: 65, minHeight: 27).padding(.horizontal, 3)
                                            .background(MLTheme.background)
                                            .help(String(format: "%.17g", value))
                                            .accessibilityLabel("\(title), leading coordinates \(slice?.leadingCoordinates.map(String.init).joined(separator: ", ") ?? ""), row \(row), column \(column), \(String(format: "%.17g", value))")
                                    }
                                }
                            }
                        }
                    }.overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1)).textSelection(.enabled)
                }.frame(height: CGFloat(page.rows.count) * 27 + 2)
                if page.rowPageCount > 1 || page.columnPageCount > 1 {
                    HStack {
                        Button("Previous rows") { rowPage = max(0, rowPage - 1) }.disabled(page.rows.lowerBound == 0)
                        Button("Next rows") { rowPage += 1 }.disabled(page.rows.upperBound == matrix.shape[0])
                        if page.columnPageCount > 1 {
                            Button("Previous columns") { columnPage = max(0, columnPage - 1) }.disabled(page.columns.lowerBound == 0)
                            Button("Next columns") { columnPage += 1 }.disabled(page.columns.upperBound == matrix.shape[1])
                        }
                    }.font(.system(size: 10)).buttonStyle(.borderless)
                    Text("Rows \(page.rows.lowerBound + 1)–\(page.rows.upperBound), columns \(page.columns.lowerBound + 1)–\(page.columns.upperBound)")
                        .font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                }
            } else { Text("This value cannot be displayed as a finite vector or matrix.").font(.caption).foregroundStyle(MLTheme.rose) }
        }
    }
}
