import SwiftUI

/// A compact source table without a second source reader or inferred cell types.
struct DataSourcePreviewView: View {
    let presentation: DataSourcePreviewPresentation
    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Source preview").font(.caption.weight(.semibold))
            Text(presentation.summary).font(.caption).foregroundStyle(MLTheme.muted)
            if !presentation.grid.columns.isEmpty {
                ThemedScrollView(.horizontal) {
                    VStack(alignment: .leading, spacing: 0) {
                        HStack(spacing: 0) {
                            cell("Row", width: 38, header: true)
                            ForEach(presentation.grid.columns) { column in cell(column.name, width: 125, header: true) }
                        }.background(MLTheme.panel)
                        Rectangle().fill(MLTheme.rule).frame(height: 1)
                        ForEach(presentation.grid.rows) { row in
                            HStack(spacing: 0) {
                                cell(String(row.index + 1), width: 38)
                                ForEach(Array(row.cells.enumerated()), id: \.offset) { _, value in cell(value.text, width: 125) }
                            }
                            Rectangle().fill(MLTheme.rule).frame(height: 1)
                        }
                    }.fixedSize(horizontal: true, vertical: false)
                }.overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1))
            }
            Text(presentation.interpretation).font(.caption2).foregroundStyle(MLTheme.muted)
            Text(presentation.budget).font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    private func cell(_ text: String, width: CGFloat, header: Bool = false) -> some View {
        Text(text).font(.system(size: 10, weight: header ? .semibold : .regular, design: .monospaced))
            .lineLimit(1).textSelection(.enabled).help(text).padding(.horizontal, 6)
            .frame(width: width, height: header ? 30 : 28, alignment: .leading)
            .overlay(alignment: .trailing) { Rectangle().fill(MLTheme.rule).frame(width: 1) }
    }
}
