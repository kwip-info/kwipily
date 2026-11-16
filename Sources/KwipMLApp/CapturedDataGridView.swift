import SwiftUI
import AppKit

/// Renders only the retained sample. It never requests or reconstructs source rows.
struct CapturedDataGridView: View {
    let grid: CapturedDataGrid
    var tableOnly = false
    var evidenceLabel: String? = nil
    var imageResolver: FileImageArtifactResolver? = nil
    @State private var mediaSelection:MediaArtifactSelection?
    @State private var imageSelection: FileImageSelection?
    @State private var query = ""
    @State private var selectedRow: Int?
    @State private var sort = TableGridSort()
    private var rows: [CapturedDataGrid.Row] { grid.displayRows(query: query, sort: sort) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Text(evidenceLabel ?? "Table · \(grid.partition)").font(.system(size: 12, weight: .semibold))
                if tableOnly {
                    Text((grid.totalRows.map { "Sample · \(rows.count)/\($0) rows" } ?? "\(rows.count) sampled rows") + " · \(grid.columns.count) columns")
                        .font(.caption2).foregroundStyle(MLTheme.muted)
                }
                Spacer()
                TextField("Find in captured rows", text: $query).flatField().frame(width: 190)
                    .accessibilityLabel("Find in captured rows")
                Button("Copy selected row") { copySelection() }.buttonStyle(FlatButtonStyle())
                    .disabled(!rows.contains { $0.index == selectedRow })
            }
            if tableOnly {
                table.frame(minHeight: 80, maxHeight: .infinity)
            } else {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 12) {
                    table.frame(minWidth: 450, minHeight: 220, maxHeight: 220)
                    schema.frame(width: 230)
                }
                VStack(alignment: .leading, spacing: 10) {
                    table.frame(height: 220)
                    DisclosureGroup("Schema · \(grid.columns.count) columns") { schema }
                }
            }
            Text("\(rows.count) shown · \(grid.rows.count) retained sample rows" + (grid.totalRows.map { " · \($0) total rows" } ?? " · total not recorded"))
                .font(.caption2).foregroundStyle(MLTheme.muted)
            Text("Search, sorting and copy use the retained preview only. Sample numbers are positions in this preview, not source row IDs." + (grid.truncated ? " Worker preview was truncated." : ""))
                .font(.caption2).foregroundStyle(MLTheme.muted)
            if grid.hasUnmappedValues {
                Text("Some retained values have no captured column definition. Raw preview below preserves that evidence.")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            }
            }
        }
        .sheet(item:$mediaSelection) { selected in if let imageResolver {MediaArtifactPreviewView(reference:selected.reference,resolver:imageResolver,selectionInterval:selected.interval)} }
        .sheet(item: $imageSelection) { selected in
            if let imageResolver { FileImagePreviewView(reference: selected.reference, resolver: imageResolver) }
        }
        .onChange(of: grid.selectionIdentity) { _, _ in selectedRow = nil }
        .onChange(of: query) { _, _ in if !rows.contains(where: { $0.index == selectedRow }) { selectedRow = nil } }
    }

    private var table: some View {
        CapturedNativeTable(columns: grid.columns, rows: rows, selectedRow: $selectedRow, sort: sort, onSort: { sort = sort.toggled($0) }, imageResolver: imageResolver, onOpenImage: { imageSelection = .init(reference: $0) },onOpenMedia:{mediaSelection = .init(reference:$0)},onOpenMediaRow:{row,ref in mediaSelection = .init(reference:ref,interval:SpeechSegmentPresentation.interval(row:row,columns:grid.columns))})
            .overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1))
    }

    private var schema: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("SCHEMA").font(.system(size: 10, weight: .semibold)).foregroundStyle(MLTheme.muted)
                .padding(10).frame(maxWidth: .infinity, alignment: .leading).background(MLTheme.panel)
            ForEach(grid.columns) { column in
                VStack(alignment: .leading, spacing: 3) {
                    Text(column.label).font(.system(size: 11, weight: .semibold))
                    if column.label != column.name { Text(column.name).font(.system(size: 10, design: .monospaced)) }
                    Text("\(column.dtype) · \(column.nullability)").font(.system(size: 10)).foregroundStyle(MLTheme.muted)
                }.textSelection(.enabled).padding(8).frame(maxWidth: .infinity, alignment: .leading)
                Rectangle().fill(MLTheme.rule).frame(height: 1)
            }
        }.overlay(Rectangle().stroke(MLTheme.rule, lineWidth: 1))
    }

    private func copySelection() {
        guard let row = rows.first(where: { $0.index == selectedRow }) else { return }
        let text = grid.columns.map { $0.name }.map(escape).joined(separator: "\t") + "\n" + row.cells.map { escape($0.text) }.joined(separator: "\t")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
    private func escape(_ value: String) -> String {
        if value.contains("\t") || value.contains("\n") || value.contains("\"") { return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        return value
    }
}
