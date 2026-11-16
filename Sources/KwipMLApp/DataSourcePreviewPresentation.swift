import Foundation
import KwipMLCore

/// Already verified, bounded worker evidence. This model performs no source I/O.
struct DataSourcePreviewPresentation {
    let grid: CapturedDataGrid
    let summary: String
    let interpretation: String
    let budget: String
    init(result: SourceInspectionResult?) {
        interpretation = result?.columnSchema == nil
            ? "Column names follow the saved header setting. Values remain text or explicit nulls; use Cast Columns for types. Empty text is shown as \"\" and missing values as null."
            : "Column names and scalar types come from the verified reader snapshot. Presentation does not infer or convert displayed values. Empty text is shown as \"\" and missing values as null; nullability remains unknown for uninspected records."
        let csv = result?.parsing.map { (try? TableInputConfiguration.format(of: $0.options)) == .csv } ?? true
        budget = csv
            ? "CSV preview captures at most 1 MiB and displays up to 8 rows, 128 columns and 200 characters per cell. Total rows and unsampled records are not validated. Full runs have a 10 MB source budget and separate selected-cell/storage limits."
            : "Structured preview captures the complete admitted file within 10 MB, then displays up to 8 rows, 128 columns and 200 characters per cell. A preview does not establish full row count or validate later records. Full execution has separate selected-cell/storage limits."
        guard let result else {
        grid = CapturedDataGrid(preview: [:]); summary = "Source not inspected. Choose a source or Refresh to see verified sample rows."; return
        }
        let missing = Set(result.nullCells ?? [])
        let rows: [[Any]] = result.rows.enumerated().map { r, values in
            values.enumerated().map { c, value -> Any in missing.contains([r, c]) ? NSNull() : value }
        }
        var preview: [String: Any] = [
            "columns": result.columnSchema?.map { ["name": $0.name, "dtype": $0.dtype] } ?? result.columns.map { ["name": $0, "dtype": "string"] },
            "rows": rows, "rowCount": result.rowCount, "partition": "source sample",
            "previewTruncated": result.parsing?.sampleOnly == true || result.rows.count < result.rowCount
        ]
        if result.parsing?.sampleOnly == true { preview.removeValue(forKey: "rowCount") }
        grid = CapturedDataGrid(preview: preview)
        summary = result.parsing?.sampleOnly == true ? "Showing \(result.rows.count) sampled rows · total unknown · \(result.columns.count) columns" : "Showing \(result.rows.count) of \(result.rowCount) inspected rows · \(result.columns.count) columns"
    }
}
