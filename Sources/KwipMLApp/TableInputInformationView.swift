import SwiftUI
import KwipMLCore

/// Explanatory copy lives separately from the working configuration controls.
struct TableInputInformationView: View {
    var configuration: [String: JSONValue] = [:]
    private var format: TableInputConfiguration.Format { (try? TableInputConfiguration.format(of: configuration)) ?? .csv }
    private var nativeCSV: Bool { configuration["parserProfile"] == .string("polars") }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            section("What this tool does", "Reads one selected source into a table with named columns. Choose the format and its reading options here; connect other components to cast, filter, join or prepare the resulting data. A reader parses its own format rather than guessing how separate files should be combined.")
            section("Source indicators", format == .database
                ? "Connection configured means a database source has been chosen; it does not establish that authentication or a read succeeded. Connect captures the selected remote table into a private snapshot. Verified preview means up to eight rows of that snapshot were read successfully. Credentials remain private and are not saved in the portable workflow. Connect again to fetch current database values; Refresh preview checks the existing captured snapshot."
                : "File linked means a local file has been chosen, but its contents have not yet been verified. Verified preview means the selected reader produced a bounded table sample. External file changes are not watched automatically; Refresh preview checks the source again. Local file permissions stay outside the portable workflow.")
            section("Changes and refresh", "Unapplied changes have not been saved. Apply & preview saves them and refreshes the table. Until then, displayed evidence belongs to the saved settings. Out of date means the retained evidence no longer matches the current source or configuration. A cancelled inspection can be retried.")
            section("Table indicators", "Results → Output shows one table. Preview identifies source-inspection data; Run output identifies captured workflow data. Both are recorded snapshots, not live views of the source. Full reading errors belong in Diagnostics. Empty text and null are different: the table marks an empty string as \"\" and a missing value as null.")
            ForEach(Array(formatSections.enumerated()), id: \.offset) { _, item in section(item.0, item.1) }
            section("Preview and run limits", format == .csv
                ? "Source preview reads at most 1 MiB and displays at most eight rows, 128 columns and 200 characters per cell. It verifies that prefix sample, not later records or complete-file identity; total source row count remains unknown. Full execution has separate source-byte, selected-cell and temporary-storage budgets. A successful preview does not guarantee that every later record will parse."
                : "Source preview displays at most eight rows, 128 columns and 200 characters per cell. File readers capture and hash the complete selected file within the 10 MB source budget. Parquet and JSON Lines can limit their parsed rows; JSON documents, spreadsheets and markup may still parse a bounded document or workbook before producing the sample. Remote database Connect captures the configured table within the source budget; preview displays only eight rows of that snapshot. SQLite reads the selected local database snapshot. Sample size does not establish a complete row count or validate every later record. Full runs have separate cell, expanded-data and storage budgets.")
            section("Sampling and complete data", "Canvas Sampling is a development work limit: it takes the first records in source order, or a prefix calculated from the selected percentage. It is not a random or representative sample. A View, file output or training consumer requests complete upstream data for its branch, within the resource budget. Complete means all rows allowed by the source's explicit read settings; a configured row limit still applies. Sample rows is a separate component that deliberately chooses a seeded subset without replacement and changes the table passed onward.")
        }
    }
    private var formatSections: [(String, String)] {
        switch format {
        case .csv:
            return [("CSV / delimited text", nativeCSV
                ? "The Polars reader supports a one-byte delimiter and quote, doubled quote escaping, comments, logical-record or physical-line skips, header choices, UTF-8 handling, null markers and empty-field policy. Values remain text by default; advanced schema and inference choices can read supported typed columns. Column selection, read limits, row indexes, decimal comma and explicit error/extra/missing-column policies affect the source table. Downstream Cast columns remains useful for a separate visible conversion. Quoted newlines belong to a single record. Inspect actual output before using unfamiliar parsing settings."
                : "The compatibility reader supports a single-character delimiter and quote, custom escaping, encoding choices and record-ending rules. Escaping reads the next character literally. Records to skip counts logical records, including multiline quoted fields. Uneven field counts stop parsing rather than silently dropping values."),
                    ("CSV missing values", "Null markers match exact, case-sensitive field text after unquoting. Empty text and missing values follow the selected policy. A preview checks only its first records; errors farther into the source are discovered during execution.")]
        case .parquet:
            return [("Parquet", "Parquet already records column types and row groups. Choose columns and an optional read limit before loading, then choose native Polars or the PyArrow reader. Statistics and parallelism affect reading strategy; they do not filter or sample observations. Schema options must match supported scalar columns. Extra-column and integer/float/categorical cast policies are explicit; allowing integer-to-float or float downcasts can lose precision. Nested, binary, decimal and temporal values are not automatically flattened."),
                    ("Compressed data", "A small file may expand into a much larger table. Metadata, selected-cell and expanded row-group budgets are checked separately. PyArrow metadata/checksum settings remain bounded; encrypted datasets, custom filesystems and extension objects are not admitted.")]
        case .json:
            return [("JSON document", "Read a document containing tabular records, using the selected schema or inference settings. Integer, floating-point, Boolean and text columns retain their types. Missing fields can produce nulls. Named schema and field maps sort by column name for reproducible output; use an ordered Columns selection to choose display order. Nested objects and arrays are not silently flattened into extra columns; choose a compatible tabular source or explicit supported types."),
                    ("Document reading", "A JSON document may need a complete bounded parse before a preview can be selected. JSON Lines is a separate format designed for one record per line.")]
        case .jsonl:
            return [("JSON Lines", "Each line contains one JSON record. Batch size, inference length, read limit and low-memory settings control reading. Schema overrides guide parsing; they are not a promise that arbitrary strings can be converted into numbers. Ignoring errors may create nulls or lose information; keep it off when strict reading is required.")]
        case .excel:
            return [("Workbook selection", "Choose one worksheet by name or index, or a named table with a supported engine. Calamine is the general reader; openpyxl and xlsx2csv offer alternate XLSX interpretation. Only options that apply to the chosen engine are shown. Header position, skipped rows, column selection, empty rows and type choices can change the output."),
                    ("Spreadsheet values", "The reader does not execute macros or recalculate formulas. Cached formula values can be stale or absent. Dates, mixed cell types and number formatting differ between engines; inspect the resulting values and schema. Workbook archive expansion is checked separately from compressed file size. External links and VBA preservation stay disabled.")]
        case .markdown:
            return [("Markdown tables", "Choose one pipe table by its zero-based index. Header, encoding, column selection and row-limit settings control the resulting table. Cell representation can keep Markdown or extract text; links can retain their label, destination or both. Image references remain text references and do not read image pixels.")]
        case .xml:
            return [("XML records", "Select record elements with a row XPath. Explicit column-to-XPath fields can select one scalar value per record; namespace mappings resolve prefixed names. Without explicit fields, simple child elements and optionally attributes become columns. Repeated or nested fields need an explicit scalar selection rather than automatic flattening."),
                    ("XML boundaries", "DTDs, entity expansion, external resources and unrestricted huge-tree parsing are disabled. XML is read as data; XPath expressions cannot install or execute custom functions.")]
        case .html:
            return [("HTML tables", "Choose one stored table by its zero-based index. Select a header row or generated headers, skip leading rows and choose columns. Merged cells are rejected unless Expand is selected. Hidden-row handling reflects stored attributes and inline styles; it is not a browser layout calculation. Links can retain labels, destinations or both."),
                    ("HTML boundaries", "This reader does not execute scripts, render a page, load linked resources or visit URLs. Text mode excludes script/style contents; HTML cell mode retains markup as data. Nested tables are separate selections rather than silently merged into the parent table.")]
        case .api:
            return [("API source", "Fetch snapshot sends the saved GET request through the isolated broker and records every configured page as a typed table. Run Flow fetches a new snapshot. Refresh preview and History inspect captured data without contacting the endpoint."),("Schema and paging", "Use JSON Pointer paths such as /data/items. Map explicit names and types; unknown fields are ignored and missing fields follow the selected policy. Cursor/next-link paging stops only when the source ends; page/offset paging requires an endpoint page size matching the configured size. Repeated links, limits, malformed values or HTTP failures publish no complete snapshot."),("Private access and limits", "Save authentication privately for the exact origin and mode. HTTPS verifies certificates and hostnames. Redirects and private destinations are refused; literal loopback HTTP is an explicit development option. The ordinary execution worker cannot access the network. Budgets bound pages, bytes, rows, time and request rate. Empty responses retain the declared column schema.")]
        case .database:
            return [("Selected database table", "SQLite reads a selected local file; PostgreSQL and MySQL use a separate source-capture broker. Choose an exact database/schema/table, columns, ordering, read limit and offset. No arbitrary SQL is executed from the workflow. An explicit stable order is useful when comparing repeated captures, because a database does not otherwise promise row order."),
                    ("Connections and captured values", "Remote TLS verifies the server certificate and hostname. Unencrypted connections are limited to an explicitly selected literal loopback address for local development. Username and password are kept outside portable configuration and run artifacts. Decimal, date/time and binary decoding policies are explicit. The normal preparation worker consumes a private typed snapshot rather than connecting to the database itself.")]
        }
    }
    private func section(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(MLTheme.teal)
            Text(detail).font(.caption).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
        }
    }
}
