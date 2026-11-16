import Foundation

/// Portable source-reader configuration. Source access and database credentials stay in private bindings.
public enum TableInputConfiguration {
    public enum Format: String, CaseIterable, Identifiable {
        case csv, parquet, json, jsonl, excel, markdown, xml, html, database, api
        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .csv: return "CSV / delimited text"
            case .parquet: return "Parquet"
            case .json: return "JSON document"
            case .jsonl: return "JSON Lines"
            case .excel: return "Excel workbook"
            case .markdown: return "Markdown table"
            case .xml: return "XML records"
            case .html: return "HTML table"
            case .database: return "Database"
            case .api: return "API · JSON REST"
            }
        }
    }
    public enum FieldKind: Equatable {
        case text(maximum: Int, multiline: Bool)
        case character
        case floatCast
        case integer(minimum: Int64, maximum: Int64, optional: Bool)
        case toggle
        case integerChoice([Int64])
        case listOrMap
        case choice([String])
        case stringList
        case columns
        case columnsOrRange
        case integerOrList
        case choiceOrMap([String])
        case stringMap
        case nullMarkerMap
        case schema
    }
    public struct Field: Identifiable, Equatable {
        public let key: String
        public let title: String
        public let kind: FieldKind
        public let defaultValue: JSONValue
        public let advanced: Bool
        public let help: String
        public var id: String { key }
        public init(_ key: String, _ title: String, _ kind: FieldKind, _ defaultValue: JSONValue, advanced: Bool = false, help: String = "") {
            self.key = key; self.title = title; self.kind = kind; self.defaultValue = defaultValue; self.advanced = advanced; self.help = help
        }
    }
    public static let scalarTypes = ["String", "Boolean", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"]
    public static var csvAdvancedDefaults: [String: JSONValue] {
        var defaults = Dictionary(uniqueKeysWithValues: fields(for: .csv).map { ($0.key, $0.defaultValue) })
        defaults["usePyarrow"] = .bool(false)
        return defaults
    }
    public static func validateCSVAdvanced(_ options: [String: JSONValue]) throws -> [String: JSONValue] {
        let defaults = csvAdvancedDefaults
        guard Set(options.keys).isSubset(of: Set(defaults.keys)) else { throw DataFlowContractError("Unknown Polars CSV reading option.") }
        let result = defaults.merging(options) { _, value in value }
        for field in fields(for: .csv) { try validate(result[field.key]!, field: field) }
        if result["schema"] != .object([:]) && result["schemaOverrides"] != .object([:]) { throw DataFlowContractError("Choose a complete CSV schema or column type overrides, not both.") }
        if result["rowIndexOffset"] != .integer(0), result["rowIndexName"] == .string("") { throw DataFlowContractError("A row index offset requires an index column name.") }
        guard result["usePyarrow"] == .bool(false) else { throw DataFlowContractError("CSV uses the native streaming reader. The PyArrow CSV reader cannot preserve this profile's lexical-text and bounded-streaming guarantees.") }
        return result
    }
    public static func transportJSON(_ config: [String: JSONValue]) throws -> String {
        if try format(of: config) == .csv { return try CSVParsingConfiguration.transportJSON(options(in: config)) }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(validate(config))
        guard data.count <= 32_768, let json = String(data: data, encoding: .utf8) else { throw DataFlowContractError("Table Input settings exceed the transport limit.") }
        return json
    }
    public static func format(of config: [String: JSONValue]) throws -> Format {
        guard let value = config["format"] else { return .csv }
        guard case .string(let name) = value, let format = Format(rawValue: name) else { throw DataFlowContractError("Choose a supported Table Input format.") }
        return format
    }
    public static func options(in config: [String: JSONValue]) throws -> [String: JSONValue] {
        if try format(of: config) == .csv { var csv = config; csv.removeValue(forKey: "format"); return csv }
        guard case .object(let options) = config["options"] else { throw DataFlowContractError("Table Input reading options must be an object.",apiIssue:(try? format(of:config)) == .api ? .inputEnvelope:nil) }
        return options
    }
    public static func defaults(for format: Format) -> [String: JSONValue] {
        if format == .csv { return CSVParsingConfiguration.newDefaults }
        return ["format": .string(format.rawValue), "options": .object(format == .api ? APITableSettings.defaults:defaultOptions(for: format))]
    }
    /// Format changes are draft-only. The inspector owns remembered drafts for other formats.
    public static func replacingFormat(_ config: [String: JSONValue], with format: Format) -> [String: JSONValue] {
        if (try? self.format(of: config)) == format { return config }
        return defaults(for: format)
    }
    public static func validate(_ config: [String: JSONValue]) throws -> [String: JSONValue] {
        let format = try format(of: config)
        if format == .csv {
            var csv = config; csv.removeValue(forKey: "format")
            return try CSVParsingConfiguration.validate(csv)
        }
        guard Set(config.keys).isSubset(of: ["format", "options"]) else { throw DataFlowContractError("Unknown Table Input configuration field.",apiIssue:format == .api ? .inputEnvelope:nil) }
        if format == .api {return ["format":.string("api"),"options":.object(try APITableSettings.normalized(options(in:config)))]}
        let options = try options(in: config), fields = fields(for: format, options: options)
        try checkKeys(options, prefix: "", fields: fields, format: format)
        var normalized = defaultOptions(for: format, options: options)
        for field in fields {
            if let selected = value(for: field.key, in: options) { set(selected, for: field.key, in: &normalized) }
            try validate(value(for: field.key, in: normalized)!, field: field)
            if field.kind == .floatCast, case .array(let choices) = value(for: field.key, in: normalized) { set(choices.count == 1 ? choices[0] : .array([.string("downcast"), .string("upcast")]), for: field.key, in: &normalized) }
        }
        if format == .database { normalized = try DatabaseSourceSettings.normalized(normalized, requireTable: false) }
        if format == .parquet, normalized["use_pyarrow"] == .bool(true) {
            guard normalized["schema"] == .object([:]) else { throw DataFlowContractError("Declared Parquet schema applies to the native reader. Select columns with the PyArrow reader.") }
            guard normalized["parallel"] == .string("auto"), normalized["use_statistics"] == .bool(true), normalized["low_memory"] == .bool(false), normalized["missing_columns"] == .string("raise"), normalized["extra_columns"] == .string("raise"), normalized["cast_options"] == .object(["integer_cast": .string("forbid"), "float_cast": .string("forbid"), "categorical_to_string": .string("forbid")]) else { throw DataFlowContractError("Polars read-strategy tuning does not apply to the PyArrow reader.") }
        }
        if format == .parquet, normalized["use_pyarrow"] != .bool(true), normalized["memory_map"] != .bool(true) { throw DataFlowContractError("Memory-map tuning applies to the PyArrow reader. Native reads use Polars streaming.") }
        if format == .excel {
            func populated(_ key: String) -> Bool {
                guard let value = value(for: key, in: normalized) else { return false }
                switch value { case .null: return false; case .string(let text): return !text.isEmpty; case .array(let values): return !values.isEmpty; case .object(let values): return !values.isEmpty; default: return true }
            }
            guard !(populated("sheet_name") && populated("table_name")), !(populated("columns") && populated("read_options.use_columns")) else { throw DataFlowContractError("Choose one worksheet/table selection and one column selection.") }
            if populated("table_name"), normalized["engine"] != .string("calamine") { throw DataFlowContractError("Named workbook tables require the Calamine reader.") }
            if populated("sheet_name") || populated("table_name"), case .integer(let sheet) = normalized["sheet_id"], sheet != 1 { throw DataFlowContractError("Choose one worksheet name, worksheet number or named table.") }
            if populated("table_name"), case .array = value(for: "read_options.skip_rows", in: normalized) { throw DataFlowContractError("Named workbook tables use an initial skipped-row count, not selected row indexes.") }
            if normalized["has_header"] == .bool(false), let header = value(for: "read_options.header_row", in: normalized), header != .null { throw DataFlowContractError("A header row requires first-row column names.") }
            guard !(populated("read_options.dtypes") && populated("schema_overrides")) else { throw DataFlowContractError("Choose either reader types or column type overrides.") }
        }
        return ["format": .string(format.rawValue), "options": .object(normalized)]
    }
    private static func validate(_ value: JSONValue, field: Field) throws {
        func invalid() throws -> Never { throw DataFlowContractError("Choose a valid \(field.title.lowercased()).") }
        switch field.kind {
        case .floatCast:
            if case .string(let choice) = value, ["forbid", "upcast", "downcast"].contains(choice) { return }
            guard case .array(let choices) = value, (1...2).contains(choices.count), Set(choices).count == choices.count, choices.allSatisfy({ $0 == .string("upcast") || $0 == .string("downcast") }) else { try invalid() }
        case .character:
            guard case .string(let character) = value, character.utf8.count == 1, character != "\0" else { try invalid() }
        case .text(let maximum, _):
            if field.defaultValue == .null && value == .null { return }
            guard case .string(let text) = value, text.unicodeScalars.count <= maximum, !text.contains("\0") else { try invalid() }
        case .integer(let minimum, let maximum, let optional):
            if optional && value == .null { return }
            guard case .integer(let number) = value, (minimum...maximum).contains(number) else { try invalid() }
        case .integerChoice(let choices):
            guard case .integer(let selected) = value, choices.contains(selected) else { try invalid() }
        case .listOrMap:
            if case .object(let values) = value {
                guard values.count <= 128, values.allSatisfy({ key, item in if case .string(let text) = item { return !key.isEmpty && key.count <= 200 && !text.isEmpty && text.count <= 200 }; return false }) else { try invalid() }
            } else {
                guard case .array(let values) = value, values.count <= 128, values.allSatisfy({ if case .string(let text) = $0 { return !text.isEmpty && text.count <= 200 }; return false }) else { try invalid() }
            }
        case .toggle:
            guard case .bool = value else { try invalid() }
        case .choice(let choices):
            guard case .string(let choice) = value, choices.contains(choice) else { try invalid() }
        case .integerOrList:
            if case .integer(let number) = value, (0...100_000).contains(number) { return }
            guard case .array(let numbers) = value, numbers.count <= 100_000, Set(numbers).count == numbers.count, numbers.allSatisfy({ if case .integer(let n) = $0 { return (0...100_000).contains(n) }; return false }) else { try invalid() }
        case .choiceOrMap(let choices):
            if value == .null { return }
            if case .string(let selected) = value, choices.contains(selected) { return }
            guard case .object(let object) = value, object.count <= 256 else { try invalid() }
            for (key, entry) in object { guard !key.isEmpty, key.count <= 500, case .string(let choice) = entry, choices.contains(choice) else { try invalid() } }
        case .stringList, .columns, .columnsOrRange:
            if field.kind == .columnsOrRange, case .string(let range) = value { guard range.count <= 500, !range.contains("\0") else { try invalid() }; return }
            if field.defaultValue == .null && value == .null { return }
            guard case .array(let items) = value, items.count <= 128 else { try invalid() }
            if (field.kind == .columns || field.kind == .columnsOrRange) && items.allSatisfy({ if case .integer(let index) = $0 { return (0...127).contains(index) }; return false }) {
                guard Set(items).count == items.count else { try invalid() }; return
            }
            var seen = Set<String>()
            for item in items {
                guard case .string(let text) = item, !text.isEmpty, text.unicodeScalars.count <= 200, !text.contains("\0"), seen.insert(text).inserted else { try invalid() }
            }
        case .stringMap, .nullMarkerMap, .schema:
            if field.defaultValue == .null && value == .null { return }
            guard case .object(let entries) = value, entries.count <= 128 else { try invalid() }
            for (key, entry) in entries {
                guard !key.isEmpty, key.unicodeScalars.count <= 200, !key.contains("\0"), case .string(let text) = entry, (field.kind == .nullMarkerMap || !text.isEmpty), text.unicodeScalars.count <= (field.kind == .stringMap ? 2000 : 200), !text.contains("\0") else { try invalid() }
                if field.kind == .schema && !scalarTypes.contains(text) { try invalid() }
            }
        }
    }
    public static func value(for key: String, in options: [String: JSONValue]) -> JSONValue? {
        let parts = key.split(separator: ".").map(String.init)
        guard !parts.isEmpty else { return nil }
        var object = options
        for part in parts.dropLast() { guard case .object(let child) = object[part] else { return nil }; object = child }
        return object[parts.last!]
    }
    public static func set(_ value: JSONValue, for key: String, in options: inout [String: JSONValue]) {
        let parts = key.split(separator: ".").map(String.init)
        guard !parts.isEmpty else { return }
        func update(_ parts: ArraySlice<String>, _ object: inout [String: JSONValue]) {
            let first = parts.first!
            if parts.count == 1 { object[first] = value; return }
            var child: [String: JSONValue] = [:]
            if case .object(let old) = object[first] { child = old }
            update(parts.dropFirst(), &child); object[first] = .object(child)
        }
        update(parts[...], &options)
    }
    private static func defaultOptions(for format: Format, options: [String: JSONValue] = [:]) -> [String: JSONValue] {
        var result: [String: JSONValue] = [:]
        for field in fields(for: format, options: options) { set(field.defaultValue, for: field.key, in: &result) }
        return result
    }
    private static func checkKeys(_ options: [String: JSONValue], prefix: String, fields: [Field], format: Format) throws {
        for (key, value) in options {
            let path = prefix.isEmpty ? key : "\(prefix).\(key)"
            if fields.contains(where: { $0.key == path }) { continue }
            guard fields.contains(where: { $0.key.hasPrefix(path + ".") }), case .object(let child) = value else { throw DataFlowContractError("Unknown \(format.title) reading option: \(path).") }
            try checkKeys(child, prefix: path, fields: fields, format: format)
        }
    }
    private static func pyarrowFields() -> [Field] {
        func flag(_ key: String, _ title: String, _ value: Bool) -> Field { Field("pyarrow_options." + key, title, .toggle, .bool(value), advanced: true) }
        return [flag("use_threads", "PyArrow parallel threads", true), flag("use_pandas_metadata", "Read pandas metadata", false),
            Field("pyarrow_options.buffer_size", "PyArrow read buffer · bytes", .integer(minimum: 0, maximum: 16_777_216, optional: false), .integer(0), advanced: true),
            flag("pre_buffer", "PyArrow pre-buffer reads", true),
            Field("pyarrow_options.coerce_int96_timestamp_unit", "INT96 timestamp unit", .choice(["", "ns", "us", "ms"]), .string(""), advanced: true),
            Field("pyarrow_options.thrift_string_size_limit", "Metadata string byte limit", .integer(minimum: 1, maximum: 16_777_216, optional: true), .null, advanced: true),
            Field("pyarrow_options.thrift_container_size_limit", "Metadata container limit", .integer(minimum: 1, maximum: 1_000_000, optional: true), .null, advanced: true),
            Field("pyarrow_options.schema_depth_limit", "Metadata schema depth limit", .integer(minimum: 1, maximum: 64, optional: true), .null, advanced: true),
            flag("page_checksum_verification", "Verify Parquet page checksums", false)]
    }
    private static func excelFields(engine: String, namedTable: Bool, hasHeader: Bool) -> [Field] {
        let count: FieldKind = .integer(minimum: 0, maximum: 2_147_483_647, optional: true)
        let text: FieldKind = .text(maximum: 500, multiline: false)
        func flag(_ key: String, _ title: String, _ value: Bool) -> Field { Field(key, title, .toggle, .bool(value), advanced: true) }
        if engine == "openpyxl" {
            return [flag("engine_options.read_only", "Workbook read-only mode", true), flag("engine_options.rich_text", "Preserve rich text while reading", false)]
        }
        if engine == "calamine" {
            return [Field("read_options.header_row", "Header row · starts at 0", count, namedTable || !hasHeader ? .null : .integer(0), advanced: true),
                Field("read_options.column_names", "Reader column names", .stringList, .array([]), advanced: true),
                Field("read_options.skip_rows", "Worksheet rows to skip", .integerOrList, .integer(0), advanced: true),
                Field("read_options.n_rows", "Worksheet rows to read", count, .null, advanced: true),
                Field("read_options.dtype_coercion", "Reader type conflicts", .choice(["coerce", "strict"]), .string("coerce"), advanced: true),
                Field("read_options.use_columns", "Reader columns / Excel range", .columnsOrRange, .array([]), advanced: true),
                Field("read_options.dtypes", "Reader types", .choiceOrMap(["int", "float", "string", "boolean"]), .null, advanced: true),
                flag("read_options.skip_whitespace_tail_rows", "Skip trailing whitespace rows", false),
                flag("read_options.whitespace_as_null", "Read whitespace-only cells as null", false)]
        }
        if engine == "xlsx2csv" {
            return [Field("engine_options.quoting", "Converted CSV quoting", .integerChoice([0, 1, 2, 3]), .integer(0), advanced: true), Field("engine_options.dateformat", "Date format", text, .null, advanced: true),
                Field("engine_options.timeformat", "Time format", text, .null, advanced: true),
                Field("engine_options.floatformat", "Float format", text, .string("%f"), advanced: true),
                flag("engine_options.scifloat", "Scientific float formatting", false),
                flag("engine_options.skip_empty_lines", "Skip empty worksheet lines", false),
                flag("engine_options.skip_trailing_columns", "Skip trailing empty columns", false),
                flag("engine_options.escape_strings", "Escape strings", false),
                flag("engine_options.no_line_breaks", "Remove cell line breaks", false),
                flag("engine_options.hyperlinks", "Include hyperlink addresses", false),
                flag("engine_options.exclude_hidden_sheets", "Exclude hidden worksheets", false),
                flag("engine_options.merge_cells", "Expand merged cells", false),
                Field("engine_options.ignore_formats", "Ignore number formats", .stringList, .array([]), advanced: true),
                flag("engine_options.skip_hidden_rows", "Skip hidden worksheet rows", false),
                flag("engine_options.ignore_invalid_char_data", "Ignore invalid XML character data", false),
                Field("read_options.quote_char", "Converted CSV quote character", text, .string("\""), advanced: true),
                Field("read_options.comment_prefix", "Converted CSV comment prefix", text, .null, advanced: true),
                Field("read_options.skip_rows", "Converted records to skip", .integer(minimum: 0, maximum: 100_000, optional: false), .integer(0), advanced: true),
                Field("read_options.skip_lines", "Converted physical lines to skip", .integer(minimum: 0, maximum: 100_000, optional: false), .integer(0), advanced: true),
                Field("read_options.skip_rows_after_header", "Converted records to skip after header", .integer(minimum: 0, maximum: 100_000, optional: false), .integer(0), advanced: true),
                Field("read_options.null_values", "Converted null markers", .listOrMap, .array([]), advanced: true),
                flag("read_options.empty_string_is_null", "Read empty converted fields as null", true),
                flag("read_options.infer_schema", "Infer converted field types", true),
                flag("read_options.ignore_errors", "Replace converted type errors with null", false),
                Field("read_options.n_rows", "Converted records to read", count, .null, advanced: true),
                Field("read_options.encoding", "Converted text decoding", .choice(["utf8", "utf8-lossy"]), .string("utf8"), advanced: true),
                flag("read_options.low_memory", "Prefer lower-memory converted reading", false),
                flag("read_options.try_parse_dates", "Try parsing converted dates", false),
                Field("read_options.row_index_name", "Converted row index name", text, .null, advanced: true),
                Field("read_options.row_index_offset", "Converted row index offset", .integer(minimum: 0, maximum: 2_147_483_647, optional: false), .integer(0), advanced: true),
                Field("read_options.new_columns", "Converted column names", .stringList, .array([]), advanced: true),
                flag("read_options.decimal_comma", "Converted decimal comma", false),
                flag("read_options.truncate_ragged_lines", "Truncate extra converted fields", false)]
        }
        return []
    }
    public static func fields(for format: Format, options: [String: JSONValue] = [:]) -> [Field] {
        let text: FieldKind = .text(maximum: 500, multiline: false)
        let optionalCount: FieldKind = .integer(minimum: 1, maximum: 2_147_483_647, optional: true)
        let offset: FieldKind = .integer(minimum: 0, maximum: 2_147_483_647, optional: false)
        let projection = Field("columns", "Read columns", .columns, .array([]), help: "Empty reads every column. Selecting columns is performed by this source reader.")
        let rowLimit = Field("n_rows", "Rows to read", optionalCount, .null, advanced: true, help: "Empty reads all source rows within the resource budget. This is a source read limit, not the preview sample size.")
        let rowName = Field("row_index_name", "Row index column name", text, .null, advanced: true, help: "Empty does not add an index column.")
        let rowOffset = Field("row_index_offset", "Row index offset", offset, .integer(0), advanced: true)
        let schema = Field("schema", "Declared columns and types", .schema, .object([:]), advanced: true)
        let overrides = Field("schema_overrides", "Column type overrides", .schema, .object([:]), advanced: true)
        let inference = Field("infer_schema_length", "Records for type discovery", optionalCount, .integer(100), advanced: format == .excel, help: "Empty examines all records for type discovery. Later incompatible values still fail unless the reader's error policy is explicitly changed.")
        switch format {
        case .csv:
            return [Field("columns", "Read columns", .columns, .array([]), advanced: true),
                Field("schema", "Declared columns and types", .schema, .object([:]), advanced: true),
                Field("schemaOverrides", "Column type overrides", .schema, .object([:]), advanced: true),
                Field("inferSchema", "Infer field types", .toggle, .bool(false), advanced: true, help: "Off preserves lexical text unless an explicit column type is supplied."),
                Field("inferSchemaLength", "Records for type discovery", .integer(minimum: 0, maximum: 5_000_000, optional: true), .integer(100), advanced: true),
                Field("nRows", "Rows to read", .integer(minimum: 0, maximum: 5_000_000, optional: true), .null, advanced: true),
                Field("lowMemory", "Prefer lower memory", .toggle, .bool(true), advanced: true),
                Field("tryParseDates", "Try parsing dates", .toggle, .bool(false), advanced: true, help: "Date/time outputs currently require an explicit supported scalar representation; unsupported temporal columns are reported."),
                Field("ignoreErrors", "Replace type-conflict values with null", .toggle, .bool(false), advanced: true),
                Field("decimalComma", "Comma decimal mark", .toggle, .bool(false), advanced: true),
                Field("newColumns", "Reader column names", .stringList, .array([]), advanced: true),
                Field("rowIndexName", "Row index column name", text, .string(""), advanced: true),
                Field("rowIndexOffset", "Row index offset", .integer(minimum: 0, maximum: 4_294_967_295, optional: false), .integer(0), advanced: true),
                Field("eolChar", "Record terminator", .character, .string("\n"), advanced: true),
                Field("extraColumns", "Extra fields", .choice(["raise", "ignore"]), .string("raise"), advanced: true),
                Field("missingColumns", "Missing declared columns", .choice(["raise", "insert"]), .string("raise"), advanced: true, help: "Insert missing fields using the empty-field policy: empty text or null. Declared column types still apply."),
                Field("raiseIfEmpty", "Report an empty source", .toggle, .bool(true), advanced: true),
                Field("columnNullValues", "Per-column null markers", .nullMarkerMap, .object([:]), advanced: true, help: "Use either per-column markers or the global null-marker list.")]

        case .parquet:
            return [projection,
                Field("parallel", "Read strategy", .choice(["auto", "columns", "row_groups", "none", "prefiltered"]), .string("auto")),
                Field("low_memory", "Prefer lower memory", .toggle, .bool(false)), rowLimit, rowName, rowOffset,
                Field("use_statistics", "Use stored statistics", .toggle, .bool(true), advanced: true), schema,
                Field("missing_columns", "Missing declared columns", .choice(["raise", "insert"]), .string("raise"), advanced: true),
                Field("extra_columns", "Extra stored columns", .choice(["raise", "ignore"]), .string("raise"), advanced: true),
                Field("cast_options.integer_cast", "Integer reader casts", .choice(["forbid", "upcast", "allow-float"]), .string("forbid"), advanced: true),
                Field("cast_options.float_cast", "Float reader casts", .floatCast, .string("forbid"), advanced: true),
                Field("cast_options.categorical_to_string", "Categorical values as text", .choice(["forbid", "allow"]), .string("forbid"), advanced: true),
                Field("use_pyarrow", "Use PyArrow reader", .toggle, .bool(false), advanced: true),
                Field("memory_map", "Memory-map PyArrow input", .toggle, .bool(true), advanced: true)] + (options["use_pyarrow"] == .bool(true) ? pyarrowFields() : [])
        case .json: return [schema, overrides, inference]
        case .jsonl:
            return [schema, overrides, inference,
                Field("batch_size", "Batch rows", .integer(minimum: 1, maximum: 2_147_483_647, optional: false), .integer(1024), advanced: true), rowLimit,
                Field("low_memory", "Prefer lower memory", .toggle, .bool(false), advanced: true), rowName, rowOffset,
                Field("ignore_errors", "Replace type-conflict values with null", .toggle, .bool(false), advanced: true)]
        case .excel:
            let engine: String = { if case .string(let name) = options["engine"] { return name }; return "calamine" }()
            return [Field("sheet_id", "Worksheet number", optionalCount, .integer(1), help: "Read one worksheet. Clear this when selecting a worksheet or named table by name."),
                Field("sheet_name", "Worksheet name", text, .null), Field("table_name", "Named table", text, .null),
                Field("has_header", "First row contains column names", .toggle, .bool(true)), projection,
                Field("drop_empty_rows", "Drop empty rows", .toggle, .bool(true)), Field("drop_empty_cols", "Drop empty columns", .toggle, .bool(true)),
                Field("engine", "Workbook reader", .choice(["calamine", "openpyxl", "xlsx2csv"]), .string("calamine"), advanced: true),
                overrides, inference, Field("raise_if_empty", "Report an empty worksheet", .toggle, .bool(true), advanced: true)] + excelFields(engine: engine, namedTable: { if case .string(let table) = options["table_name"] { return !table.isEmpty }; return false }(), hasHeader: options["has_header"] != .bool(false))
        case .markdown:
            return [Field("table_index", "Table number · starts at 0", offset, .integer(0)),
                Field("has_header", "First row contains column names", .toggle, .bool(true)),
                Field("cell_mode", "Cell representation", .choice(["text", "markdown"]), .string("text")),
                Field("link_mode", "Links", .choice(["label", "url", "both"]), .string("label")),
                Field("line_breaks", "Line breaks", .choice(["newline", "space"]), .string("newline")),
                projection, Field("encoding", "Encoding", .choice(["utf-8", "utf-8-sig", "utf-16", "cp1252"]), .string("utf-8"), advanced: true), rowLimit, overrides]
        case .xml:
            return [Field("row_xpath", "Record path", text, .string("./row")),
                Field("fields", "Column paths", .stringMap, .object([:])),
                Field("include_attributes", "Include record attributes", .toggle, .bool(false)),
                Field("attribute_prefix", "Attribute column prefix", text, .string("@"), advanced: true),
                Field("namespaces", "Namespace prefixes", .stringMap, .object([:]), advanced: true),
                Field("encoding", "Encoding", .choice(["utf-8", "utf-8-sig", "utf-16", "cp1252"]), .string("utf-8"), advanced: true), rowLimit, overrides]
        case .html:
            return [Field("table_index", "Table number · starts at 0", offset, .integer(0)),
                Field("header", "Header row · empty generates names", .integer(minimum: 0, maximum: 100_000, optional: true), .integer(0)),
                Field("cell_mode", "Cell representation", .choice(["text", "html"]), .string("text")),
                Field("link_mode", "Links", .choice(["label", "url", "both"]), .string("label")),
                Field("include_hidden", "Include stored hidden rows", .toggle, .bool(false)), projection,
                Field("span_policy", "Merged cells", .choice(["reject", "expand"]), .string("reject"), advanced: true),
                Field("skip_rows", "Rows to skip", .integer(minimum: 0, maximum: 100_000, optional: false), .integer(0), advanced: true),
                Field("encoding", "Encoding", .choice(["utf-8", "utf-8-sig", "utf-16", "cp1252"]), .string("utf-8"), advanced: true), rowLimit, overrides]
        case .api: return []
        case .database:
            return [Field("backend", "Database type", .choice(["sqlite", "postgresql", "mysql"]), .string("sqlite")),
                Field("host", "Host", text, .string("localhost")),
                Field("port", "Port", .integer(minimum: 1, maximum: 65_535, optional: false), .integer(5432)),
                Field("database", "Database", text, .string("")), Field("schema", "Schema", text, .string("main")), Field("table", "Table", text, .string("")),
                Field("columns", "Read columns", .stringList, .array([])),
                Field("orderBy", "Order by · +column ascending / -column descending", .stringList, .array([])),
                Field("rowLimit", "Rows to read · 0 reads all", .integer(minimum: 0, maximum: 5_000_000, optional: false), .integer(0), advanced: true),
                Field("rowOffset", "Rows to skip", offset, .integer(0), advanced: true),
                Field("fetchBatchRows", "Fetch batch rows", .integer(minimum: 1, maximum: 8192, optional: false), .integer(2048), advanced: true),
                Field("timeoutSeconds", "Request timeout · seconds", .integer(minimum: 1, maximum: 300, optional: false), .integer(30), advanced: true),
                Field("tlsMode", "Transport security", .choice(["verifyFull", "disabledLoopback"]), .string("verifyFull"), advanced: true),
                Field("decimalMode", "Decimal values", .choice(["text", "error"]), .string("text"), advanced: true),
                Field("dateTimeMode", "Date and time values", .choice(["iso8601", "error"]), .string("iso8601"), advanced: true),
                Field("binaryMode", "Binary values", .choice(["error", "base64", "hex"]), .string("error"), advanced: true)]
        }
    }
}
