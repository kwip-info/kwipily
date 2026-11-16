import XCTest
@testable import KwipMLCore

final class TableInputConfigurationTests: XCTestCase {
    func testLegacyCSVRemainsFlatAndPreservesItsParserProfile() throws {
        let old: [String: JSONValue] = ["separator": .string(";"), "encoding": .string("cp1252"), "escapeMode": .string("custom"), "escapeChar": .string("~")]
        XCTAssertEqual(try TableInputConfiguration.format(of: old), .csv)
        XCTAssertEqual(try TableInputConfiguration.validate(old), try CSVParsingConfiguration.validate(old))
        XCTAssertNil(try TableInputConfiguration.validate(old)["format"])
        XCTAssertEqual(TableInputConfiguration.replacingFormat(old, with: .csv), old)
        XCTAssertEqual(try TableInputConfiguration.options(in: old), old)
    }
    func testEveryFormatHasTypedReaderControlsAndValidDefaults() throws {
        for format in TableInputConfiguration.Format.allCases {
            let defaults = TableInputConfiguration.defaults(for: format)
            XCTAssertEqual(try TableInputConfiguration.format(of: defaults), format)
            if format == .api {XCTAssertThrowsError(try TableInputConfiguration.validate(defaults));continue}
            XCTAssertEqual(try TableInputConfiguration.validate(defaults), defaults)
            if format != .csv { XCTAssertFalse(TableInputConfiguration.fields(for: format).isEmpty, format.rawValue) }
        }
    }
    func testNewFormatsKeepExistingSourceOperatorIdentityAcrossProjectRoundTrip() throws {
        for format in TableInputConfiguration.Format.allCases where format != .api {
            let source = WorkflowNode(address: .init(domain: .data, localID: "source"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: TableInputConfiguration.defaults(for: format)))
            let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [source])
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
        }
    }
    func testJSONSchemaOverridesAreTypedAndUnknownReaderOptionsAreRejected() throws {
        let json: [String: JSONValue] = ["format": .string("json"), "options": .object(["schema_overrides": .object(["count": .string("Int64")]), "infer_schema_length": .null])]
        let result = try TableInputConfiguration.options(in: TableInputConfiguration.validate(json))
        XCTAssertEqual(result["schema_overrides"], .object(["count": .string("Int64")]))
        XCTAssertEqual(result["infer_schema_length"], .null)
        for bad: [String: JSONValue] in [["schema_overrides": .object(["amount": .string("Decimal")])], ["infer_schema_length": .integer(-1)], ["storage_options": .object(["token": .string("secret")])], ["path": .string("/private/source.json")]] {
            XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("json"), "options": .object(bad)]))
        }
    }
    func testParquetProjectionKeepsOrderedNamesOrIndexesAndRejectsMixtures() throws {
        for columns: JSONValue in [.array([.string("target"), .string("feature")]), .array([.integer(2), .integer(0)])] {
            let config = try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object(["columns": columns])])
            XCTAssertEqual(try TableInputConfiguration.options(in: config)["columns"], columns)
        }
        for columns: JSONValue in [.array([.string("a"), .integer(1)]), .array([.string("a"), .string("a")]), .array([.integer(-1)])] {
            XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object(["columns": columns])]))
        }
    }
    func testParquetScalarCastOptionsNormalizeAndRejectUnsupportedControls() throws {
        let config = try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object([
            "extra_columns": .string("ignore"), "cast_options": .object(["integer_cast": .string("upcast"), "float_cast": .array([.string("upcast"), .string("downcast")]), "categorical_to_string": .string("allow")])
        ])])
        let options = try TableInputConfiguration.options(in: config)
        XCTAssertEqual(TableInputConfiguration.value(for: "cast_options.float_cast", in: options), .array([.string("downcast"), .string("upcast")]))
        for options: [String: JSONValue] in [
            ["cast_options": .object(["float_cast": .array([.string("forbid"), .string("upcast")])])],
            ["cast_options": .object(["datetime_cast": .string("nanosecond-downcast")])],
            ["memory_map": .bool(false)],
            ["use_pyarrow": .bool(true), "extra_columns": .string("ignore")],
            ["use_pyarrow": .bool(true), "schema": .object(["value": .string("Int64")])],
            ["use_pyarrow": .bool(true), "cast_options": .object(["integer_cast": .string("upcast")])]
        ] { XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object(options)])) }
        XCTAssertNoThrow(try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object(["use_pyarrow": .bool(true), "memory_map": .bool(false)])]))
    }
    func testFormatSwitchIsDraftOnlyAndDoesNotCarryPrivateOrUnrelatedReaderOptions() throws {
        let csv: [String: JSONValue] = ["separator": .string(";"), "encoding": .string("cp1252")]
        let parquet = TableInputConfiguration.replacingFormat(csv, with: .parquet)
        XCTAssertEqual(parquet, TableInputConfiguration.defaults(for: .parquet))
        XCTAssertEqual(csv["separator"], .string(";"))
        XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object([:]), "bookmark": .string("private")]))
    }
    func testMarkupFieldMappingsAreOrdinaryTypedOptions() throws {
        let config = try TableInputConfiguration.validate(["format": .string("xml"), "options": .object(["row_xpath": .string("./m:row"), "namespaces": .object(["m": .string("urn:sample")]), "fields": .object(["id": .string("@id"), "value": .string("m:value")])])])
        XCTAssertEqual(try TableInputConfiguration.options(in: config)["fields"], .object(["id": .string("@id"), "value": .string("m:value")]))
        XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("xml"), "options": .object(["namespaces": .object(["": .string("urn:sample")])])]))
    }
    func testWorkbookEngineOptionsAreVisibleAndValidatedOnlyForTheirReader() throws {
        let calamine = try TableInputConfiguration.validate(["format": .string("excel"), "options": .object([
            "read_options": .object(["skip_rows": .array([.integer(1), .integer(3)]), "dtypes": .object(["count": .string("int")])])
        ])])
        XCTAssertEqual(TableInputConfiguration.value(for: "read_options.skip_rows", in: try TableInputConfiguration.options(in: calamine)), .array([.integer(1), .integer(3)]))
        let openpyxl = try TableInputConfiguration.validate(["format": .string("excel"), "options": .object(["engine": .string("openpyxl"), "engine_options": .object(["rich_text": .bool(true)])])])
        XCTAssertEqual(TableInputConfiguration.value(for: "engine_options.rich_text", in: try TableInputConfiguration.options(in: openpyxl)), .bool(true))
        for options: [String: JSONValue] in [
            ["engine": .string("openpyxl"), "read_options": .object(["skip_rows": .integer(1)])],
            ["engine_options": .object(["keep_vba": .bool(true)])],
            ["read_options": .object(["schema_sample_rows": .integer(1000)])],
            ["columns": .array([.string("a")]), "read_options": .object(["use_columns": .array([.string("b")])])]
        ] { XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("excel"), "options": .object(options)])) }
    }
    func testNamedWorkbookTableRetainsItsNativeHeaderByDefault() throws {
        let config = try TableInputConfiguration.validate(["format": .string("excel"), "options": .object(["table_name": .string("Measurements")])])
        XCTAssertEqual(TableInputConfiguration.value(for: "read_options.header_row", in: try TableInputConfiguration.options(in: config)), .null)
    }
    func testDatabaseReaderControlsNeverAcceptPrivateCredentialsOrSQL() throws {
        let options = try TableInputConfiguration.options(in: TableInputConfiguration.validate(["format": .string("database"), "options": .object(["backend": .string("postgresql"), "schema": .string("public"), "table": .string("examples"), "columns": .array([.string("feature"), .string("target")]), "orderBy": .array([.string("+feature")])])]))
        XCTAssertEqual(options["table"], .string("examples"))
        for key in ["password", "username", "sql", "connectionString", "certificatePath"] {
            XCTAssertThrowsError(try TableInputConfiguration.validate(["format": .string("database"), "options": .object([key: .string("private")])]))
        }
    }
    func testGenericSettingsTransportIsBoundedAndCSVRetainsItsSmallerLimit() throws {
        let config: [String: JSONValue] = ["format": .string("json"), "options": .object(["schema_overrides": .object(["count": .string("Int64")])])]
        XCTAssertEqual(try JSONDecoder().decode([String: JSONValue].self, from: Data(TableInputConfiguration.transportJSON(config).utf8)), try TableInputConfiguration.validate(config))
        let oversized = Dictionary(uniqueKeysWithValues: (0..<150).map { ("column_\($0)_" + String(repeating: "x", count: 300), JSONValue.string("String")) })
        XCTAssertThrowsError(try TableInputConfiguration.transportJSON(["format": .string("json"), "options": .object(["schema_overrides": .object(oversized)])]))
        XCTAssertEqual(try TableInputConfiguration.transportJSON(CSVParsingConfiguration.newDefaults), try CSVParsingConfiguration.transportJSON(CSVParsingConfiguration.newDefaults))
    }
    func testNativeReaderDefaultsAreAcceptedByThePythonReaderContract() throws {
        guard let python = ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {
            throw XCTSkip("Set KWIPML_BACKEND_TEST_PYTHON for the native/Python reader-option parity check.")
        }
        var configurations = [CSVParsingConfiguration.newDefaults] + TableInputConfiguration.Format.allCases.filter { ![.csv, .database, .api].contains($0) }.map(TableInputConfiguration.defaults)
        for engine in ["openpyxl", "xlsx2csv"] {
            configurations.append(try TableInputConfiguration.validate(["format": .string("excel"), "options": .object(["engine": .string(engine)])]))
        }
        configurations.append(try TableInputConfiguration.validate(["format": .string("parquet"), "options": .object(["use_pyarrow": .bool(true)])]))
        let process = Process(), input = Pipe(), output = Pipe(), errors = Pipe()
        process.executableURL = URL(fileURLWithPath: python)
        process.currentDirectoryURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        process.arguments = ["-c", "import json,sys; from runner.table_source_options import normalize_options; from runner.polars_csv_configuration import normalize_options as normalize_csv; configs=json.load(sys.stdin); [normalize_csv(config) if config.get('parserProfile')=='polars' else normalize_options(config) for config in configs]; print('validated',len(configs))"]
        process.standardInput = input; process.standardOutput = output; process.standardError = errors
        try process.run()
        try input.fileHandleForWriting.write(contentsOf: JSONEncoder().encode(configurations)); try input.fileHandleForWriting.close()
        process.waitUntilExit()
        let error = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, error)
        XCTAssertEqual(String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines), "validated \(configurations.count)")
    }
    func testNestedReaderFieldsPreserveStructuredOptionsWithoutArbitraryKeys() {
        var options: [String: JSONValue] = [:]
        TableInputConfiguration.set(.integer(2), for: "read_options.header_row", in: &options)
        XCTAssertEqual(options, ["read_options": .object(["header_row": .integer(2)])])
        XCTAssertEqual(TableInputConfiguration.value(for: "read_options.header_row", in: options), .integer(2))
        XCTAssertNil(TableInputConfiguration.value(for: "read_options.missing", in: options))
    }
}
