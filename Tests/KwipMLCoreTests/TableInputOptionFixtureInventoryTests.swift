import XCTest
@testable import KwipMLCore

/// Freezes the user-visible option denominator, including conditional engine panels.
/// Python's fixture coverage guard separately requires an executed test for every ID.
final class TableInputOptionFixtureInventoryTests: XCTestCase {
    func testEveryExposedReaderOptionHasAReviewedFixtureInventoryEntry() throws {
        var contexts: [String: JSONValue] = [:]
        contexts["csv.compatibility"] = .array(CSVParsingConfiguration.legacyDefaults.keys.sorted().map(JSONValue.string))
        // These two markers are fixed compatibility transport, never form controls.
        contexts["csv.polars"] = .array(CSVParsingConfiguration.newDefaults.keys.filter { !["parserProfile", "usePyarrow"].contains($0) }.sorted().map(JSONValue.string))
        for format in TableInputConfiguration.Format.allCases where format != .csv {
            let variants: [(String, [String: JSONValue])]
            switch format {
            case .excel: variants = ["calamine", "openpyxl", "xlsx2csv"].map { ("excel." + $0, ["engine": .string($0)]) }
            case .parquet: variants = [("parquet.native", [:]), ("parquet.pyarrow", ["use_pyarrow": .bool(true)])]
            case .database: variants = ["sqlite", "postgresql", "mysql"].map { ("database." + $0, ["backend": .string($0)]) }
            default: variants = [(format.rawValue, [:])]
            }
            for (name, options) in variants {
                var fields = TableInputConfiguration.fields(for: format, options: options)
                // Match the actual conditional surfaces in TableInputFormatFields.
                if format == .excel, options["engine"] != .string("calamine") { fields.removeAll { $0.key == "table_name" } }
                if format == .parquet, options["use_pyarrow"] == .bool(true) {
                    fields.removeAll { ["parallel", "use_statistics", "low_memory", "missing_columns", "extra_columns", "schema"].contains($0.key) || $0.key.hasPrefix("cast_options.") }
                } else if format == .parquet { fields.removeAll { $0.key == "memory_map" } }
                if format == .database, options["backend"] == .string("sqlite") { fields.removeAll { ["host", "port", "database", "tlsMode"].contains($0.key) } }
                XCTAssertEqual(Set(fields.map(\.key)).count, fields.count, name)
                contexts[name] = .array(fields.map(\.key).sorted().map(JSONValue.string))
            }
        }
        let inventory: JSONValue = .object(["schemaVersion": .integer(1), "contexts": .object(contexts)])
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let path = root.appendingPathComponent("runner/tests/fixtures/table_options/inventory.json")
        if ProcessInfo.processInfo.environment["KWIPML_WRITE_TABLE_OPTION_INVENTORY"] == "1" {
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            try encoder.encode(inventory).write(to: path)
        }
        XCTAssertEqual(try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: path)), inventory,
                       "An exposed option changed: add meaningful fixture evidence and review the inventory before regenerating it.")
    }
}
