import XCTest
@testable import KwipMLCore

final class PolarsCSVConfigurationTests: XCTestCase {
    func testNewSourcesUseExplicitNativeProfileWithoutLegacySwitches() throws {
        let config = try CSVParsingConfiguration.validate(CSVParsingConfiguration.newDefaults)
        XCTAssertEqual(config["parserProfile"], .string("polars"))
        XCTAssertNil(config["escapeMode"])
        XCTAssertNil(config["recordEndings"])
        XCTAssertEqual(config["skipLines"], .integer(0))
        XCTAssertEqual(config["skipRowsAfterHeader"], .integer(0))
        XCTAssertEqual(config["commentPrefix"], .string(""))
        XCTAssertEqual(config["emptyField"], .string("text"))
    }
    func testNativeProfileRejectsOptionsThatWouldOtherwiseBeSilentlyIgnored() {
        for option: [String: JSONValue] in [
            ["escapeMode": .string("custom")], ["skipInitialSpace": .bool(true)],
            ["allowQuotedNewlines": .bool(false)], ["recordEndings": .string("cr")],
            ["separator": .string("é")], ["quoteChar": .string("é")],
            ["encoding": .string("cp1252")], ["skipLines": .integer(-1)],
            ["skipRowsAfterHeader": .string("2")], ["commentPrefix": .string("\n")]
        ] {
            XCTAssertThrowsError(try CSVParsingConfiguration.validate(["parserProfile": .string("polars")].merging(option) { _, value in value }))
        }
    }
    func testNativeOptionsRoundTripWithExplicitSkipsAndComments() throws {
        let config = try CSVParsingConfiguration.validate([
            "parserProfile": .string("polars"), "separator": .string("\t"), "quoteChar": .string(""),
            "encoding": .string("utf8-lossy"), "skipRecords": .integer(0), "skipLines": .integer(3),
            "skipRowsAfterHeader": .integer(4), "commentPrefix": .string("##")
        ])
        let encoded = try CSVParsingConfiguration.transportJSON(config)
        XCTAssertEqual(try JSONDecoder().decode([String: JSONValue].self, from: Data(encoded.utf8)), config)
        let node = WorkflowNode(address: .init(domain: .data, localID: "source"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: config))
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [node])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
    }
    func testExplicitMigrationKeepsCompatibleValuesAndResetsUnsupportedDialect() throws {
        let legacy: [String: JSONValue] = ["separator": .string(";"), "encoding": .string("cp1252"), "escapeMode": .string("custom"), "escapeChar": .string("~"), "skipRecords": .integer(3), "nullMarkers": .array([.string("NA")])]
        let migrated = CSVParsingConfiguration.polarsMigration(legacy)
        XCTAssertEqual(migrated["separator"], .string(";"))
        XCTAssertEqual(migrated["skipRecords"], .integer(3))
        XCTAssertEqual(migrated["nullMarkers"], .array([.string("NA")]))
        XCTAssertEqual(migrated["encoding"], .string("utf-8-sig"))
        XCTAssertNil(migrated["escapeMode"])
        XCTAssertEqual(try CSVParsingConfiguration.validate(migrated), migrated)
        XCTAssertEqual(legacy["encoding"], .string("cp1252"))
        let swapped = CSVParsingConfiguration.polarsMigration(["separator": .string(";"), "quoteChar": .string(",")])
        XCTAssertEqual(swapped["quoteChar"], .string(","))
        var carried = migrated; carried["skipBlankRecords"] = .bool(false)
        XCTAssertNil(try CSVParsingConfiguration.validate(carried)["skipBlankRecords"])
        carried["skipBlankRecords"] = .string("false")
        XCTAssertThrowsError(try CSVParsingConfiguration.validate(carried))
    }
    func testUnprofiledSavedConfigurationRetainsLegacySemantics() throws {
        let config = try CSVParsingConfiguration.validate(["separator": .string("é"), "encoding": .string("cp1252"), "escapeMode": .string("custom"), "escapeChar": .string("~")])
        XCTAssertNil(config["parserProfile"])
        XCTAssertEqual(config["skipBlankRecords"], .bool(false))
        XCTAssertEqual(config["emptyField"], .string("null"))
        XCTAssertEqual(config["escapeMode"], .string("custom"))
        XCTAssertThrowsError(try CSVParsingConfiguration.validate(["parserProfile": .string("unknown")]))
    }
    func testPhysicalAndRecordSkipsCannotBeCombined() {
        var config = CSVParsingConfiguration.newDefaults
        config["skipRecords"] = .integer(1); config["skipLines"] = .integer(1)
        XCTAssertThrowsError(try CSVParsingConfiguration.validate(config))
    }
}
