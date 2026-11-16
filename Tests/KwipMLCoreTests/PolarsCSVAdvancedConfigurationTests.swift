import XCTest
@testable import KwipMLCore

final class PolarsCSVAdvancedConfigurationTests: XCTestCase {
    func testNativeDefaultsPreserveLexicalTextUntilInferenceIsExplicit() throws {
        let defaults = try CSVParsingConfiguration.validate(CSVParsingConfiguration.newDefaults)
        XCTAssertEqual(defaults["inferSchema"], .bool(false))
        XCTAssertEqual(defaults["schema"], .object([:]))
        XCTAssertEqual(defaults["schemaOverrides"], .object([:]))
        XCTAssertEqual(defaults["columns"], .array([]))
        XCTAssertEqual(defaults["extraColumns"], .string("raise"))
        XCTAssertEqual(defaults["missingColumns"], .string("raise"))
        XCTAssertEqual(defaults["eolChar"], .string("\n"))
    }
    func testCSVReaderProjectionOverridesAndDecimalOptionsArePortable() throws {
        var options = CSVParsingConfiguration.newDefaults
        options["separator"] = .string(";")
        options["columns"] = .array([.string("amount"), .string("label")])
        options["schemaOverrides"] = .object(["amount": .string("Float64")])
        options["decimalComma"] = .bool(true)
        options["rowIndexName"] = .string("record")
        options["rowIndexOffset"] = .integer(10)
        options["nRows"] = .integer(500_001)
        XCTAssertEqual(try CSVParsingConfiguration.validate(options), options)
        XCTAssertEqual(try JSONDecoder().decode([String: JSONValue].self, from: Data(CSVParsingConfiguration.transportJSON(options).utf8)), options)
    }
    func testUnknownUnsafeAndUnsupportedTypedOptionsRemainExplicitErrors() {
        for option: [String: JSONValue] in [
            ["storage_options": .object([:])], ["credential_provider": .string("auto")],
            ["with_column_names": .string("callback")], ["glob": .bool(true)],
            ["schemaOverrides": .object(["date": .string("Date")])],
            ["nRows": .integer(-1)], ["rowIndexOffset": .integer(-1)],
            ["columns": .array([.string("a"), .integer(1)])]
        ] { XCTAssertThrowsError(try CSVParsingConfiguration.validate(["parserProfile": .string("polars")].merging(option) { _, value in value })) }
    }
    func testLegacyProfileStillRejectsNativeReaderKnobsInsteadOfChangingParsing() {
        XCTAssertThrowsError(try CSVParsingConfiguration.validate(["decimalComma": .bool(true)]))
        XCTAssertThrowsError(try CSVParsingConfiguration.validate(["inferSchema": .bool(true)]))
    }
}
