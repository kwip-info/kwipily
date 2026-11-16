import XCTest
@testable import KwipMLCore

final class CSVParsingConfigurationTests: XCTestCase {
    func testRichSourceConfigurationRoundTripsWithoutPrivatePaths() throws {
        let source = WorkflowNode(address: .init(domain: .data, localID: "source"), payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults))
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [source])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
        var forbidden = document
        forbidden.nodes[0].payload = .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["path": .string("/private/file.csv")])
        XCTAssertThrowsError(try WorkflowCodec.encode(forbidden))
    }
    func testCustomEscapeRoundTripAndCollisions() throws {
        let config = try CSVParsingConfiguration.validate(["escapeMode": .string("custom"), "escapeChar": .string("~")])
        XCTAssertEqual(config["escapeChar"], .string("~"))
        XCTAssertEqual(try JSONDecoder().decode([String: JSONValue].self, from: Data(CSVParsingConfiguration.transportJSON(config).utf8)), config)
        for value in ["", "xx", "\n", ",", "\""] {
            XCTAssertThrowsError(try CSVParsingConfiguration.validate(["escapeMode": .string("custom"), "escapeChar": .string(value)]))
        }
    }
    func testLegacyDefaultsPreserveMissingValuesAndStrictBlankRecords() throws {
        let config = try CSVParsingConfiguration.validate(["separator": .string(";")])
        XCTAssertEqual(config["emptyField"], .string("null"))
        XCTAssertEqual(config["skipBlankRecords"], .bool(false))
        XCTAssertEqual(config["separator"], .string(";"))
    }
    func testCustomDelimiterAndRoundTrip() throws {
        let config: [String: JSONValue] = ["separator": .string(":"), "quoteChar": .string("'"), "encoding": .string("cp1252"), "headerMode": .string("generated"), "nullMarkers": .array([.string("NA")])]
        let data = Data(try CSVParsingConfiguration.transportJSON(config).utf8)
        let decoded = try JSONDecoder().decode([String: JSONValue].self, from: data)
        XCTAssertEqual(decoded, try CSVParsingConfiguration.validate(config))
    }
    func testInvalidOrAmbiguousDialectsAreRejected() {
        for config: [String: JSONValue] in [["separator": .string("||")], ["separator": .string("\n")], ["separator": .string("\"")], ["quoteChar": .string("xx")], ["skipRecords": .integer(-1)], ["encoding": .string("unknown")], ["extra": .string("ignored")], ["nullMarkers": .array([.string("")])]] {
            XCTAssertThrowsError(try CSVParsingConfiguration.validate(config))
        }
    }
}
