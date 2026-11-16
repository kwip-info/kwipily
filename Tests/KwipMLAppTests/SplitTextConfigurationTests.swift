import XCTest
@testable import KwipMLApp
import KwipMLCore

final class SplitTextConfigurationTests: XCTestCase {
    private let schema = [DataColumnSchema(name: "text", dtype: "string", nullable: true), DataColumnSchema(name: "id", dtype: "int64", nullable: false)]
    func testRoundTripAndColumnSchemaPreserveOriginals() throws {
        var draft = try SplitTextConfigurationDraft(configuration: SplitTextConfigurationDraft.defaults)
        draft.values["source"] = .string("text")
        draft.fields = [.init(name: "amount", type: "float64"), .init(name: "flag", type: "bool")]
        let config = try draft.validatedConfiguration(schema: schema)
        let reloaded = try SplitTextConfigurationDraft(configuration: config)
        XCTAssertEqual(reloaded.configuration, config)
        let output = try reloaded.outputSchema(schema)
        XCTAssertEqual(output.map(\.name), ["id", "amount", "flag"])
        XCTAssertEqual(output.map(\.dtype), ["int64", "float64", "bool"])
        XCTAssertEqual(output[0], schema[1])
    }
    func testRowsUsesOneOutputAndOrdinal() throws {
        var draft = try SplitTextConfigurationDraft(configuration: SplitTextConfigurationDraft.defaults)
        draft.values["source"] = .string("text")
        draft.values["outputMode"] = .string("rows")
        draft.values["ordinal"] = .bool(true)
        draft.fields.append(.init(name: "dormant"))
        XCTAssertEqual(try draft.outputSchema(schema).map(\.name), ["id", "part", "token_ordinal"])
        XCTAssertEqual(try draft.outputSchema(schema).last?.nullable, false)
    }
    func testFixedWidthOverlapsAndMulticharDelimiter() throws {
        var draft = try SplitTextConfigurationDraft(configuration: SplitTextConfigurationDraft.defaults)
        draft.values["source"] = .string("text")
        draft.values["delimiter"] = .string("::")
        XCTAssertNoThrow(try draft.validatedConfiguration(schema: schema))
        draft.values["mode"] = .string("fixedWidth")
        draft.values["measurement"] = .string("utf8")
        draft.fields = [.init(name: "a", start: 1, width: 4), .init(name: "b", start: 2, width: 4)]
        XCTAssertNoThrow(try draft.validatedConfiguration(schema: schema))
    }
    func testInvalidQuoteDelimiterAndEscaping() throws {
        var draft = try SplitTextConfigurationDraft(configuration: SplitTextConfigurationDraft.defaults)
        draft.values["source"] = .string("text")
        draft.values["delimiter"] = .string("\"|")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.values["delimiter"] = .string("::")
        draft.values["escaping"] = .string("escape")
        draft.values["escape"] = .string("\"")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
    }
    func testStaleSourceTypeCollisionsAndZeroStartRefused() throws {
        var draft = try SplitTextConfigurationDraft(configuration: SplitTextConfigurationDraft.defaults)
        draft.values["source"] = .string("gone")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.values["source"] = .string("id")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.values["source"] = .string("text")
        draft.fields[0].name = "id"
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.fields[0].name = "part"; draft.fields[0].start = 0
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
    }
    func testPracticeIsCleanUnscoredAndUsesActualResource() throws {
        XCTAssertFalse(SplitTextLessonContent.isAssessed)
        XCTAssertEqual(SplitTextPracticeExample.observationsResource, "split-text-observations")
        let doc = try SplitTextPracticeExample.document(projectID: "11111111-2222-3333-4444-555555555555")
        XCTAssertEqual(doc.nodes.count, 3)
        XCTAssertEqual(doc.edges.count, 2)
        XCTAssertEqual(doc.nodes.first { $0.address == SplitTextPracticeExample.splitAddress }?.payload, .dataOperator(operatorID: "kwip.data.splitText", config: SplitTextPracticeExample.configuration))
        XCTAssertNoThrow(try SplitTextConfigurationDraft(configuration: SplitTextPracticeExample.configuration))
    }
}
