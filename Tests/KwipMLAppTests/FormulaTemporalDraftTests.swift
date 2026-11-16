import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FormulaTemporalDraftTests: XCTestCase {
    func testGuidedTemporalRoundTripsEveryOperation() throws {
        for kind in FormulaGuidedTemporal.kinds {
            var guided = FormulaGuidedTemporal(kind: kind, column: "appointment")
            guided.otherColumn = "other"
            guided.pattern = "%Y-%m-%d %H:%M:%S"
            guided.timezone = "America/New_York"
            let decoded = try XCTUnwrap(FormulaGuidedTemporal(expression: guided.expression))
            XCTAssertEqual(decoded.expression, guided.expression)
            XCTAssertEqual(decoded.outputType, guided.outputType)
        }
    }
    func testExplicitPoliciesAndOutputTypes() throws {
        var parse = FormulaGuidedTemporal(kind: "temporal_parse", column: "text")
        parse.dtype = "date"; parse.invalid = "missing"; parse.ambiguous = "latest"
        let decoded = try XCTUnwrap(FormulaGuidedTemporal(expression: parse.expression))
        XCTAssertEqual(decoded.dtype, "date")
        XCTAssertEqual(decoded.invalid, "missing")
        XCTAssertEqual(decoded.ambiguous, "latest")
        XCTAssertEqual(decoded.outputType, "date")
        var difference = FormulaGuidedTemporal(kind: "temporal_difference", column: "a")
        difference.otherColumn = "b"
        XCTAssertEqual(difference.outputType, "float64")
        difference.representation = "integer"
        XCTAssertEqual(difference.outputType, "int64")
    }
    func testMalformedOrNestedCallsStayExpression() {
        for expression in ["temporal_parse([text], \"datetime\")", "temporal_timezone([a], \"UTC\") + 1", "temporal_difference([a], [b], 1, \"integer\")", "temporal_format(lower([text]), \"iso\", \"UTC\")"] {
            XCTAssertNil(FormulaGuidedTemporal(expression: expression))
        }
    }
    func testOrderedTemporalSchemaMetadataAndKnownColumnReplacement() throws {
        var draft = try FormulaDraft(configuration: FormulaDraft.defaults)
        var parse = FormulaGuidedTemporal(kind: "temporal_parse", column: "appointment")
        parse.timezone = "America/New_York"
        draft.rows[0].temporal = parse
        draft.rows[0].outputs = [.init(name: "instant", dtype: "datetime")]
        let incoming = [DataColumnSchema(name: "appointment", dtype: "string", nullable: true)]
        let schema = try draft.outputSchema(incoming)
        XCTAssertEqual(schema.last?.temporal, ["version": .integer(1), "unit": .string("microsecond"), "timezone": .string("America/New_York")])
        draft.rows[0].outputs[0].write = "replace"
        draft.rows[0].outputs[0].name = "appointment"
        XCTAssertEqual(try draft.outputSchema(incoming).count, 1)
        draft.rows[0].outputs[0].name = "missing"
        XCTAssertThrowsError(try draft.outputSchema(incoming))
    }
    func testNativeTemporalSourceTypesAreValidated() throws {
        var draft = try FormulaDraft(configuration: FormulaDraft.defaults)
        draft.rows[0].temporal = FormulaGuidedTemporal(kind: "temporal_parse", column: "number")
        draft.rows[0].outputs = [.init(name: "instant", dtype: "datetime")]
        XCTAssertThrowsError(try draft.outputSchema([.init(name: "number", dtype: "int64", nullable: false)]))
    }

    func testNestedTemporalExpressionSchemaIsInferredWithoutTimezoneGuessing() throws {
        let parse = FormulaGuidedTemporal(kind: "temporal_parse", column: "text").expression
        let expression = "coalesce(temporal_timezone(" + parse + ", \"America/New_York\"), temporal_timezone(" + parse + ", \"America/New_York\"))"
        let metadata = try FormulaGuidedTemporal.metadata(for: expression, outputType: "datetime", schema: [.init(name: "text", dtype: "string", nullable: true)])
        XCTAssertEqual(metadata["timezone"], .string("America/New_York"))
        XCTAssertThrowsError(try FormulaGuidedTemporal.metadata(for: "coalesce(" + parse + ", temporal_timezone(" + parse + ", \"America/New_York\"))", outputType: "datetime", schema: []))
    }

}
