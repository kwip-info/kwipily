import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FilterRowsConfigurationTests: XCTestCase {
    private let schema = [DataColumnSchema(name:"score",dtype:"float64",nullable:true),DataColumnSchema(name:"status",dtype:"string",nullable:false),DataColumnSchema(name:"picture",dtype:"image",nullable:true)]
    func testImportedMembershipPreservesEmbeddedNewlinesAndRejectsMixedTypes() throws {
        var config = FilterRowsConfigurationDraft.defaults
        let row:[String:JSONValue] = ["column":.string("status"),"comparison":.string("in"),"value":.array([.string("two\nlines"),.string("ready")]),"caseSensitive":.bool(true)]
        config["conditions"] = .array([.object(row)])
        XCTAssertEqual(try FilterRowsConfigurationDraft(configuration:config).validatedConfiguration(schema:schema),config)
        var invalid=row;invalid["value"] = .array([.string("ready"),.integer(2)])
        config["conditions"] = .array([.object(invalid)])
        XCTAssertThrowsError(try FilterRowsConfigurationDraft(configuration:config))
    }
    func testTypedDraftRoundTripAndNoSilentCast() throws {
        var draft = try FilterRowsConfigurationDraft(configuration:FilterRowsConfigurationDraft.defaults)
        draft.conditions[0].column = "score"
        draft.conditions[0].valueText = "70"
        draft.conditions[0].valueType = "number"
        let config = try draft.validatedConfiguration(schema:schema)
        XCTAssertEqual(try FilterRowsConfigurationDraft(configuration:config).validatedConfiguration(schema:schema),config)
        draft.conditions[0].valueType = "text"
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:schema))
        draft.conditions[0].comparison = "isMissing"
        XCTAssertNoThrow(try draft.validatedConfiguration(schema:schema))
    }
    func testMembershipValuesRemainTypedAndImageOnlyAllowsMissing() throws {
        var draft = try FilterRowsConfigurationDraft(configuration:FilterRowsConfigurationDraft.defaults)
        draft.conditions[0].column = "status";draft.conditions[0].comparison = "in"
        draft.conditions[0].valueType = "text";draft.conditions[0].valueText = "active\npending"
        let config = try draft.validatedConfiguration(schema:schema)
        guard case .array(let rows) = config["conditions"],case .object(let row) = rows[0] else {return XCTFail("condition missing")}
        XCTAssertEqual(row["value"],.array([.string("active"),.string("pending")]))
        draft.conditions[0].column = "picture"
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:schema))
        draft.conditions[0].comparison = "isNotMissing"
        XCTAssertNoThrow(try draft.validatedConfiguration(schema:schema))
    }
    func testInactiveDraftPreservedButActiveModeValidated() throws {
        var draft = try FilterRowsConfigurationDraft(configuration:FilterRowsConfigurationDraft.defaults)
        draft.conditions = [];draft.mode = "expression";draft.expression = "[score] >= 70"
        XCTAssertNoThrow(try draft.validatedConfiguration(schema:schema))
        draft.mode = "simple"
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:schema))
        draft.mode = "pattern";draft.column = "status";draft.pattern = "^active$";draft.flags = ["ignoreCase","multiline","dotAll"]
        XCTAssertNoThrow(try draft.validatedConfiguration(schema:schema))
        draft.column = "score"
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:schema))
    }
}
