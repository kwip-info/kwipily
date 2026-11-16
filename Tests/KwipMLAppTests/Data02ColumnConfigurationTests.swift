import XCTest
import KwipMLCore
@testable import KwipMLApp

final class Data02ColumnConfigurationTests: XCTestCase {
    func testSelectionPreservesExactNamesOrderAndSimultaneousSwap() throws {
        var draft = DataColumnConfigurationDraft(configuration: ["columns": .array([.string(" a"), .string("b")]), "rename": .object([:])], mode: .selection)
        draft.setOutputName("b", for: " a")
        draft.setOutputName(" a", for: "b")
        draft.move("b", by: -1)
        let result = try draft.validatedConfiguration(schema: [.init(name:" a",dtype:"string"), .init(name:"b",dtype:"int64")])
        XCTAssertEqual(result["columns"], .array([.string("b"), .string(" a")]))
        XCTAssertEqual(result["rename"], .object([" a":.string("b"), "b":.string(" a")]))
    }
    func testOmittedColumnsRemainAvailableAndMissingSelectionsStayVisible() {
        var draft = DataColumnConfigurationDraft(configuration:["columns":.array([.string("gone"), .string("a")]),"rename":.object([:])],mode:.selection)
        draft.setIncluded(false, for:"a")
        let rows = draft.rows(schema:[.init(name:"a",dtype:"string"),.init(name:"b",dtype:"bool")])
        XCTAssertEqual(rows.map(\.name),["gone","a","b"])
        XCTAssertTrue(rows[0].isMissing)
        XCTAssertFalse(rows[1].included)
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:[.init(name:"a",dtype:"string")]))
    }
    func testCollisionAndEmptyOutputAreRejectedWithoutMutatingDraft() {
        var draft=DataColumnConfigurationDraft(configuration:["columns":.array([.string("a"),.string("b")]),"rename":.object([:])],mode:.selection)
        draft.setOutputName("a",for:"b")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:[]))
        XCTAssertEqual(draft.outputName(for:"b"),"a")
        draft.setOutputName("  ",for:"b")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:[]))
    }
    func testCastSharesRowsAndRejectsInvalidTypeWithoutLosingOtherSettings() throws {
        var draft=DataColumnConfigurationDraft(configuration:["columns":.object(["a":.string("float64")]),"missingPolicy":.string("preserve")],mode:.cast)
        draft.setIncluded(true,for:"b")
        draft.setTargetType("bool",for:"b")
        let result=try draft.validatedConfiguration(schema:[.init(name:"a",dtype:"string"),.init(name:"b",dtype:"string")])
        XCTAssertEqual(result["missingPolicy"],.string("preserve"))
        XCTAssertEqual(result["columns"],.object(["a":.string("float64"),"b":.string("bool")]))
        draft.setTargetType("madeUp",for:"b")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema:[]))
    }
    func testDraftReloadIsExplicitAndConfigurationRemainsAtomic() throws {
        let original:[String:JSONValue]=["columns":.array([.string("a")]),"rename":.object([:])]
        var draft=DataColumnConfigurationDraft(configuration:original,mode:.selection)
        draft.setOutputName("new",for:"a")
        XCTAssertNotEqual(draft.configuration,original)
        XCTAssertEqual(original["rename"],.object([:]))
        draft=DataColumnConfigurationDraft(configuration:original,mode:.selection)
        XCTAssertEqual(draft.configuration,original)
    }
}
