import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ColumnsGridDraftTests: XCTestCase {
    private let schema: [DataColumnSchema] = [.init(name: "a", dtype: "string", nullable: true), .init(name: "b", dtype: "int64", nullable: false), .init(name: "c", dtype: "string", nullable: false)]
    func testReconciledSavedRowsDoNotBecomeAnUnappliedDraft() throws {
        var saved = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        saved.add(source: "a")
        saved.update("a") { $0.outputName = "renamed" }
        saved.newColumns = "drop"
        let configuration = saved.configuration
        var displayed = try ColumnsConfigurationDraft(configuration: configuration)
        displayed.reconcile(schema: schema)
        XCTAssertFalse(displayed.hasUserChanges(comparedTo: configuration, schema: schema))
        displayed.update("b") { $0.included = true }
        XCTAssertTrue(displayed.hasUserChanges(comparedTo: configuration, schema: schema))
        displayed = try ColumnsConfigurationDraft(configuration: configuration)
        displayed.reconcile(schema: schema)
        XCTAssertFalse(displayed.hasUserChanges(comparedTo: configuration, schema: schema))
        XCTAssertEqual(configuration, saved.configuration, "Viewing or reverting must not persist implicit rows")
    }

    func testRequireGridChoicesStillOffersApplyWithoutBlockingSaveOnPresentation() throws {
        var saved = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        saved.add(source: "a")
        saved.newColumns = "error"
        let configuration = saved.configuration
        var displayed = try ColumnsConfigurationDraft(configuration: configuration)
        displayed.reconcile(schema: schema)
        XCTAssertFalse(displayed.hasUserChanges(comparedTo: configuration, schema: schema))
        XCTAssertTrue(displayed.requiresSchemaReview(comparedTo: configuration, schema: schema))
        XCTAssertFalse(displayed.requiresSchemaReview(comparedTo: displayed.configuration, schema: schema))
        saved.newColumns = "drop"
        XCTAssertFalse(displayed.requiresSchemaReview(comparedTo: saved.configuration, schema: schema))
    }
    func testGridKeepsDeselectionDraftsAndOrdersThroughArrowsAndDrop() throws {
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.reconcile(schema: schema)
        draft.update("a") { $0.outputName = "renamed"; $0.targetType = "float64"; $0.included = false }
        XCTAssertEqual(draft.keptCount, 2)
        draft.move("c", by: -1)
        XCTAssertEqual(draft.rows.map(\.source), ["a", "c", "b"])
        XCTAssertTrue(draft.move("b", before: "a"))
        XCTAssertEqual(draft.rows.map(\.source), ["b", "a", "c"])
        XCTAssertFalse(draft.move("outside", before: "a"))
        draft.move("b", to: 2)
        XCTAssertEqual(draft.rows.map(\.source), ["a", "c", "b"])
        draft.move("b", to: 0)
        draft.move("b", to: 99)
        XCTAssertEqual(draft.rows.map(\.source), ["b", "a", "c"])
        draft.update("a") { $0.included = true }
        XCTAssertEqual(draft.rows[1].outputName, "renamed")
        XCTAssertEqual(draft.rows[1].targetType, "float64")
        XCTAssertEqual(try draft.outputSchema(schema).map(\.name), ["b", "renamed", "c"])
    }
    func testSimultaneousNamesAndTypeNullabilityProjectExactly() throws {
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.reconcile(schema: schema)
        draft.update("a") { $0.outputName = "b"; $0.targetType = "float64" }
        draft.update("b") { $0.outputName = "a" }
        var output = try draft.outputSchema(schema)
        XCTAssertEqual(output.map(\.name), ["b", "a", "c"])
        XCTAssertEqual(output.map(\.dtype), ["float64", "int64", "string"])
        XCTAssertEqual(output[0].nullable, true)
        draft.missingPolicy = "error"
        output = try draft.outputSchema(schema)
        XCTAssertEqual(output[0].nullable, false)
        draft.castFailure = "null"
        XCTAssertEqual(try draft.outputSchema(schema)[0].nullable, true)
    }
    func testClearCannotCreateZeroColumnOutputAndSelectAllIsIndependentOfSearch() throws {
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.reconcile(schema: schema); draft.setAllIncluded(false)
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.setAllIncluded(true)
        XCTAssertEqual(try draft.outputSchema(schema).map(\.name), schema.map(\.name))
    }
    func testNewColumnsPoliciesAndCollisionWithPassthrough() throws {
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.add(source: "a")
        XCTAssertEqual(try draft.outputSchema(schema).map(\.name), ["a", "b", "c"])
        draft.update("a") { $0.outputName = "b" }
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.newColumns = "drop"
        XCTAssertEqual(try draft.outputSchema(schema).map(\.name), ["b"])
        draft.newColumns = "error"
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.reconcile(schema: schema)
        XCTAssertFalse(draft.rows[1].included)
        XCTAssertNoThrow(try draft.validatedConfiguration(schema: schema))
    }
    func testMissingExcludedColumnRequiresExplicitRemovalAndNeverRepairsNames() throws {
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.add(source: "gone"); draft.update("gone") { $0.included = false }
        draft.reconcile(schema: schema)
        XCTAssertEqual(draft.rows.first?.source, "gone")
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.removeMissing("gone", schema: schema)
        XCTAssertNoThrow(try draft.validatedConfiguration(schema: schema))
        draft.removeMissing("a", schema: schema)
        XCTAssertEqual(draft.rows.count, 3)
    }
    func testValidationRefusesMalformedShapeUnknownTypesAndBlankOrDuplicateNames() throws {
        var config = ColumnsConfigurationDraft.defaults
        config["surprise"] = .bool(true)
        XCTAssertThrowsError(try ColumnsConfigurationDraft(configuration: config))
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.reconcile(schema: schema)
        draft.update("a") { $0.targetType = "made-up" }
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.update("a") { $0.targetType = "keep"; $0.outputName = " " }
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
        draft.update("a") { $0.outputName = "b" }
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: schema))
    }
    func testSavedGridRoundTripAndLegacySourceDecodeRemainReadableButRetiredEditingRefuses() throws {
        let legacySelect: [String: JSONValue] = ["columns": .array([.string("a")]), "rename": .object(["a": .string("old")])]
        let legacyCast: [String: JSONValue] = ["columns": .object(["a": .string("int64")]), "missingPolicy": .string("preserve")]
        let legacy=DataFlowDocument(nodes:[.init(id:"select",operatorID:"kwip.data.selectRename",config:legacySelect),.init(id:"cast",operatorID:"kwip.data.cast",config:legacyCast)])
        let original=try DataFlowCodec.encodeGraph(legacy)
        XCTAssertEqual(try DataFlowCodec.decodeGraph(original),legacy,"Original saved configuration remains readable")
        XCTAssertThrowsError(try DataTableConfiguration.validate(legacySelect, operatorID: "kwip.data.selectRename", schema: schema))
        XCTAssertThrowsError(try DataTableConfiguration.validate(legacyCast, operatorID: "kwip.data.cast", schema: schema))
        XCTAssertEqual(try DataFlowCodec.encodeGraph(legacy),original,"Refusing retired editing must not rewrite original source")
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.reconcile(schema: schema)
        let saved = draft.configuration
        draft.update("a") { $0.outputName = "edited" }
        XCTAssertNotEqual(draft.configuration, saved)
        XCTAssertEqual(try ColumnsConfigurationDraft(configuration: saved).configuration, saved)
    }
    func testImageCanPassThroughButCannotBeSilentlyNullCast() throws {
        let image = [DataColumnSchema(name: "pixels", dtype: "image", nullable: false)]
        var draft = try ColumnsConfigurationDraft(configuration: ColumnsConfigurationDraft.defaults)
        draft.reconcile(schema: image)
        XCTAssertEqual(try draft.outputSchema(image)[0].dtype, "image")
        draft.update("pixels") { $0.targetType = "image"; $0.outputName = "picture" }
        draft.missingPolicy = "error"; draft.trimText = true
        XCTAssertEqual(try draft.outputSchema(image), [.init(name: "picture", dtype: "image", nullable: false)])
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: [.init(name: "pixels", dtype: "string", nullable: false)]))
        draft.castFailure = "null"; draft.update("pixels") { $0.targetType = "string" }
        XCTAssertThrowsError(try draft.validatedConfiguration(schema: image))
    }
}
