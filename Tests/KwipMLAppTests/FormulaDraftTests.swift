import XCTest
import KwipMLCore
@testable import KwipMLApp
final class FormulaDraftTests:XCTestCase {
    func testGuidedRegexRoundTripEscapesAndCaptureMappings()throws {
        var guided=FormulaGuidedRegex(kind:"extract",column:"product_label")
        guided.pattern="^([A-Z]+)-([0-9]+)$";guided.matchMode="full";guided.groups=["1","2"]
        guided.flags=["ignoreCase"]
        let decoded=try XCTUnwrap(FormulaGuidedRegex(expression:guided.expression))
        XCTAssertEqual(decoded,guided)
        var replace=FormulaGuidedRegex(kind:"replace",column:"name");replace.pattern="\\s+";replace.replacement="\"\n\\1"
        XCTAssertEqual(FormulaGuidedRegex(expression:replace.expression),replace)
    }
    func testGridReorderDuplicateRemoveAndSequentialSchema()throws {
        var draft=try FormulaDraft(configuration:FormulaDraft.defaults)
        draft.rows[0].outputs[0].name="total";draft.rows[0].outputs[0].dtype="float64";draft.rows[0].expression="[price] * 2"
        draft.add(kind:"expression",column:"total");draft.rows[1].outputs[0].name="next";draft.rows[1].outputs[0].dtype="float64";draft.rows[1].expression="[total] + 1"
        let schema=[DataColumnSchema(name:"price",dtype:"float64",nullable:false)]
        XCTAssertEqual(try draft.outputSchema(schema).map(\.name),["price","total","next"])
        let id=draft.rows[1].id;draft.move(id,by:-1);XCTAssertThrowsError(try draft.validated(schema:schema))
        draft.move(id,by:1);draft.duplicate(id);XCTAssertEqual(Set(draft.rows.map(\.id)).count,3)
        XCTAssertThrowsError(try draft.validated(schema:schema));draft.remove(draft.rows.last!.id)
        XCTAssertEqual(try FormulaDraft(configuration:draft.configuration),draft)
    }
    func testReplacementChoicesExcludeImagesAndOtherWritesWithoutChangingDraft()throws {
        var draft=try FormulaDraft(configuration:FormulaDraft.defaults)
        draft.rows[0].outputs=[.init(name:"name",dtype:"string",write:"replace"),.init(name:"capture",dtype:"string")]
        draft.add(kind:"expression",column:"name")
        draft.rows[1].outputs[0].name="price"
        let schema=[DataColumnSchema(name:"name",dtype:"string",nullable:false),
                    .init(name:"price",dtype:"float64",nullable:false),
                    .init(name:"pixels",dtype:"image",nullable:false),
                    .init(name:"active",dtype:"bool",nullable:false)]
        let before=draft.configuration
        XCTAssertEqual(draft.replacementColumns(for:draft.rows[0].id,output:0,schema:schema).map(\.name),["name","active"])
        XCTAssertEqual(draft.replacementColumns(for:draft.rows[0].id,output:1,schema:schema).map(\.name),["active"])
        XCTAssertEqual(draft.replacementColumns(for:draft.rows[1].id,output:0,schema:schema).map(\.name),["price","active"])
        XCTAssertTrue(draft.replacementColumns(for:draft.rows[0].id,output:0,schema:[]).isEmpty)
        XCTAssertEqual(draft.configuration,before)
    }
    func testImagePassThroughCannotBecomeScalarOrReplacementTarget()throws {
        var draft=try FormulaDraft(configuration:FormulaDraft.defaults)
        let schema=[DataColumnSchema(name:"pixels",dtype:"image",nullable:false)]
        XCTAssertEqual(try draft.outputSchema(schema).first?.dtype,"image")
        draft.rows[0].expression="[pixels]";XCTAssertThrowsError(try draft.validated(schema:schema))
        draft.rows[0].expression="0";draft.rows[0].outputs[0].name="pixels";draft.rows[0].outputs[0].write="replace"
        XCTAssertThrowsError(try draft.validated(schema:schema))
    }
}
