import XCTest
@testable import KwipMLCore

final class AttentionMaskListFieldTests:XCTestCase {
    func testDeclaredEmptyCoordinatesAnd4096PositionCeilingStayStrict() throws {
        let field=ComponentField(key:"padding_keys",title:"Blocked source positions",kind:.integerList,defaultValue:.array([]),min:0,max:4095,minItems:0,maxItems:4096)
        XCTAssertTrue(field.isValidDefinition);XCTAssertTrue(field.accepts(.array([])))
        XCTAssertTrue(field.accepts(.array((0..<4096).map {.integer(Int64($0))})))
        for value:JSONValue in [.null,.array([true]),.array([-1]),.array([4096]),.array([0.5]),.array(Array(repeating:1,count:4097))] {XCTAssertFalse(field.accepts(value))}
        XCTAssertEqual(try JSONDecoder().decode(ComponentField.self,from:JSONEncoder().encode(field)),field)
        for change in [-1,4097] {var invalid=field;if change<0 {invalid.minItems=change} else {invalid.maxItems=change};XCTAssertFalse(invalid.isValidDefinition)}
    }
    func testLegacyShapeListBoundsAndNonnullableDefaultsArePreserved() {
        let old=ComponentField(key:"shape",title:"Shape",kind:.integerList,defaultValue:.array([-1]),min:-1,max:100_000_000,minItems:1,maxItems:7)
        XCTAssertTrue(old.isValidDefinition);XCTAssertFalse(old.accepts(.array([])));XCTAssertFalse(old.accepts(.array(Array(repeating:1,count:8))))
        var invalid=old;invalid.defaultValue = .null;XCTAssertFalse(invalid.isValidDefinition)
        for id in AttentionRuntimeCapabilities.operations {for field in BundledAttentionComponents.definition(id).fields {XCTAssertTrue(field.isValidDefinition,"\(id).\(field.key)")}}
    }
}
