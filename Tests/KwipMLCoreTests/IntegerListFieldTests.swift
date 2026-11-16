import XCTest
@testable import KwipMLCore

final class IntegerListFieldTests:XCTestCase {
    func testBoundedListsAndLegacyFieldRoundTrip()throws {
        let field=ComponentField(key:"shape",title:"Shape",kind:.integerList,defaultValue:.array([-1]),min:-1,max:100_000_000,minItems:1,maxItems:7)
        XCTAssertTrue(field.isValidDefinition)
        XCTAssertTrue(field.accepts(.array([2.0,-1])))
        for value:JSONValue in [.array([]),.array([true]),.array([1.5]),.array(Array(repeating:1,count:8)),.array([-2]),.string("2,3")] {
            XCTAssertFalse(field.accepts(value))
        }
        XCTAssertEqual(try JSONDecoder().decode(ComponentField.self,from:JSONEncoder().encode(field)),field)
        let old=try JSONDecoder().decode(ComponentField.self,from:Data(#"{"key":"width","title":"Width","defaultValue":2,"min":1,"max":512}"#.utf8))
        XCTAssertTrue(old.isValidDefinition)
        XCTAssertNil(old.minItems)
        var narrower=field;narrower.maxItems=3
        XCTAssertTrue(CompositeDefinitionValidation.canExpose(narrower,to:field))
        XCTAssertFalse(CompositeDefinitionValidation.canExpose(field,to:narrower))
        narrower.minItems = -1
        XCTAssertFalse(narrower.isValidDefinition)
    }
}
