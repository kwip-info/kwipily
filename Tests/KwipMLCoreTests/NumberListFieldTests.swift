import XCTest
@testable import KwipMLCore

final class NumberListFieldTests:XCTestCase {
    func testFiniteDecimalsBoundsNullAndEmptyAreIndependent() throws {
        let field=ComponentField(key:"weights",title:"Weights",kind:.numberList,defaultValue:.array([0.25,1]),min:0,max:2,minItems:1,maxItems:3,allowsNull:true)
        XCTAssertTrue(field.isValidDefinition)
        XCTAssertTrue(field.accepts(.array([0,0.5,2])))
        XCTAssertTrue(field.accepts(.null))
        for value:JSONValue in [.array([]),.array([true]),.array([.string("1")]),.array([.null]),.array([.number(.infinity)]),.array([-0.1]),.array([2.1]),.array([1,1,1,1]),.number(1)] {XCTAssertFalse(field.accepts(value))}
        XCTAssertEqual(try JSONDecoder().decode(ComponentField.self,from:JSONEncoder().encode(field)),field)
        let empty=ComponentField(key:"w",title:"W",kind:.numberList,defaultValue:.array([]),min:-2.5,max:3.5,minItems:0,maxItems:4096)
        XCTAssertTrue(empty.isValidDefinition);XCTAssertTrue(empty.accepts(.array([])));XCTAssertFalse(empty.accepts(.null))
        XCTAssertTrue(empty.accepts(.array(Array(repeating:0.5,count:4096))))
        XCTAssertFalse(empty.accepts(.array(Array(repeating:0.5,count:4097))))
    }
    func testOptionalEnableAndCompositeContainment() {
        let field=ComponentField(key:"w",title:"W",kind:.numberList,defaultValue:.null,min:0.25,max:0.75,minItems:2,maxItems:3,allowsNull:true)
        XCTAssertTrue(field.isValidDefinition);XCTAssertEqual(field.valueWhenEnabled,.array([0.75,0.75]))
        var target=field;target.min=0;target.max=1;target.minItems=0;target.maxItems=4
        XCTAssertTrue(CompositeDefinitionValidation.canExpose(field,to:target));XCTAssertFalse(CompositeDefinitionValidation.canExpose(target,to:field))
        target.kind = .integerList;XCTAssertFalse(CompositeDefinitionValidation.canExpose(field,to:target))
        var bad=field;bad.maxItems=4097;XCTAssertFalse(bad.isValidDefinition)
        bad=field;bad.min = .nan;XCTAssertFalse(bad.isValidDefinition)
        bad=field;bad.enabledValue = .array([true,true]);XCTAssertFalse(bad.isValidDefinition)
        bad=field;bad.choices=[.init(value:"x",title:"X")];XCTAssertFalse(bad.isValidDefinition)
    }
}
