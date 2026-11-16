import XCTest
@testable import KwipMLCore

final class OptionalComponentFieldTests:XCTestCase {
    func testOptionalNumericValuesAndEnableDefault() throws {
        let field=ComponentField(key:"padding",title:"Padding row",defaultValue:.null,min:-10,max:9,allowsNull:true)
        XCTAssertTrue(field.isValidDefinition)
        XCTAssertTrue(field.accepts(.null));XCTAssertTrue(field.accepts(0))
        XCTAssertFalse(field.accepts(true));XCTAssertFalse(field.accepts(.number(.nan)))
        XCTAssertEqual(field.valueWhenEnabled,0)
        let number=ComponentField(key:"norm",title:"Maximum norm",kind:.number,defaultValue:.null,min:0.01,max:100,allowsNull:true)
        XCTAssertEqual(number.valueWhenEnabled,.number(1))
        var bounded=number;bounded.max=0.5
        XCTAssertEqual(bounded.valueWhenEnabled,.number(0.5))
        bounded.min=2;bounded.max=10
        XCTAssertEqual(bounded.valueWhenEnabled,.number(2))
        let embedding=try XCTUnwrap(EmbeddingComponent.definition.fields.first {$0.key == "max_norm"})
        XCTAssertEqual(embedding.valueWhenEnabled,.number(1))
        XCTAssertEqual(try JSONDecoder().decode(ComponentField.self,from:JSONEncoder().encode(field)),field)
    }
    func testLegacyFieldsRemainRequiredAndExposureCannotBroadenNullability() throws {
        let json=Data(#"{"key":"size","title":"Size","defaultValue":2,"min":1,"max":10}"#.utf8)
        let required=try JSONDecoder().decode(ComponentField.self,from:json)
        let encoded=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(required)) as? [String:Any])
        XCTAssertNil(encoded["allowsNull"],"Required legacy fields must not gain serialized default metadata")
        XCTAssertFalse(required.allowsNull);XCTAssertFalse(required.accepts(.null))
        var optional=required;optional.allowsNull=true
        XCTAssertFalse(CompositeDefinitionValidation.canExpose(optional,to:required))
        XCTAssertTrue(CompositeDefinitionValidation.canExpose(required,to:optional))
        var malformed=optional;malformed.kind = .boolean
        XCTAssertFalse(malformed.isValidDefinition)
    }
}
