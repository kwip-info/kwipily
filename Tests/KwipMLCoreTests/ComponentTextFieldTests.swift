import XCTest
@testable import KwipMLCore

final class ComponentTextFieldTests:XCTestCase {
    private func field(_ changes:[String:Any]=[:])throws->ComponentField {
        var record:[String:Any]=["key":"equation","title":"Custom notation","kind":"text","defaultValue":"nab,nbc->nac","minItems":1,"maxItems":128]
        changes.forEach {record[$0.key]=$0.value}
        return try JSONDecoder().decode(ComponentField.self,from:JSONSerialization.data(withJSONObject:record))
    }
    func testBoundedTextTemplateAdmitsStringsAndRefusesOtherJSONTypes()throws {
        let f=try field();XCTAssertTrue(f.isValidDefinition)
        XCTAssertTrue(f.accepts(.string("naa->na")));XCTAssertTrue(f.accepts(.string(String(repeating:"a",count:128))))
        for value in [.string(""),.string(String(repeating:"a",count:129)),1,true,.null,.array([1])] as [JSONValue] {XCTAssertFalse(f.accepts(value))}
    }
    func testTextTemplateRejectsMissingInvalidOrNontextDefinitionProperties()throws {
        for change in [["minItems":0],["maxItems":0],["maxItems":4097],["min":1],["max":128],["allowsNull":true],["defaultValue":1],["choices":[["value":"x","title":"X"]]]] as [[String:Any]] {
            XCTAssertFalse(try field(change).isValidDefinition)
        }
    }
    func testExistingIntegerAndChoiceBytesRetainTheirHistoricalFields()throws {
        let f=ComponentField(key:"size",title:"Size",kind:.integer,defaultValue:2,min:1,max:8)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        XCTAssertEqual(String(data:try encoder.encode(f),encoding:.utf8),"{\"defaultValue\":2,\"key\":\"size\",\"kind\":\"integer\",\"max\":8,\"min\":1,\"title\":\"Size\"}")
        XCTAssertTrue(f.isValidDefinition);XCTAssertFalse(f.accepts(.string("2")))
    }
}
