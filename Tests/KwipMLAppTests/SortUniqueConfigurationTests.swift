import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SortUniqueConfigurationTests:XCTestCase {
    let schema:[DataColumnSchema]=[.init(name:"customer",dtype:"string",nullable:true),.init(name:"priority",dtype:"int64",nullable:false),.init(name:"photo",dtype:"image",nullable:true)]
    func testKeysAreKnownReadonlyScalarColumns()throws {
        let good:[String:JSONValue]=["keys":.array([SortConfiguration.Key(column:"customer").value])]
        XCTAssertEqual(try SortTableConfiguration.validate(good,schema:schema),good)
        for column in ["missing","photo"] {XCTAssertThrowsError(try SortTableConfiguration.validate(["keys":.array([SortConfiguration.Key(column:column).value])],schema:schema))}
        var unique=UniqueConfiguration.defaults;unique["keys"] = .array(["customer","priority"].map(JSONValue.string))
        XCTAssertEqual(try UniqueTableConfiguration.validate(unique,schema:schema),unique)
        unique["keys"] = .array([.string("photo")]);XCTAssertThrowsError(try UniqueTableConfiguration.validate(unique,schema:schema))
    }
}
