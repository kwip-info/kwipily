import XCTest
import KwipMLCore
@testable import KwipMLApp
final class ReshapeTableConfigurationTests:XCTestCase {
    let schema:[DataColumnSchema]=[.init(name:"id",dtype:"string",nullable:false),.init(name:"jan",dtype:"int64",nullable:false),.init(name:"feb",dtype:"float64",nullable:true)]
    func testLongProjectionAndExplicitNumericWidening()throws {
        var c=ReshapeTableConfiguration.defaults;c["idColumns"] = .array([.string("id")]);c["valueColumns"] = .array([.string("feb"),.string("jan")])
        XCTAssertThrowsError(try ReshapeTableSchema.projected(c,schema:schema))
        c["typePolicy"] = .string("losslessNumeric")
        let result=try ReshapeTableSchema.projected(c,schema:schema)
        XCTAssertEqual(result.map(\.name),["id","field","value"]);XCTAssertEqual(result.last?.dtype,"float64");XCTAssertEqual(result.last?.nullable,true)
        c["missingValues"] = .string("drop");XCTAssertEqual(try ReshapeTableSchema.projected(c,schema:schema).last?.nullable,false)
        c["fieldName"] = .string("id");XCTAssertThrowsError(try ReshapeTableSchema.projected(c,schema:schema))
    }
    func testWideVocabularyRemainsObservedAndTypesValidated()throws {
        var c=ReshapeTableConfiguration.defaults;c["mode"] = .string("wide")
        XCTAssertTrue(try ReshapeTableSchema.projected(c,schema:schema).isEmpty)
        c["columnKey"] = .string("id");c["values"] = .array([.object(["column":.string("id"),"aggregation":.string("sum"),"prefix":.string(""),"nullPolicy":.string("skip")])])
        XCTAssertThrowsError(try ReshapeTableSchema.validate(c,schema:schema))
    }
}
