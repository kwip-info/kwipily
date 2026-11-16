import XCTest
@testable import KwipMLCore

final class SortUniqueContractTests:XCTestCase {
    func testOrderedSortKeysAndPoliciesRoundTrip()throws {
        let values:[String:JSONValue]=["keys":.array([SortConfiguration.Key(column:"customer",comparison:"casefold").value,SortConfiguration.Key(column:"priority",direction:"descending",missing:"first").value])]
        let config=try SortConfiguration(values)
        XCTAssertEqual(config.values,values)
        XCTAssertEqual(config.keys.map(\.column),["customer","priority"])
        XCTAssertThrowsError(try SortConfiguration(SortConfiguration.defaults))
        XCTAssertThrowsError(try SortConfiguration(["keys":.array([config.keys[0].value,config.keys[0].value])]))
        var bad=values;bad["extra"] = .bool(true);XCTAssertThrowsError(try SortConfiguration(bad))
    }
    func testUniquePoliciesAndDeterministicPractice()throws {
        var values=UniqueConfiguration.defaults;values["keys"] = .array(["customer","region"].map(JSONValue.string));values["keep"] = .string("last");values["missing"] = .string("distinct");values["comparison"] = .string("casefold")
        XCTAssertEqual(try UniqueConfiguration(values).values,values)
        values["missing"] = .string("guess");XCTAssertThrowsError(try UniqueConfiguration(values))
        XCTAssertThrowsError(try UniqueConfiguration(UniqueConfiguration.defaults))
        for document in [try SortPracticeExample.document(),try UniquePracticeExample.document()] {XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)}
        XCTAssertTrue(FileManager.default.fileExists(atPath:try SortPracticeExample.resourceURL().path))
        XCTAssertTrue(FileManager.default.fileExists(atPath:try UniquePracticeExample.resourceURL().path))
    }
}
