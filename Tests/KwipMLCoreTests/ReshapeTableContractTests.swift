import XCTest
@testable import KwipMLCore
final class ReshapeTableContractTests:XCTestCase {
    func testDefaultsAndPolicies()throws {
        XCTAssertEqual(try ReshapeTableConfiguration(ReshapeTableConfiguration.defaults).string("mode"),"long")
        for (key,value) in [("mode",JSONValue.string("guess")),("maxColumns",.integer(129)),("idColumns",.array([.string("id"),.string("id")]))] {
            var c=ReshapeTableConfiguration.defaults;c[key]=value;XCTAssertThrowsError(try ReshapeTableConfiguration(c))
        }
    }
    func testDedicatedPracticeResourcesAndDocuments()throws {
        for wide in [false,true] {
            let data=try String(contentsOf:ReshapeTablePracticeExample.resourceURL(wide:wide),encoding:.utf8)
            XCTAssertTrue(data.contains(wide ? "sales,units":"jan,feb"))
            let document=try ReshapeTablePracticeExample.document(projectID:"12345678-1234-4234-8234-123456789ABC",wide:wide)
            XCTAssertEqual(document.nodes.count,3);XCTAssertEqual(document.edges.count,2)
            try WorkflowCodec.validate(document)
        }
    }
}
