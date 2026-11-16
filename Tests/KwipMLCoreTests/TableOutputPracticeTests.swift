import XCTest
@testable import KwipMLCore

final class TableOutputPracticeTests:XCTestCase {
    func testCleanPracticeHasRealSourceAndStablePassThroughOutput()throws {
        let document=try TableOutputPracticeExample.document(projectID:"00000000-0000-4000-8000-000000000012")
        try WorkflowCodec.validate(document)
        XCTAssertEqual(document.nodes.count,2)
        let output=try XCTUnwrap(document.nodes.first {$0.address.localID == "table-output"})
        guard case .dataOperator(let id,let config)=output.payload else {return XCTFail("Missing Table Output")}
        XCTAssertEqual(id,"kwip.data.outputDelimited")
        XCTAssertEqual(config["overwrite"],.bool(false))
        XCTAssertEqual(config["format"],.string("csv"))
        XCTAssertFalse(config.keys.contains("destination"))
        let source=try String(contentsOf:TableOutputPracticeExample.resourceURL(),encoding:.utf8)
        XCTAssertTrue(source.contains("A12,6,3.125,last row"))
        XCTAssertEqual(TableOutputPracticeExample.sourceResources.count,1)
    }
}
