import XCTest
import KwipMLCore
@testable import KwipMLApp
final class Data02ObservedSchemaProjectionTests:XCTestCase {
    func testObservedColumnsRequireCurrentEvidenceAndRetainMeasuredTypes() {
        let preview:JSONValue = .object(["preview":.object(["columns":.array([.object(["name":.string("renamed"),"dtype":.string("float64"),"nullable":.bool(false)])])])])
        XCTAssertEqual(DataObservedSchemaProjection.columns(preview,isCurrent:true),[.init(name:"renamed",dtype:"float64",nullable:false)])
        XCTAssertNil(DataObservedSchemaProjection.columns(preview,isCurrent:false))
        XCTAssertNil(DataObservedSchemaProjection.columns(.object([:]),isCurrent:true))
    }
}
