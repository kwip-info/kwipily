import XCTest
@testable import KwipMLCore

final class PreparationSamplingTests: XCTestCase {
    func testPolicyTransportHasOnlyActiveLimit() throws {
        let value = try PreparationSampling(mode: .rowCount, rowCount: 1000).transportJSON()
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(value.utf8)) as? [String:Any])
        XCTAssertEqual(Set(document.keys), ["schemaVersion","mode","rowCount"])
        XCTAssertEqual(document["rowCount"] as? Int, 1000)
    }
    func testInvalidAndInactiveLimitsRefuse() {
        for value in [PreparationSampling(mode:.rowCount,rowCount:0), .init(mode:.rowCount,rowCount:5_000_001), .init(mode:.percentage,percentage:.nan), .init(mode:.percentage,percentage:0), .init(mode:.all,rowCount:1), .init(mode:.percentage,rowCount:1,percentage:10)] {
            XCTAssertThrowsError(try value.validated())
        }
        XCTAssertNoThrow(try PreparationSampling(mode:.percentage,percentage:0.01).validated())
    }
}
