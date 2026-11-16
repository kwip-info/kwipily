import XCTest
@testable import KwipMLCore

final class DesktopLicenseWireTests:XCTestCase {
    func testRetiredLicensePaginationIsRefused() throws {
        let id="00000000-0000-0000-0000-000000000001"
        XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(Data("{\"schemaVersion\":1,\"operation\":\"licenses\",\"bearer\":\"test\",\"body\":{\"after\":\"\(id)\"}}".utf8)))
        for body in [#"{"after":"https://evil.example"}"#,#"{"after":"00000000-0000-0000-0000-000000000001","url":"x"}"#] {
            XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(Data("{\"schemaVersion\":1,\"operation\":\"licenses\",\"bearer\":\"test\",\"body\":\(body)}".utf8)))
        }
    }
    func testEmptyPageIsHonestAndUnknownRightsFail() throws {
        XCTAssertTrue(try DesktopLicenseWire.licenses(Data(#"{"schemaVersion":1,"nextCursor":null,"licenses":[]}"#.utf8)).licenses.isEmpty)
        for text in [#"{"schemaVersion":true,"nextCursor":null,"licenses":[]}"#,#"{"schemaVersion":1,"nextCursor":"wrong","licenses":[]}"#,#"{"schemaVersion":1,"nextCursor":null,"licenses":[{"licenseID":"00000000-0000-0000-0000-000000000001","majorVersion":2,"perpetual":true,"updates":"v1.x","maximumDevices":2,"state":"settled","activations":[]}]}"#] {
            XCTAssertThrowsError(try DesktopLicenseWire.licenses(Data(text.utf8)))
        }
    }
}
