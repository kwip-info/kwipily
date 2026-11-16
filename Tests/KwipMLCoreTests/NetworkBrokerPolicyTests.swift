import Foundation
import XCTest
@testable import KwipMLCore

final class NetworkBrokerPolicyTests: XCTestCase {
    func testRetiredCommerceOperationsAreRefusedEvenWithCredentials() throws {
        for operation in ["activate", "releaseDevice", "account", "licenses", "signOut", "exchangeCode"] {
            let message=try JSONSerialization.data(withJSONObject:["schemaVersion":1,"operation":operation,"body":[:],"bearer":"old-session"] as [String:Any])
            XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(message))
        }
    }
    func testRejectsMalformedOversizedAndUnknownMessages() {
        for text in ["{}", "[]", #"{"schemaVersion":2,"operation":"health"}"#,
                     #"{"schemaVersion":true,"operation":"health"}"#,
                     #"{"schemaVersion":1,"operation":"https://evil.example"}"#,
                     #"{"schemaVersion":1,"operation":"health","url":"https://evil.example"}"#,
                     #"{"schemaVersion":1,"operation":"health","bearer":"x\r\nInjected: yes"}"#,
                     #"{"schemaVersion":1,"operation":"activate","body":[]}"#] {
            XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(Data(text.utf8)), text)
        }
        XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(Data(repeating: 32, count: 16_385)))
    }
    func testIdentityOperationsRequireBearerAndHealthHasNoBody() {
        XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(Data(#"{"schemaVersion":1,"operation":"activate","body":{}}"#.utf8)))
        XCTAssertThrowsError(try NetworkBrokerPolicy.prepare(Data(#"{"schemaVersion":1,"operation":"health","body":{"x":1}}"#.utf8)))
    }
    func testNoCookiePersistenceAndBoundedTimeout() throws {
        let request = try NetworkBrokerPolicy.prepare(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8))
        XCTAssertEqual(request.httpMethod,"GET")
        XCTAssertFalse(request.httpShouldHandleCookies)
        XCTAssertEqual(request.timeoutInterval,20)
        XCTAssertNil(request.httpBody)
    }
}
