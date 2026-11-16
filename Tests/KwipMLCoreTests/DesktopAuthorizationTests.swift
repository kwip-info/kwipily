import Foundation
import XCTest
@testable import KwipMLCore

final class DesktopAuthorizationTests:XCTestCase {
    func testPKCEUsesPublishedS256VectorAndSeparateRandomState() throws {
        XCTAssertEqual(DesktopAuthorization.challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),"E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")
        let first=try DesktopAuthorization(), second=try DesktopAuthorization()
        XCTAssertNotEqual(first.state,second.state)
        XCTAssertNotEqual(first.verifier,second.verifier)
        XCTAssertEqual(first.startURL.host,URL(string:DesktopDeployment.origin)?.host)
        XCTAssertFalse(first.startURL.absoluteString.contains(first.verifier))
    }
    func testCallbackRequiresExactDestinationStateAndSingleCode() throws {
        let flow=try DesktopAuthorization()
        let code=String(repeating:"c",count:43)
        XCTAssertEqual(try flow.code(from:URL(string:"\(DesktopDeployment.callbackScheme)://auth/callback?code=\(code)&state=\(flow.state)")!),code)
        for raw in ["\(DesktopDeployment.callbackScheme)://other/callback?code=\(code)&state=\(flow.state)","\(DesktopDeployment.callbackScheme)://auth/callback?code=\(code)&state=wrong", "\(DesktopDeployment.callbackScheme)://auth/callback?code=\(code)&state=\(flow.state)&code=duplicate","\(DesktopDeployment.callbackScheme)://auth/callback?code=\(code)&state=\(flow.state)#fragment","https://auth/callback?code=\(code)&state=\(flow.state)"] {
            XCTAssertThrowsError(try flow.code(from:URL(string:raw)!))
        }
    }
    func testWireResponsesRejectUnexpectedVersionTokenAndIdentity() throws {
        XCTAssertThrowsError(try DesktopAuthorization.token(from:Data(#"{"schemaVersion":true,"token":"bad"}"#.utf8)))
        XCTAssertThrowsError(try DesktopAuthorization.token(from:Data(#"{"schemaVersion":1,"token":"contains spaces"}"#.utf8)))
        XCTAssertThrowsError(try DesktopAuthorization.customer(from:Data(#"{"schemaVersion":1,"customerID":"user_provider"}"#.utf8)))
        XCTAssertEqual(try DesktopAuthorization.token(from:JSONSerialization.data(withJSONObject:["schemaVersion":1,"token":String(repeating:"t",count:43)])),String(repeating:"t",count:43))
    }
}
