import Foundation
import XCTest
@testable import KwipMLNetworking
import KwipMLCore

private final class FixtureProtocol: URLProtocol {
    static var responseURL: URL?
    static var status = 200
    static var bytes = Data("{}".utf8)
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let response = HTTPURLResponse(url: Self.responseURL ?? request.url!, statusCode: Self.status, httpVersion: "HTTP/1.1", headerFields: ["Content-Type":"application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.bytes)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
final class BoundedTransportTests: XCTestCase {
    override func setUpWithError() throws {
        #if !DEBUG
        throw XCTSkip("URLProtocol fixture injection is intentionally unavailable in production builds; run these controls in debug.")
        #endif
    }
    private func transport() -> BoundedNetworkTransport {
        let value = BoundedNetworkTransport()
        #if DEBUG
        value.testProtocolClasses = [FixtureProtocol.self]
        #endif
        return value
    }
    func testBoundedSuccessAndOversizedStreamRefusal() async throws {
        FixtureProtocol.status = 200; FixtureProtocol.bytes = Data("{}".utf8)
        let good = try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8))
        XCTAssertEqual(good, FixtureProtocol.bytes)
        FixtureProtocol.bytes = Data(repeating: 65, count: 65_537)
        do {
            _ = try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8))
            XCTFail("Oversized body accepted")
        } catch {}
    }
    func testRedirectAndErrorResponsesNeverBecomeSuccess() async {
        FixtureProtocol.bytes = Data("{}".utf8)
        for status in [301,302,307,401,403,500] {
            FixtureProtocol.status = status
            do {
                _ = try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8))
                XCTFail("HTTP \(status) accepted")
            } catch {}
        }
    }
    func testUnauthorizedCanBeDistinguishedFromTemporaryServiceFailure() async {
        FixtureProtocol.bytes=Data("{}".utf8)
        FixtureProtocol.status=401
        do {_=try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8));XCTFail("Unauthorized accepted")}
        catch NetworkBrokerFailure.signInRequired {} catch {XCTFail("Missing sign-in failure")}
        FixtureProtocol.status=503
        do {_=try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8));XCTFail("Outage accepted")}
        catch NetworkBrokerFailure.signInRequired {XCTFail("Outage must not look like account expiry")} catch {}
    }
    func testDeviceLimitAndRefundAreActionableWithoutTrustingServerCopy() async {
        FixtureProtocol.status=409;FixtureProtocol.bytes=Data(#"{"code":"device_limit","error":"untrusted provider copy"}"#.utf8)
        do {_=try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8));XCTFail("Limit accepted")}
        catch NetworkBrokerFailure.deviceLimit {} catch {XCTFail("Missing device-limit failure")}
        FixtureProtocol.status=403;FixtureProtocol.bytes=Data(#"{"code":"not_entitled","error":"untrusted provider copy"}"#.utf8)
        do {_=try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8));XCTFail("Refund accepted")}
        catch NetworkBrokerFailure.notEntitled {} catch {XCTFail("Missing entitlement failure")}
    }
    func testResponseCannotSubstituteHostPortCredentialsOrRoute() async {
        FixtureProtocol.status = 200; FixtureProtocol.bytes = Data("{}".utf8)
        defer { FixtureProtocol.responseURL = nil }
        for address in ["https://evil.example/api/v1/health", DesktopDeployment.origin + "/api/v1/account",
                        "http://127.0.0.1:8769/api/v1/health", "https://user@apps.kwip.info/api/v1/health"] {
            FixtureProtocol.responseURL = URL(string: address)!
            do {
                _ = try await transport().perform(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8))
                XCTFail("Substituted response destination accepted")
            } catch {}
        }
    }
    func testRedirectDelegateRefusesAlternateDestination() {
        let response = HTTPURLResponse(url: URL(string:"https://apps.kwip.info/api/v1/health")!, statusCode:302,httpVersion:nil,headerFields:nil)!
        let session = URLSession(configuration:.ephemeral)
        defer { session.invalidateAndCancel() }
        let task = session.dataTask(with: response.url!)
        transport().urlSession(session,task:task,willPerformHTTPRedirection:response,newRequest:URLRequest(url:URL(string:"https://evil.example")!)) { XCTAssertNil($0) }
    }
}
