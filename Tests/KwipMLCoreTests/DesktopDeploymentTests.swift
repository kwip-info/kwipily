import XCTest
@testable import KwipMLCore

final class DesktopDeploymentTests: XCTestCase {
    func testStagingAndProductionPreserveSeparateIdentities() {
        XCTAssertFalse(DesktopDeployment.isCommercialEdition(bundle: "other.app"))
        XCTAssertEqual(DesktopDeployment.applicationSupportDirectoryName, DesktopDeployment.isStaging ? "KwipML Staging" : "KwipML")
        XCTAssertNotEqual(DesktopDeployment.keychainService("account"), DesktopDeployment.keychainService("license"))
    }
    func testCompiledEditionPinsEveryTrustBoundary() throws {
        #if KWIPML_COMMERCE_STAGING
        XCTAssertEqual(DesktopDeployment.origin, "http://127.0.0.1:8768")
        XCTAssertEqual(DesktopDeployment.callbackScheme, "kwipml-staging")
        XCTAssertEqual(DesktopDeployment.bundleIdentifier, "info.kwip.KwipML.POC")
        XCTAssertEqual(DesktopDeployment.environment, "staging")
        #else
        XCTAssertEqual(DesktopDeployment.origin, "https://apps.kwip.info")
        XCTAssertEqual(DesktopDeployment.callbackScheme, "kwipml")
        XCTAssertEqual(DesktopDeployment.bundleIdentifier, "info.kwip.KwipML")
        XCTAssertEqual(DesktopDeployment.environment, "production")
        #endif
        XCTAssertEqual(DesktopDeployment.networkServiceIdentifier, DesktopDeployment.bundleIdentifier + ".NetworkService")
        XCTAssertEqual(DesktopDeployment.keychainService("account"), "info.kwip.KwipML.account." + DesktopDeployment.environment + ".v1")
        let flow = try DesktopAuthorization()
        XCTAssertEqual(URLComponents(url: flow.startURL, resolvingAgainstBaseURL: false)?.host,
                       URL(string: DesktopDeployment.origin)?.host)
        let code = String(repeating: "a", count: 43)
        let callback = "\(DesktopDeployment.callbackScheme)://auth/callback?code=\(code)&state=\(flow.state)"
        XCTAssertEqual(try flow.code(from: URL(string: callback)!), code)
        let otherScheme = DesktopDeployment.environment == "production" ? "kwipml-staging" : "kwipml"
        XCTAssertThrowsError(try flow.code(from: URL(string: callback.replacingOccurrences(of: DesktopDeployment.callbackScheme + ":", with: otherScheme + ":"))!))
        let request = try NetworkBrokerPolicy.prepare(Data(#"{"schemaVersion":1,"operation":"health"}"#.utf8))
        XCTAssertEqual(request.url?.absoluteString, DesktopDeployment.origin + "/api/v1/health")
    }
}
