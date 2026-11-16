import XCTest
@testable import KwipMLCore

final class ProductBrandTests: XCTestCase {
    func testIsolatedServiceUsesOnlyItsSignedMetadata() {
        XCTAssertEqual(ProductBrand.serviceName(in:["KwipMLProductName":"Kwipily"]),"Kwipily")
        XCTAssertEqual(ProductBrand.serviceName(in:[:]),"Local workflow service")
        XCTAssertEqual(ProductBrand.serviceName(in:["KwipMLProductName":String(repeating:"x",count:121)]),"Local workflow service")
    }
    func testVisibleBrandDoesNotMoveCompatibilityIdentity() {
        XCTAssertEqual(ProductBrand.name, "Kwipily")
        XCTAssertEqual(ProductBrand.formerName, "Kwip ML")
        XCTAssertEqual(DesktopDeployment.bundleIdentifier, "info.kwip.KwipML")
        XCTAssertEqual(DesktopDeployment.callbackScheme, "kwipml")
        XCTAssertEqual(DesktopDeployment.applicationSupportDirectoryName, "KwipML")
        XCTAssertEqual(DesktopDeployment.keychainService("license"), "info.kwip.KwipML.license.production.v1")
        XCTAssertTrue(ProductBrand.transitionMessage.contains("existing v1 license"))
    }
}
