import XCTest
@testable import KwipMLApp

final class DatabaseSourceCredentialPolicyTests: XCTestCase {
    func testProductionAndUnknownIdentitiesKeepDataProtection() {
        XCTAssertTrue(DatabaseSourceKeychain.storagePolicy(bundleIdentifier: "info.kwip.KwipML").dataProtection)
        XCTAssertTrue(DatabaseSourceKeychain.storagePolicy(bundleIdentifier: nil).dataProtection)
    }
    #if DEBUG
    func testOnlyDedicatedDevelopmentIdentityUsesSeparateCreatorProtectedService() {
        let policy = DatabaseSourceKeychain.storagePolicy(bundleIdentifier: "info.kwip.KwipML.POC")
        XCTAssertFalse(policy.dataProtection)
        XCTAssertEqual(policy.service, "info.kwip.KwipML.POC.database-source")
        XCTAssertNotEqual(policy.service, DatabaseSourceKeychain.storagePolicy(bundleIdentifier: "info.kwip.KwipML").service)
    }
    #endif
}
