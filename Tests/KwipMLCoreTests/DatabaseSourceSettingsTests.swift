import XCTest
@testable import KwipMLCore

final class DatabaseSourceSettingsTests: XCTestCase {
    func testExactEndpointIdentityChangesWithoutEmbeddingCredentials() throws {
        let first: [String: JSONValue] = ["backend": .string("postgresql"), "host": .string("db.example.com"), "database": .string("learning"), "table": .string("observations")]
        let a = try DatabaseSourceSettings.connectionIdentity(first)
        var changed = first; changed["host"] = .string("different.example.com")
        XCTAssertNotEqual(a, try DatabaseSourceSettings.connectionIdentity(changed))
        changed = first; changed["rowLimit"] = .integer(20)
        XCTAssertEqual(a, try DatabaseSourceSettings.connectionIdentity(changed))
        XCTAssertThrowsError(try DatabaseSourceSettings.normalized(first.merging(["password": .string("private")]) { _, new in new }))
    }
    func testTLSAndNumericBoundsAreReal() throws {
        var config: [String: JSONValue] = ["backend": .string("mysql"), "host": .string("example.com"), "database": .string("learning"), "table": .string("observations"), "tlsMode": .string("disabledLoopback")]
        XCTAssertThrowsError(try DatabaseSourceSettings.normalized(config))
        config["host"] = .string("127.0.0.1")
        XCTAssertNoThrow(try DatabaseSourceSettings.normalized(config))
        config["rowLimit"] = .integer(Int64.max)
        XCTAssertThrowsError(try DatabaseSourceSettings.normalized(config))
    }
}
