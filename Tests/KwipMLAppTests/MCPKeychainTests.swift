import XCTest
@testable import KwipMLApp
final class MCPKeychainTests:XCTestCase {
    func testDisposableNamespaceRoundtripRotationAndRemoval()throws {
        let store=MCPSecretKeychain(namespace:"test-"+UUID().uuidString),id=UUID().uuidString
        defer{try? store.remove(clientID:id)}
        let first=try MCPSecretKeychain.makeToken(),second=try MCPSecretKeychain.makeToken()
        XCTAssertEqual(first.utf8.count,43);XCTAssertNotEqual(first,second)
        try store.write(first,clientID:id);XCTAssertTrue(try store.read(clientID:id)==first)
        try store.write(second,clientID:id);XCTAssertTrue(try store.read(clientID:id)==second)
        try store.remove(clientID:id);XCTAssertNil(try store.read(clientID:id))
    }
}
