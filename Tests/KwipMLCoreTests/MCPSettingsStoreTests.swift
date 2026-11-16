import XCTest
@testable import KwipMLCore
final class MCPSettingsStoreTests:XCTestCase {
    func root()throws->URL {let url=FileManager.default.temporaryDirectory.appendingPathComponent("mcp-test-"+UUID().uuidString);try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true);addTeardownBlock{try? FileManager.default.removeItem(at:url)};return url}
    func testPrivateAtomicRoundtripAndMissingDefaults()throws {
        let root=try root(),store=MCPSettingsStore(root:root)
        XCTAssertEqual(try store.load(),.factory)
        var s=MCPSettings.factory;s.fixedPort=8788;try store.save(s)
        XCTAssertEqual(try store.load(),s)
        let mode=(try FileManager.default.attributesOfItem(atPath:root.appendingPathComponent("mcp/settings.json").path)[.posixPermissions] as? NSNumber)?.intValue
        XCTAssertEqual(mode,0o600)
    }
    func testSymlinkAndMalformedPolicyRefused()throws {
        let root=try root(),store=MCPSettingsStore(root:root);_ = try store.load()
        let target=root.appendingPathComponent("other");try Data("{}".utf8).write(to:target)
        let settings=root.appendingPathComponent("mcp/settings.json")
        try FileManager.default.createSymbolicLink(at:settings,withDestinationURL:target)
        XCTAssertThrowsError(try store.load());XCTAssertThrowsError(try store.save(.factory))
        XCTAssertEqual(try Data(contentsOf:target),Data("{}".utf8))
    }
    func testSharedDirectoryRefused()throws {
        let root=try root(),store=MCPSettingsStore(root:root);_ = try store.load()
        try FileManager.default.setAttributes([.posixPermissions:0o777],ofItemAtPath:root.appendingPathComponent("mcp").path)
        XCTAssertThrowsError(try store.load());XCTAssertThrowsError(try store.save(.factory))
    }
}
