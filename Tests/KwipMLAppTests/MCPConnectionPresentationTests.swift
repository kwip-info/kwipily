import XCTest
@testable import KwipMLApp

final class MCPConnectionPresentationTests:XCTestCase {
    func testMaskedDisplayKeepsActualCopyPayloadAndExplicitReveal()throws {
        let credential=String(repeating:"a",count:43)
        let json="{\"mcpServers\":{\"kwip-ml\":{\"url\":\"http://127.0.0.1:8781/mcp\",\"headers\":{\"Authorization\":\"Bearer \(credential)\"}}}}"
        let details=try XCTUnwrap(MCPConnectionDetails(configuration:json))
        XCTAssertEqual(details.endpoint,"http://127.0.0.1:8781/mcp")
        XCTAssertEqual(details.authorization,"Bearer "+credential)
        XCTAssertFalse(details.displayConfiguration(reveal:false).contains(credential))
        XCTAssertTrue(details.displayConfiguration(reveal:false).contains("<client credential>"))
        XCTAssertEqual(details.displayConfiguration(reveal:true),json)
    }
    func testIncompleteOrNonlocalConnectionCannotBePresented() {
        for url in ["https://example.com/mcp","http://127.0.0.1:8781/mcp?token=secret","http://127.0.0.1:8781/mcp#secret"] {
            XCTAssertNil(MCPConnectionDetails(configuration:"{\"mcpServers\":{\"kwip-ml\":{\"url\":\"\(url)\",\"headers\":{\"Authorization\":\"Bearer value\"}}}}"))
        }
        XCTAssertNil(MCPConnectionDetails(configuration:"{}"))
        XCTAssertNil(MCPConnectionDetails(configuration:"not JSON"))
    }
}
