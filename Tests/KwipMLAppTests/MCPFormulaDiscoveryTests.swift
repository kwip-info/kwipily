import XCTest
import KwipMLCore
@testable import KwipMLApp

extension MCPWorkflowGatewayTests {
    func testFormulaCatalogPublishesSupportedTextCompositionWithoutInventedConcat()async throws {
        let f=try fixture()
        let response=try await call(f,.catalogGet,["componentID":.string("kwip.data.formula")])
        guard case .array(let components)=response["components"],case .object(let item)=components.first,case .object(let schema)=item["configurationSchema"] else{return XCTFail("Missing Formula schema")}
        let encoded=String(decoding:try JSONEncoder().encode(schema),as:UTF8.self)
        XCTAssertTrue(encoded.contains("numeric-only"))
        XCTAssertTrue(encoded.contains("is_finite(number)"))
        XCTAssertTrue(encoded.contains("is_integer(number)"))
        XCTAssertTrue(encoded.contains("cannot recover lost source precision"))
        XCTAssertTrue(encoded.contains("castFailure null"))
        XCTAssertTrue(encoded.contains("regex_replace(text, pattern, replacement, occurrence, flags)"))
        XCTAssertTrue(encoded.contains("column-to-column"))
        guard case .array(let examples)=schema["examples"],!examples.isEmpty else{return XCTFail("Missing executable configuration example")}
    }
}
