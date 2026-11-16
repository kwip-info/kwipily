import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

final class MCPWorkflowValidationDiagnosticTests:XCTestCase {
    func testFiniteReasonsNeverExposeCompilerPayloadAndVerifyIdentityAndPort()throws {
        let address=WorkflowNodeAddress(domain:.control,localID:"fit")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,payload:.estimator(config:[:]))],edges:[])
        let secret="/private/customer.csv Bearer-secret row-value@example.com"
        for code in [WorkflowCompilerError.Code.requiredInput,.configuration,.lineage,.invalidDocument] {
            var error=WorkflowCompilerError(code:code,message:secret,node:address,port:"training")
            error.modelIssues=[.init(address:address,message:secret)]
            let result=MCPWorkflowValidationDiagnostic.project(error,document:document,ports:{_ in ["training"]})
            XCTAssertEqual(result["reason"],.string(code.rawValue));XCTAssertEqual(result["port"],.string("training"));XCTAssertNotNil(result["component"])
            XCTAssertEqual(result["valid"],.bool(false));XCTAssertNil(result["jobID"])
            let wire=String(decoding:try JSONEncoder().encode(result),as:UTF8.self)
            for fragment in ["/private","Bearer-secret","row-value@example.com"] {XCTAssertFalse(wire.contains(fragment))}
        }
    }
    func testForgedStaleIdentityUnknownPortAndUnknownErrorAreNotEchoed()throws {
        let address=WorkflowNodeAddress(domain:.control,localID:"fit")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,payload:.estimator(config:[:]))],edges:[])
        let forged=WorkflowCompilerError(code:.requiredInput,message:"secret",node:.init(domain:.data,localID:"missing"),port:"training")
        XCTAssertNil(MCPWorkflowValidationDiagnostic.project(forged,document:document,ports:{_ in ["training"]})["component"])
        let unknownPort=WorkflowCompilerError(code:.requiredInput,message:"secret",node:address,port:"private-field")
        XCTAssertNil(MCPWorkflowValidationDiagnostic.project(unknownPort,document:document,ports:{_ in ["training"]})["port"])
        XCTAssertNil(MCPWorkflowValidationDiagnostic.project(unknownPort,document:document,ports:{_ in throw DataFlowContractError("secret")})["port"])
        XCTAssertNil(MCPWorkflowValidationDiagnostic.project(unknownPort,document:nil,ports:{_ in ["training"]})["reason"])
        let unknown=MCPWorkflowValidationDiagnostic.project(DataFlowContractError("/private/token"),document:document,ports:{_ in []})
        XCTAssertNil(unknown["reason"]);XCTAssertNil(unknown["component"])
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(unknown),as:UTF8.self).contains("/private"))
    }
}
