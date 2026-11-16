import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MCPJobFailureDiagnosticTests:XCTestCase {
    func testTrainTypeFailureUsesExactAllowlistWithoutWorkerProse()throws {
        let known="Mapped integer values require explicit int64 columns; floating values require numeric columns."
        XCTAssertEqual(MCPJobFailureDiagnostic.execution(known)["code"],.string("mapped_column_types_required"))
        for message in [known+" /private/customer.csv", "Bearer secret", "", known+"\n"] {
            let result=MCPJobFailureDiagnostic.execution(message)
            XCTAssertEqual(result["code"],.string("execution_failed"))
            let wire=String(decoding:try JSONEncoder().encode(result),as:UTF8.self)
            XCTAssertFalse(wire.contains("Bearer"));XCTAssertFalse(wire.contains("/private"))
        }
    }
    func testAdmissionExposesOnlyFixedSaveDestinationGuidance()throws {
        for message in ["Choose Save model’s destination before running this branch.","Choose Save model’s destination before running."] {
            let result=MCPJobFailureDiagnostic.admission(message)
            XCTAssertEqual(result["code"],.string("model_destination_required"))
            XCTAssertEqual(result["isError"],.bool(true))
            XCTAssertNotNil(result["message"])
        }
        let secret="Choose Save model’s destination before running. /private/customer.csv"
        let unknown=MCPJobFailureDiagnostic.admission(secret)
        XCTAssertEqual(unknown["code"],.string("run_refused"))
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(unknown),as:UTF8.self).contains("/private"))
    }
    func testAPIStageDiagnosticsUseFixedCodesAndNeverEndpointOrWorkerText()throws {
        let graph=DataFlowDocument(nodes:[.init(id:"api",operatorID:"kwip.data.sourceCSV",config:[:])],edges:[])
        for stage in ["configuration","connection","snapshot","broker","verification","publication"] {
            let detail=DataFlowFailureDetail(schemaVersion:1,code:"api_source_"+stage+"_failed",nodeID:"api",nodeLabel:"PRIVATE_SENTINEL",operatorID:"kwip.data.sourceCSV",componentTitle:"PRIVATE_SENTINEL",inputRowCount:nil,cause:"https://private.example Bearer secret",suggestion:"PRIVATE_SENTINEL")
            let result=MCPJobFailureDiagnostic.project(detail,graph:graph)
            XCTAssertEqual(result["code"],.string(detail.code))
            let wire=String(decoding:try JSONEncoder().encode(result),as:UTF8.self)
            for secret in ["PRIVATE_SENTINEL","private.example","Bearer secret"] {XCTAssertFalse(wire.contains(secret))}
        }
    }
    func testOnlyKnownCodeAndVerifiedIdentityAreExposed()throws {
        let graph=DataFlowDocument(nodes:[.init(id:"formula",operatorID:"kwip.data.formula",config:[:])],edges:[])
        let secret="/private/data/customer.csv Bearer secret customer@example.com"
        let detail=DataFlowFailureDetail(schemaVersion:1,code:"expression_numeric_required",nodeID:"formula",nodeLabel:secret,operatorID:"kwip.data.formula",componentTitle:secret,inputRowCount:123,cause:secret,suggestion:secret)
        let result=MCPJobFailureDiagnostic.project(detail,graph:graph)
        XCTAssertEqual(result["code"],.string("expression_numeric_required"))
        let wire=String(decoding:try JSONEncoder().encode(result),as:UTF8.self)
        for privateValue in ["/private","Bearer","customer@example.com","123"] {XCTAssertFalse(wire.contains(privateValue))}
        let unknown=DataFlowFailureDetail(schemaVersion:1,code:secret,nodeID:"formula",nodeLabel:nil,operatorID:"kwip.data.formula",componentTitle:"Formula",inputRowCount:nil,cause:secret,suggestion:secret)
        XCTAssertEqual(MCPJobFailureDiagnostic.project(unknown,graph:graph)["code"],.string("execution_failed"))
        XCTAssertNil(MCPJobFailureDiagnostic.project(detail,graph:DataFlowDocument(nodes:[],edges:[]))["component"])
        XCTAssertNil(MCPJobFailureDiagnostic.project(nil,graph:graph)["component"])
    }
}
