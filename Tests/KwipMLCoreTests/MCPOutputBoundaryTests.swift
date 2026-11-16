import XCTest
@testable import KwipMLCore
final class MCPOutputBoundaryTests:XCTestCase {
    func testObservationBudgetsAreIndependentOfLargeDiscoverySchemas() {
        XCTAssertEqual(MCPOutputBoundary.responseByteLimit(action:"results_read"),32_768)
        XCTAssertEqual(MCPOutputBoundary.responseByteLimit(action:"job_get"),32_768)
        XCTAssertEqual(MCPOutputBoundary.responseByteLimit(action:"tools_list"),300_000)
        XCTAssertLessThan(MCPOutputBoundary.responseByteLimit(action:"workflow_inspect"),100_000)
    }
    func testExplicitlySharedValuesDoNotExposeCredentialsOrLocalPaths() {
        let source:JSONValue = .object(["password":.string("do-not-return"),"sourcePath":.string("/Users/owner/private.csv"),"rows":.array([.array([.integer(7),.string("ordinary captured text"),.string("file:///Users/owner/private.csv")])])])
        XCTAssertEqual(MCPOutputBoundary.sanitize(source),.object(["rows":.array([.array([.integer(7),.string("ordinary captured text"),.string("[local path]")])])]))
    }
    func testObservationCapsDoNotMutateCapturedValue() {
        let rows=JSONValue.array((0..<1001).map{.integer(Int64($0))})
        guard case .array(let bounded)=MCPOutputBoundary.sanitize(rows,truncateValues:true)else{return XCTFail()}
        XCTAssertEqual(bounded.count,1000)
        guard case .array(let original)=rows else{return XCTFail()};XCTAssertEqual(original.count,1001)
    }
}
