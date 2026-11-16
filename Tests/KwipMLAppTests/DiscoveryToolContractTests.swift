import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DiscoveryToolContractTests: XCTestCase {
    func testCapabilityDiscoveryHasNoInputsAndNoArbitraryArguments() throws {
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: nil)
        let request = WorkflowToolRequest(operationID: UUID().uuidString, projectID: UUID().uuidString,
            requestIdentity: String(repeating: "a", count: 64), toolMode: .runtimeCapabilities,
            arguments: ["--capabilities"], inputFiles: [], executionBudget: policy, remainingSeconds: 3600)
        XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()), request)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        for arguments in [["--capabilities", "--registry", "/tmp/alternate.json"], ["--graph", "/tmp/graph.json"], ["--dataflow-registry"], []] {
            var bad = raw; bad["arguments"] = arguments
            XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject: bad)))
        }
        var bad = raw
        bad["inputFiles"] = [["path": "input/graph.json", "bytes": 0, "sha256": String(repeating: "b", count: 64)]]
        XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject: bad)))
    }
}
