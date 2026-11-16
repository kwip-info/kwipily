import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PoolingExecutionWarningTests: XCTestCase {
    private func bytes(_ changes: [String:Any] = [:]) throws -> Data {
        let record: [String:Any] = ["event":"warning", "schemaVersion":1, "runID":"11111111-1111-4111-8111-111111111111", "nodeID":"pool", "code":"pooling.cpuFallback", "computeDevice":"cpu", "returnDevice":"mps", "message":"Nondivisible adaptive average bins use CPU; values and gradients return to MPS."]
        return try JSONSerialization.data(withJSONObject: record.merging(changes) { _,new in new })
    }
    func testReviewedWarningLinksToSelectedAdaptiveAverageNode() throws {
        let graph = GraphDocument(nodes: [.init(id:"pool", type:"adaptive_avg_pool2d")])
        let warning = try XCTUnwrap(PoolingExecutionWarning.decode(bytes(), graph: graph))
        XCTAssertEqual(warning.nodeID, "pool")
        XCTAssertTrue(warning.message.contains("CPU"))
        XCTAssertTrue(warning.message.contains("MPS"))
    }
    func testUnrelatedMalformedAndUnboundedWarningsAreNotPresented() throws {
        let graph = GraphDocument(nodes: [.init(id:"pool", type:"adaptive_avg_pool2d")])
        for changes: [String:Any] in [["schemaVersion":true], ["code":"other"], ["nodeID":"missing"], ["runID":"invalid"], ["computeDevice":"mps"], ["message":String(repeating:"x", count:5000)], ["extra":"unexpected"]] {
            XCTAssertNil(PoolingExecutionWarning.decode(try bytes(changes), graph: graph))
        }
        XCTAssertNil(PoolingExecutionWarning.decode(try bytes(), graph: GraphDocument(nodes:[.init(id:"pool", type:"adaptive_max_pool2d")])))
    }
}
