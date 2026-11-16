import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataFlowFailureTests: XCTestCase {
    private let graph = DataFlowDocument(nodes: [DataFlowNode(id: "filter", operatorID: "kwip.data.filter")])
    private var event: [String: Any] { ["schemaVersion": 1, "code": "emptyFilterResult", "nodeID": "filter", "nodeLabel": "Retain quality", "operatorID": "kwip.data.filter", "componentTitle": "Filter rows", "inputRowCount": 48, "cause": "No rows remain", "suggestion": "Inspect the input and literal."] }
    private func decode(_ object: [String: Any]) throws -> DataFlowFailureDetail {
        try DataFlowFailureDetail.decode(JSONSerialization.data(withJSONObject: object), graph: graph, labels: ["filter": "Retain quality"], titles: ["kwip.data.filter": "Filter rows"])
    }
    func testExactCapturedNodeAndBoundedMetadataRequired() throws {
        let detail = try decode(event)
        XCTAssertEqual(detail.nodeID, "filter"); XCTAssertEqual(detail.inputRowCount, 48)
        for (key, value) in [("schemaVersion", 2 as Any), ("nodeID", "other"), ("operatorID", "kwip.data.cast"), ("nodeLabel", "Wrong label"), ("componentTitle", "Invented title"), ("inputRowCount", -1), ("inputRowCount", true), ("cause", String(repeating: "x", count: 501))] {
            var changed = event; changed[key] = value
            XCTAssertThrowsError(try decode(changed), key)
        }
    }
    func testNullOptionalFieldsAndOversizedPayload() throws {
        var detail = event; detail["nodeLabel"] = NSNull(); detail["inputRowCount"] = NSNull()
        XCTAssertThrowsError(try decode(detail), "Missing supplied label is not the captured context.")
        let bytes = try JSONSerialization.data(withJSONObject: detail)
        let decoded = try DataFlowFailureDetail.decode(bytes, graph: graph, labels: [:], titles: ["kwip.data.filter": "Filter rows"])
        XCTAssertNil(decoded.nodeLabel); XCTAssertNil(decoded.inputRowCount)
        XCTAssertThrowsError(try DataFlowFailureDetail.decode(Data(repeating: 32, count: 8193), graph: graph, labels: [:], titles: [:]))
    }
}
