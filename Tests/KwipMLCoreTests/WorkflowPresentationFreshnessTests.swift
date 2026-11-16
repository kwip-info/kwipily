import XCTest
@testable import KwipMLCore

final class WorkflowPresentationFreshnessTests: XCTestCase {
    func testLayoutLabelsAndRevisionDoNotInvalidateCapturedComputation() {
        let a = WorkflowNodeAddress(domain: .data, localID: "view")
        let original = WorkflowDocument(projectID: "fixture", nodes: [.init(address: a, name: "View", x: 10, y: 20, payload: .dataOperator(operatorID: "kwip.data.view", config: [:]))])
        var moved = original
        moved.nodes[0].x = 100; moved.nodes[0].y = -20
        moved.nodes[0].name = "Review"; moved.name = "Renamed"; moved.revision = 10
        XCTAssertTrue(moved.matchesCapturedConstruction(original))
        XCTAssertTrue(original.matchesCapturedConstruction(moved))
        moved.nodes[0].payload = .dataOperator(operatorID: "kwip.data.view", config: ["changed": .bool(true)])
        XCTAssertFalse(moved.matchesCapturedConstruction(original))
        moved = original; moved.projectID = "other"
        XCTAssertFalse(moved.matchesCapturedConstruction(original))
        moved = original; moved.nodes[0].address.localID = "other"
        XCTAssertFalse(moved.matchesCapturedConstruction(original))
        moved = original; moved.edges = [.init(from: .init(address: a, port: "table"), to: .init(address: a, port: "table"), kind: .table)]
        XCTAssertFalse(moved.matchesCapturedConstruction(original))
    }
}
