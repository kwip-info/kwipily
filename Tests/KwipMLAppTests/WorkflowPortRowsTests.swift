import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPortRowsTests: XCTestCase {
    private let selected = WorkflowNodeAddress(domain: .data, localID: "selected")
    private let upstream = WorkflowNodeAddress(domain: .data, localID: "upstream")
    private let downstream = WorkflowNodeAddress(domain: .data, localID: "downstream")
    func testFanOutUsesOneRowPerConnectionWithoutRepeatingConnectedPort() {
        let source = WorkflowEndpoint(address: selected, port: "table")
        let edges = [
            WorkflowEdge(id: "b", from: source, to: .init(address: downstream, port: "right"), kind: .table),
            WorkflowEdge(id: "a", from: source, to: .init(address: downstream, port: "left"), kind: .table)
        ]
        let rows = WorkflowPortRow.project(address: selected, inputs: ["source"], outputs: ["table", "report"], edges: edges)
        XCTAssertEqual(rows.compactMap(\.edge).map(\.id), ["a", "b"])
        XCTAssertEqual(rows.filter { $0.edge == nil }.map(\.endpoint.port), ["source", "report"])
        XCTAssertEqual(rows.count, 4)
        XCTAssertEqual(Set(rows.map(\.id)).count, rows.count)
    }
    func testIncomingConnectionInspectsLocalInputAndPreservesMissingRepairEndpoint() {
        let edge = WorkflowEdge(id: "repair", from: .init(address: upstream, port: "old-output"), to: .init(address: selected, port: "old-input"), kind: .table)
        let unrelated = WorkflowEdge(id: "elsewhere", from: .init(address: upstream, port: "out"), to: .init(address: downstream, port: "in"), kind: .table)
        let rows = WorkflowPortRow.project(address: selected, inputs: ["in"], outputs: [], edges: [edge, unrelated])
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows.first?.endpoint, edge.to)
        XCTAssertEqual(rows.first?.direction, .input)
        XCTAssertEqual(rows.first?.edge, edge)
        XCTAssertEqual(rows.last?.endpoint.port, "in")
    }
    func testEmptyDeclarationDoesNotInventOptionalInputAndDirectionsStayDistinct() {
        XCTAssertTrue(WorkflowPortRow.project(address: selected, inputs: [], outputs: [], edges: []).isEmpty)
        let rows = WorkflowPortRow.project(address: selected, inputs: ["data"], outputs: ["data"], edges: [])
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(Set(rows.map(\.id)).count, 2)
        XCTAssertEqual(rows.map(\.direction), [.input, .output])
    }
    func testSelfConnectionAppearsOnlyOnce() {
        let edge = WorkflowEdge(id: "loop", from: .init(address: selected, port: "out"), to: .init(address: selected, port: "in"), kind: .table)
        let rows = WorkflowPortRow.project(address: selected, inputs: ["in"], outputs: ["out"], edges: [edge])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].edge, edge)
    }
}
