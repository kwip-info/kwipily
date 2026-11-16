import XCTest
@testable import KwipMLCore

final class RemainingDataLineageTests: XCTestCase {
    private let source = WorkflowNodeAddress(domain: .data, localID: "source")
    private let tool = WorkflowNodeAddress(domain: .data, localID: "tool")
    private let split = WorkflowNodeAddress(domain: .data, localID: "partition")
    private var registry: DataFlowOperatorRegistry {
        get throws {
            let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            return try .load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        }
    }
    private func resolve(_ id: String, config: [String: JSONValue], partitioned: Bool) throws -> [WorkflowEndpoint: WorkflowDataLineage.Evidence] {
        var nodes: [WorkflowNode] = [.init(address: source, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults))]
        var edges: [WorkflowEdge] = []
        if partitioned {
            nodes.append(.init(address: split, payload: .dataOperator(operatorID: "kwip.data.partition", config: ["strategy":.string("random"), "seed": .integer(42), "validationPercent": .integer(25), "testPercent":.integer(25)])))
            edges.append(.init(from: .init(address: source, port: "table"), to: .init(address: split, port: "table"), kind: .table))
        }
        nodes.append(.init(address: tool, payload: .dataOperator(operatorID: id, config: config)))
        edges.append(.init(from: .init(address: partitioned ? split : source, port: partitioned ? "training" : "table"), to: .init(address: tool, port: "table"), kind: .table))
        return try WorkflowDataLineage.resolve(.init(nodes: Dictionary(uniqueKeysWithValues: nodes.map { ($0.address, $0) }), incoming: Dictionary(grouping: edges, by: { $0.to.address }), selected: Set(nodes.map(\.address)), order: nodes.map(\.address)), dataRegistry: registry)
    }
    func testSplitColumnsPreservesPartitionAndRowsRefusesIt() throws {
        var config = SplitTextPracticeExample.configuration
        let evidence = try resolve("kwip.data.splitText", config: config, partitioned: true)
        guard case .rows(let rows) = evidence[.init(address: tool, port: "table")] else { return XCTFail("Missing rows") }
        XCTAssertEqual(rows.source, source); XCTAssertEqual(rows.split, split); XCTAssertEqual(rows.role, "training")
        config["outputMode"] = .string("rows")
        XCTAssertThrowsError(try resolve("kwip.data.splitText", config: config, partitioned: true))
        let expanded = try resolve("kwip.data.splitText", config: config, partitioned: false)
        guard case .rows(let newRows) = expanded[.init(address: tool, port: "table")] else { return XCTFail("Missing expansion") }
        XCTAssertEqual(newRows.source, tool); XCTAssertNil(newRows.split); XCTAssertEqual(newRows.role, "unassigned")
    }
    func testFixedWidthPreservesAndReshapeChangesPopulation() throws {
        var config = SplitTextPracticeExample.configuration; config["mode"] = .string("fixedWidth")
        XCTAssertNoThrow(try resolve("kwip.data.splitText", config: config, partitioned: true))
        let definition = try XCTUnwrap(registry.operatorDefinition(id: "kwip.data.reshapeTable"))
        XCTAssertThrowsError(try resolve("kwip.data.reshapeTable", config: definition.defaultConfig, partitioned: true))
        let result = try resolve("kwip.data.reshapeTable", config: definition.defaultConfig, partitioned: false)
        guard case .rows(let rows) = result[.init(address: tool, port: "table")] else { return XCTFail("Missing reshape rows") }
        XCTAssertEqual(rows.source, tool)
    }
    func testSortAndBothUniquePortsPreserveLineage() throws {
        for id in ["kwip.data.sort", "kwip.data.unique"] {
            let definition = try XCTUnwrap(registry.operatorDefinition(id: id))
            var config = definition.defaultConfig
            config["keys"] = id.hasSuffix("unique") ? .array([.string("id")]) : .array([.object(["column": .string("id"), "direction": .string("ascending"), "missing": .string("last"), "comparison": .string("exact")])])
            let result = try resolve(id, config: config, partitioned: true)
            for port in id.hasSuffix("unique") ? ["unique", "duplicates"] : ["table"] {
                guard case .rows(let rows) = result[.init(address: tool, port: port)] else { return XCTFail("Missing \(port)") }
                XCTAssertEqual(rows.source, source); XCTAssertEqual(rows.split, split); XCTAssertEqual(rows.role, "training")
            }
        }
    }
    func testPreparationOnlyRefusesRowsAfterPartition() throws {
        var config = SplitTextPracticeExample.configuration; config["outputMode"] = .string("rows")
        let document = WorkflowDocument(projectID: "11111111-2222-3333-4444-555555555555", name: "Population guard", nodes: [
            .init(address: source, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: CSVParsingConfiguration.newDefaults)),
            .init(address: split, payload: .dataOperator(operatorID: "kwip.data.partition", config: ["strategy":.string("random"), "seed": .integer(42), "validationPercent": .integer(25), "testPercent":.integer(25)])),
            .init(address: tool, payload: .dataOperator(operatorID: "kwip.data.splitText", config: config))
        ], edges: [
            .init(from: .init(address: source, port: "table"), to: .init(address: split, port: "table"), kind: .table),
            .init(from: .init(address: split, port: "training"), to: .init(address: tool, port: "table"), kind: .table)
        ])
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(document, registry: registry)) { error in
            XCTAssertTrue(error.localizedDescription.contains("before Partition"), error.localizedDescription)
        }
    }
}
