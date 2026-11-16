import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPortInspectionTests: XCTestCase {
    private func setup() throws -> (URL, WorkflowAuthoringStore, WorkflowCanvasAdapter) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = ProjectLibrary(directory: root)
        let data = WorkflowNodeAddress(domain: .data, localID: "same")
        let model = WorkflowNodeAddress(domain: .model, localID: "same")
        let linear = WorkflowNodeAddress(domain: .model, localID: "linear")
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: data, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:])),
            .init(address: model, payload: .modelPrimitive(type: "input", configurationVersion: 1, config: ["features": 2])),
            .init(address: linear, payload: .modelPrimitive(type: "linear", configurationVersion: 2, config: ["in_features": 2, "out_features": 2, "bias": true]))
        ], edges: [.init(id: "model-link", from: .init(address: model, port: "out"), to: .init(address: linear, port: "in"), kind: .tensor)])
        _ = try library.saveSource(.workflow(document))
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let modelRegistry = try ComponentRegistry.builtIn()
        let dataRegistry = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: modelRegistry, dataRegistry: dataRegistry)
        return (root, owner, try WorkflowCanvasAdapter(owner: owner, modelRegistry: modelRegistry, dataRegistry: dataRegistry))
    }
    func testInputResolvesOnlyExplicitQualifiedSourceAndDoesNotEditDocument() throws {
        let (root, owner, adapter) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        let address = WorkflowNodeAddress(domain: .model, localID: "linear")
        let id = try XCTUnwrap(adapter.projection.displayIDByAddress[address])
        let original = owner.document
        XCTAssertTrue(adapter.inspectPort(nodeID: id, port: "in", direction: .input))
        let inspection = try XCTUnwrap(adapter.currentPortInspection)
        XCTAssertEqual(inspection.endpoint.address, address)
        XCTAssertEqual(adapter.inspectedSourceEndpoints, [.init(address: .init(domain: .model, localID: "same"), port: "out")])
        XCTAssertEqual(owner.document, original)
        XCTAssertFalse(adapter.inspectPort(nodeID: id, port: "in", direction: .output))
        XCTAssertEqual(adapter.currentPortInspection, inspection)
        let selection = owner.selection
        adapter.dismissPortInspection()
        XCTAssertNil(adapter.currentPortInspection)
        XCTAssertEqual(owner.selection, selection)
        XCTAssertEqual(owner.document, original)
    }
    func testReadOnlyInspectionWorksWhileRunningAndRejectsInactiveOwner() throws {
        let (root, owner, adapter) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        let address = WorkflowNodeAddress(domain: .data, localID: "same")
        let id = try XCTUnwrap(adapter.projection.displayIDByAddress[address])
        owner.setExecutionBusy(true)
        XCTAssertTrue(adapter.inspectPort(nodeID: id, port: "table", direction: .output))
        XCTAssertEqual(adapter.inspectedSourceEndpoints, [.init(address: address, port: "table")])
        XCTAssertEqual(owner.selection.nodeAddresses, [])
        adapter.invalidate()
        XCTAssertNil(adapter.currentPortInspection)
        XCTAssertFalse(adapter.inspectPort(nodeID: id, port: "table", direction: .output))
    }
    func testOrdinarySelectionClearsExplicitPortAndUnlinkedInputInfersNoSource() throws {
        let (root, _, adapter) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        let id = try XCTUnwrap(adapter.projection.displayIDByAddress[.init(domain: .model, localID: "linear")])
        XCTAssertTrue(adapter.inspectPort(nodeID: id, port: "in", direction: .input))
        adapter.selectNode(id, extending: false)
        XCTAssertNil(adapter.currentPortInspection)
        XCTAssertTrue(adapter.inspectPort(nodeID: id, port: "out", direction: .output))
        XCTAssertEqual(adapter.inspectedSourceEndpoints.count, 1)
        XCTAssertFalse(adapter.inspectPort(nodeID: "unknown", port: "out", direction: .output))
    }
}
