import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CanvasNodeFeedbackTests: XCTestCase {
    func testExecutionEvidenceIsSpecificToActiveTerminalAndPhase() {
        for phase in [WorkflowExecutionPhase.idle, .preparing, .completed, .failed, .cancelled] {
            XCTAssertFalse(CanvasNodeFeedback.isExecuting(phase: phase, matchesTerminal: true, isCurrent: true))
        }
        XCTAssertTrue(CanvasNodeFeedback.isExecuting(phase: .training, matchesTerminal: true, isCurrent: true))
        XCTAssertFalse(CanvasNodeFeedback.isExecuting(phase: .training, matchesTerminal: false, isCurrent: true), "Busy neighbors must not look executing")
        XCTAssertFalse(CanvasNodeFeedback.isExecuting(phase: .training, matchesTerminal: true, isCurrent: false), "Retired or edited requests are not live evidence")
    }

    func testOnlyNodeOwnedBlockingIssuesProduceMisconfigurationBadge() {
        let issues = [ValidationIssue(id: "one", message: "Choose an input", nodeID: "a"),
                      ValidationIssue(id: "two", message: "Dimension mismatch", nodeID: "a"),
                      ValidationIssue(id: "three", message: "Missing field", nodeID: "b"),
                      ValidationIssue(id: "global", message: "Cannot validate the document")]
        let feedback = CanvasNodeFeedback(nodeID: "a", issues: issues, isExecuting: false)
        XCTAssertTrue(feedback.isMisconfigured)
        XCTAssertEqual(feedback.issueMessages, ["Choose an input", "Dimension mismatch"])
        XCTAssertEqual(feedback.accessibilityDescription, "Configuration needs attention. Choose an input. Dimension mismatch")
        XCTAssertFalse(CanvasNodeFeedback(nodeID: "c", issues: issues, isExecuting: false).isMisconfigured)
    }

    func testReducedMotionRetainsExecutingIndicatorWithoutAnimation() {
        let feedback = CanvasNodeFeedback(nodeID: "a", issues: [], isExecuting: true)
        XCTAssertTrue(feedback.shouldAnimate(reduceMotion: false))
        XCTAssertFalse(feedback.shouldAnimate(reduceMotion: true))
        XCTAssertEqual(feedback.accessibilityDescription, "Executing this component's operation")
        XCTAssertFalse(CanvasNodeFeedback(nodeID: "a", issues: [], isExecuting: false).shouldAnimate(reduceMotion: false))
    }

    func testIssuesAndExecutionRemainDistinctEvidence() {
        let feedback = CanvasNodeFeedback(nodeID: "a", issues: [.init(message: "Connect input", nodeID: "a")], isExecuting: true)
        XCTAssertTrue(feedback.isExecuting)
        XCTAssertTrue(feedback.isMisconfigured)
        XCTAssertTrue(feedback.accessibilityDescription.contains("Configuration needs attention"))
        XCTAssertTrue(feedback.accessibilityDescription.contains("Executing"))
    }
    @MainActor func testUniversalOrphanRequiredInputWarnsAndConnectedModelClearsWithoutMarkingUnusedSources() throws {
        let input = WorkflowNodeAddress(domain: .model, localID: "input")
        let activation = WorkflowNodeAddress(domain: .model, localID: "relu")
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: input, payload: .modelPrimitive(type: "input", configurationVersion: 1, config: ["features": 2])),
            .init(address: activation, payload: .modelPrimitive(type: "relu", configurationVersion: 1, config: [:]))])
        let (root, adapter) = try feedbackAdapter(document)
        defer { try? FileManager.default.removeItem(at: root) }
        let inputID = try XCTUnwrap(adapter.projection.displayIDByAddress[input])
        let reluID = try XCTUnwrap(adapter.projection.displayIDByAddress[activation])
        XCTAssertFalse(adapter.canvasNodeFeedback(for: inputID).isMisconfigured, "Unused fully configured sources are legitimate drafts")
        XCTAssertTrue(adapter.canvasNodeFeedback(for: reluID).isMisconfigured, "A newly dropped ReLU needs its required tensor input")
        XCTAssertTrue(adapter.canvasNodeFeedback(for: reluID).issueMessages.contains { $0.contains("in") })
        let checks=adapter.feedbackConnectionValidationCount
        for _ in 0..<100 {_=adapter.canvasNodeFeedback(for:reluID)}
        XCTAssertEqual(adapter.feedbackConnectionValidationCount,checks,"Layout-only reads reuse the current projection's checks")
        adapter.connect(source: inputID, sourcePort: "out", target: reluID, targetPort: "in")
        XCTAssertFalse(adapter.canvasNodeFeedback(for: reluID).isMisconfigured)
        XCTAssertGreaterThan(adapter.feedbackConnectionValidationCount,checks,"A connection edit must invalidate cached warnings")
    }

    @MainActor func testUniversalOptionalInputAndRequiredOutputDoNotWarnButTooManyInputsDo() throws {
        let source = WorkflowNodeAddress(domain: .data, localID: "source")
        let target = WorkflowNodeAddress(domain: .data, localID: "target")
        var data = try feedbackDataRegistry()
        let index = try XCTUnwrap(data.operators.firstIndex { $0.id == "kwip.data.columns" })
        data.operators[index].inputs[0].required = false
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: source, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator": .string(",")])),
            .init(address: target, payload: .dataOperator(operatorID: "kwip.data.columns", config: ColumnsConfigurationDraft.defaults))])
        let (root, adapter) = try feedbackAdapter(document, dataRegistry: data)
        defer { try? FileManager.default.removeItem(at: root) }
        for node in adapter.graph.nodes { XCTAssertFalse(adapter.canvasNodeFeedback(for: node.id).isMisconfigured) }
        var overconnected = document
        let port = data.operators[index].inputs[0].id
        overconnected.edges = (0..<2).map { WorkflowEdge(id: "edge-\($0)", from: .init(address: source, port: "table"), to: .init(address: target, port: port), kind: .table) }
        XCTAssertThrowsError(try WorkflowCodec.encode(overconnected), "Canonical source storage already rejects duplicate input occupancy")
        let descriptor = try XCTUnwrap(adapter.projection.portDescriptor(address: target, port: port, direction: .input)?.contract)
        XCTAssertTrue(CanvasNodeFeedback.connectionIssueMessages(port: descriptor, inboundCount: 2).contains { $0.contains("at most 1") })
    }

    @MainActor func testConnectedUniversalControlInputsHaveNoConnectionWarnings() throws {
        let document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        let (root, adapter) = try feedbackAdapter(document)
        defer { try? FileManager.default.removeItem(at: root) }
        for node in document.nodes where node.address.domain == .control {
            let display = try XCTUnwrap(adapter.projection.displayIDByAddress[node.address])
            XCTAssertFalse(adapter.canvasNodeFeedback(for: display).isMisconfigured, node.name)
        }
    }

    private func feedbackDataRegistry() throws -> DataFlowOperatorRegistry {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
    }
    @MainActor private func feedbackAdapter(_ document: WorkflowDocument, dataRegistry: DataFlowOperatorRegistry? = nil) throws -> (URL, WorkflowCanvasAdapter) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = ProjectLibrary(directory: root)
        _ = try library.saveSource(.workflow(document))
        let models = try ComponentRegistry.builtIn(), data = try dataRegistry ?? feedbackDataRegistry()
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: models, dataRegistry: data)
        return (root, try WorkflowCanvasAdapter(owner: owner, modelRegistry: models, dataRegistry: data))
    }

}
