import XCTest
@testable import KwipMLCore

final class BilinearExampleTests: XCTestCase {
    func testExampleUsesIndependentLearnedBranchesAndCompilesPreparedMapping() throws {
        let registry = try ComponentRegistry.builtIn(), graph = ExampleFlows.bilinearClassifier
        let result = GraphValidator.validate(graph, registry: registry)
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.widths["left"], 3); XCTAssertEqual(result.widths["right"], 5)
        XCTAssertEqual(result.widths["interaction"], 2)
        let layer = try XCTUnwrap(graph.nodes.first { $0.type == "bilinear" })
        XCTAssertEqual(layer.config, ["in1_features": 3, "in2_features": 5, "out_features": 2, "bias": true])
        XCTAssertEqual(Set(graph.edges.filter { $0.target == layer.id }.map(\.targetPort)), ["a", "b"])
        let workflow = try WorkflowExamples.preparedBilinearClassifier(device: "cpu")
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let plan = try WorkflowCompiler.compile(workflow, train: WorkflowExamples.trainAddress, modelRegistry: registry,
            dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.model.graph.nodes.filter { $0.type == "bilinear" }.count, 1)
        XCTAssertEqual(plan.settings.device, "cpu")
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)), workflow)
        XCTAssertEqual(PreparedWorkflowExample.bilinear.observationsResource, "workflow-observations")
        XCTAssertEqual(try PreparedWorkflowExample.bilinear.document().nodes.filter { if case .modelPrimitive(type: "bilinear", configurationVersion: _, config: _) = $0.payload { return true }; return false }.count, 1)
    }
    func testPreparedLayoutKeepsCardsSeparatedAndEdgesForward() throws {
        let document = try WorkflowExamples.preparedBilinearClassifier()
        let nodes = Dictionary(uniqueKeysWithValues: document.nodes.map { ($0.address, $0) })
        for (index, node) in document.nodes.enumerated() {
            for other in document.nodes.dropFirst(index + 1) {
                XCTAssertTrue(abs(node.x-other.x) >= 200 || abs(node.y-other.y) >= 130, "Overlapping cards: \(node.name), \(other.name)")
            }
        }
        for edge in document.edges {
            XCTAssertLessThan(try XCTUnwrap(nodes[edge.from.address]).x, try XCTUnwrap(nodes[edge.to.address]).x)
        }
        let original = try WorkflowExamples.preparedClassifier(projectID: document.projectID, device: "auto", model: ExampleFlows.bilinearClassifier)
        XCTAssertEqual(document.edges, original.edges)
        XCTAssertEqual(document.nodes.map(\.payload), original.nodes.map(\.payload))
    }
}
