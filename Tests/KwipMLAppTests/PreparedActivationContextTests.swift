import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparedActivationContextTests: XCTestCase {
    private func fixture() -> (GraphDocument, [String: JSONValue]) {
        let graph = GraphDocument(schemaVersion: 2, nodes: [
            .init(id: "input", type: "input", config: ["features": 3]),
            .init(id: "head", type: "linear", config: ["in_features": 3, "out_features": 2, "bias": true]),
            .init(id: "output", type: "output")
        ], edges: [.init(id: "a", source: "input", target: "head"), .init(id: "b", source: "head", target: "output")])
        func tensor(_ width: Int64) -> JSONValue {
            .object(["shape": .array([8, .integer(width)]), "dtype": .string("float32"), "values": .array([0])])
        }
        return (graph, ["input": tensor(3), "head": tensor(2), "output": tensor(2)])
    }
    func testCapturedPreparedRegistryAcceptsThreeFeatureModel() throws {
        let (graph, previews) = fixture()
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph,
            sampleCapture: nil, capturedRegistry: ComponentRegistry.builtIn()))
    }
    func testPreparedContextStillRejectsIncompatibleLayerWidth() throws {
        var (graph, previews) = fixture()
        graph.nodes[1].config["in_features"] = 4
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews, graph: graph,
            sampleCapture: nil, capturedRegistry: ComponentRegistry.builtIn()))
    }
}
