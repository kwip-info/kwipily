import XCTest
@testable import KwipMLCore

final class ConvTranspose1dWorkflowExampleTests: XCTestCase {
    func testProjectedClassifierShapesGroupsAndBothBiasModesCompile() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "conv_transpose1d"), ConvTranspose1dComponent.definition)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for bias in [false, true] {
            let graph = ExampleFlows.convTranspose1dClassifier(bias: bias)
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertTrue(result.isValid, "\(result.issues)")
            XCTAssertEqual(result.tensorLayouts["reshape"]?.sample.dimensions, [2, 7])
            XCTAssertEqual(result.tensorLayouts["convolution"]?.sample.dimensions, [4, 14])
            XCTAssertEqual(result.tensorLayouts["activation"]?.sample.dimensions, [4, 14])
            XCTAssertEqual(result.tensorLayouts["flatten"]?.sample.dimensions, [56])
            XCTAssertEqual(result.tensorLayouts["classifier"]?.sample.dimensions, [2])
            let convolution = try XCTUnwrap(graph.nodes.first { $0.type == "conv_transpose1d" })
            let config = try ConvTranspose1dConfiguration(values: convolution.config)
            XCTAssertEqual(config.parameterCount, bias ? 16 : 12)
            XCTAssertEqual(convolution.config["groups"], 2)
            XCTAssertEqual(convolution.config["stride"], 2)
            XCTAssertEqual(convolution.config["output_padding"], 1)
            XCTAssertEqual(convolution.config["bias"], .bool(bias))
            XCTAssertTrue(graph.nodes.contains { $0.type == "relu" })
            let workflow = try WorkflowExamples.preparedConvTranspose1dClassifier(device: "cpu", bias: bias)
            let plan = try WorkflowCompiler.compile(workflow, train: WorkflowExamples.trainAddress, modelRegistry: registry,
                dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.settings.device, "cpu")
            XCTAssertEqual(plan.model.graph.nodes.first { $0.type == "conv_transpose1d" }?.config, convolution.config)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)), workflow)
        }
    }

    func testPreparedLayoutKeepsCardsSeparateAndConnectionsForward() throws {
        let document = try PreparedWorkflowExample.convTranspose1d.document()
        XCTAssertNil(document.guidedBuild)
        for (index, node) in document.nodes.enumerated() {
            for other in document.nodes.dropFirst(index + 1) {
                XCTAssertTrue(abs(node.x-other.x) >= 200 || abs(node.y-other.y) >= 130,
                    "Overlapping cards: \(node.name), \(other.name)")
            }
        }
        let nodes = Dictionary(uniqueKeysWithValues: document.nodes.map { ($0.address, $0) })
        for edge in document.edges {
            XCTAssertLessThan(try XCTUnwrap(nodes[edge.from.address]).x, try XCTUnwrap(nodes[edge.to.address]).x)
        }
    }
}
