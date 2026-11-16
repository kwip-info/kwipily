import XCTest
@testable import KwipMLCore

final class ConvTranspose3dWorkflowExampleTests: XCTestCase {
    func testProjectedClassifierShapesGroupsAndBothBiasModesCompile() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "conv_transpose3d"), ConvTranspose3dComponent.definition)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for bias in [false, true] {
            let graph = ExampleFlows.convTranspose3dClassifier(bias: bias)
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertTrue(result.isValid, "\(result.issues)")
            XCTAssertEqual(result.tensorLayouts["reshape"]?.sample.dimensions, [2, 2, 3, 4])
            XCTAssertEqual(result.tensorLayouts["convolution"]?.sample.dimensions, [4, 3, 5, 7])
            XCTAssertEqual(result.tensorLayouts["activation"]?.sample.dimensions, [4, 3, 5, 7])
            XCTAssertEqual(result.tensorLayouts["flatten"]?.sample.dimensions, [420])
            XCTAssertEqual(result.tensorLayouts["classifier"]?.sample.dimensions, [2])
            let convolution = try XCTUnwrap(graph.nodes.first { $0.type == "conv_transpose3d" })
            let config = try ConvTranspose3dConfiguration(values: convolution.config)
            XCTAssertEqual(config.parameterCount, bias ? 112 : 108)
            XCTAssertEqual(convolution.config["groups"], 2)
            XCTAssertEqual(convolution.config["stride"], .array([2, 2, 2]))
            XCTAssertEqual(convolution.config["output_padding"], .array([0, 0, 0]))
            XCTAssertEqual(convolution.config["bias"], .bool(bias))
            XCTAssertTrue(graph.nodes.contains { $0.type == "relu" })
            let workflow = try WorkflowExamples.preparedConvTranspose3dClassifier(device: "cpu", bias: bias)
            let plan = try WorkflowCompiler.compile(workflow, train: WorkflowExamples.trainAddress, modelRegistry: registry,
                dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.settings.device, "cpu")
            XCTAssertEqual(plan.model.graph.nodes.first { $0.type == "conv_transpose3d" }?.config, convolution.config)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)), workflow)
        }
    }

    func testPreparedLayoutKeepsCardsSeparateAndConnectionsForward() throws {
        let document = try PreparedWorkflowExample.convTranspose3d.document()
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
