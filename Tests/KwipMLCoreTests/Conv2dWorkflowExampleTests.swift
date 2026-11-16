import XCTest
@testable import KwipMLCore

final class Conv2dWorkflowExampleTests: XCTestCase {
    func testProjectedClassifierShapesGroupsAndBothBiasModesCompile() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "conv2d"), Conv2dComponent.definition)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for bias in [false, true] {
            let graph = ExampleFlows.conv2dClassifier(bias: bias)
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertTrue(result.isValid, "\(result.issues)")
            XCTAssertEqual(result.tensorLayouts["reshape"]?.sample.dimensions, [2, 4, 5])
            XCTAssertEqual(result.tensorLayouts["convolution"]?.sample.dimensions, [4, 4, 5])
            XCTAssertEqual(result.tensorLayouts["activation"]?.sample.dimensions, [4, 4, 5])
            XCTAssertEqual(result.tensorLayouts["flatten"]?.sample.dimensions, [80])
            XCTAssertEqual(result.tensorLayouts["classifier"]?.sample.dimensions, [2])
            let convolution = try XCTUnwrap(graph.nodes.first { $0.type == "conv2d" })
            let config = try Conv2dConfiguration(values: convolution.config)
            XCTAssertEqual(config.parameterCount, bias ? 28 : 24)
            XCTAssertEqual(convolution.config["groups"], 2)
            XCTAssertEqual(convolution.config["padding_policy"], .string("same"))
            XCTAssertEqual(convolution.config["padding_mode"], .string("zeros"))
            XCTAssertEqual(convolution.config["bias"], .bool(bias))
            XCTAssertTrue(graph.nodes.contains { $0.type == "relu" })
            let workflow = try WorkflowExamples.preparedConv2dClassifier(device: "cpu", bias: bias)
            let plan = try WorkflowCompiler.compile(workflow, train: WorkflowExamples.trainAddress, modelRegistry: registry,
                dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.settings.device, "cpu")
            XCTAssertEqual(plan.model.graph.nodes.first { $0.type == "conv2d" }?.config, convolution.config)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)), workflow)
        }
    }

    func testPreparedLayoutKeepsCardsSeparateAndConnectionsForward() throws {
        let document = try PreparedWorkflowExample.conv2d.document()
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
