import XCTest
@testable import KwipMLCore

final class LazyLinearExampleTests: XCTestCase {
    func testExampleInfersTwoDifferentWidthsPreservesPrefixAndCompilesBothBiasModes() throws {
        let registry = try ComponentRegistry.builtIn()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for bias in [false, true] {
            let graph = ExampleFlows.lazyLinearClassifier(bias: bias)
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertTrue(result.isValid, "\(result.issues)")
            XCTAssertEqual(result.tensorLayouts["reshape"]?.sample.dimensions, [2, 3])
            XCTAssertEqual(result.tensorLayouts["local"]?.sample.dimensions, [2, 4])
            XCTAssertEqual(result.tensorLayouts["flatten"]?.sample.dimensions, [8])
            XCTAssertEqual(result.tensorLayouts["classifier"]?.sample.dimensions, [2])
            let layers = graph.nodes.filter { $0.type == "lazy_linear" }
            XCTAssertEqual(layers.count, 2)
            XCTAssertTrue(layers.allSatisfy { Set($0.config.keys) == ["out_features", "bias"] && $0.config["bias"] == .bool(bias) })
            let workflow = try WorkflowExamples.preparedLazyLinearClassifier(device: "cpu", bias: bias)
            let plan = try WorkflowCompiler.compile(workflow, train: WorkflowExamples.trainAddress, modelRegistry: registry,
                dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(plan.settings.device, "cpu")
            XCTAssertEqual(plan.model.graph.nodes.filter { $0.type == "lazy_linear" }.map(\.config), layers.map(\.config))
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(workflow)), workflow)
        }
    }
    func testBundledResourceAndCompactPreparedLayout() throws {
        XCTAssertEqual(PreparedWorkflowExample.lazyLinear.observationsResource, "workflow-observations")
        let document = try PreparedWorkflowExample.lazyLinear.document()
        XCTAssertEqual(document.nodes.filter {
            if case .modelPrimitive(type: "lazy_linear", configurationVersion: _, config: _) = $0.payload { return true }
            return false
        }.count, 2)
        for (index, node) in document.nodes.enumerated() {
            for other in document.nodes.dropFirst(index + 1) {
                XCTAssertTrue(abs(node.x-other.x) >= 200 || abs(node.y-other.y) >= 130, "Overlapping cards: \(node.name), \(other.name)")
            }
        }
        let nodes = Dictionary(uniqueKeysWithValues: document.nodes.map { ($0.address, $0) })
        for edge in document.edges { XCTAssertLessThan(try XCTUnwrap(nodes[edge.from.address]).x, try XCTUnwrap(nodes[edge.to.address]).x) }
    }
}
