import XCTest
@testable import KwipMLCore

final class LazyLinearGraphTests: XCTestCase {
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "lazy_linear" }
        registry.components.append(.init(id: "lazy_linear", title: "LazyLinear", category: "Layers", inputs: ["in"], outputs: ["out"], fields: [
            .init(key: "out_features", title: "Output features", kind: .integer, defaultValue: 2, min: 1, max: 512),
            .init(key: "bias", title: "Bias", kind: .boolean, defaultValue: true, min: nil, max: nil)]))
        return registry
    }
    private func flow() -> GraphDocument {
        .init(nodes: [.init(id: "input", type: "input", config: ["features": 2]),
            .init(id: "lazy", type: "lazy_linear", config: ["out_features": 2, "bias": true]), .init(id: "output", type: "output")],
            edges: [.init(source: "input", target: "lazy"), .init(source: "lazy", target: "output")])
    }
    func testLazyOnlyClassifierIsTrainableAndConfigurationStaysUnresolved() throws {
        let registry = try registry(), graph = flow()
        let result = GraphValidator.validate(graph, registry: registry)
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorContracts["lazy"]?.dtype, .float32)
        XCTAssertEqual(result.widths["output"], 2)
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of: graph.nodes[1], schemaVersion: 2, registry: registry), graph.nodes[1].config)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: graph.nodes[1], schemaVersion: 1, registry: registry))
    }
    func testNestedSamplePrefixesSurviveLazyProjectionAndFlatten() throws {
        var graph = flow()
        graph.nodes.append(.init(id: "shape", type: "reshape", config: ["shape": .array([1, 1, 2])]))
        graph.nodes.append(.init(id: "flat", type: "flatten", config: ["start_dim": 1, "end_dim": -1]))
        graph.edges = [.init(source: "input", target: "shape"), .init(source: "shape", target: "lazy"), .init(source: "lazy", target: "flat"), .init(source: "flat", target: "output")]
        let result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["lazy"]?.sample.dimensions, [1, 1, 2])
        XCTAssertEqual(result.tensorLayouts["output"]?.sample.dimensions, [2])
    }
    func testIntegerInputAndPerLayerParameterOverflowDoNotProduceContracts() throws {
        var graph = flow()
        graph.nodes[0] = .init(id: "input", type: "index_input", config: ["shape": .array([2])])
        var result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertFalse(result.isValid)
        XCTAssertNil(result.tensorContracts["lazy"])
        graph = flow()
        graph.nodes.append(.init(id: "projection", type: "linear", config: ["in_features": 2, "out_features": 512, "bias": true]))
        graph.edges = [.init(source: "input", target: "projection")]
        var previous = "projection"
        for index in 0..<3 {
            let id = "concat\(index)"
            graph.nodes.append(.init(id: id, type: "concat"))
            graph.edges += [.init(source: previous, target: id, targetPort: "a"), .init(source: previous, target: id, targetPort: "b")]
            previous = id
        }
        graph.edges += [.init(source: previous, target: "lazy"), .init(source: "lazy", target: "output")]
        graph.nodes[1].config["out_features"] = 489
        result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertEqual(result.tensorLayouts["concat2"]?.sample.dimensions, [4096])
        XCTAssertTrue(result.issues.contains { $0.nodeID == "lazy" && $0.message.contains("2 million") })
        XCTAssertNil(result.tensorContracts["lazy"])
    }
    func testAggregateParameterLimitCountsEveryLazyLayer() throws {
        var graph = flow()
        graph.nodes.remove(at: 1); graph.edges = []
        var previous = "input"
        for index in 0..<9 {
            let id = "lazy\(index)"
            graph.nodes.append(.init(id: id, type: "lazy_linear", config: ["out_features": 512, "bias": true]))
            graph.edges.append(.init(source: previous, target: id)); previous = id
        }
        graph.nodes.append(.init(id: "head", type: "lazy_linear", config: ["out_features": 2, "bias": false]))
        graph.edges += [.init(source: previous, target: "head"), .init(source: "head", target: "output")]
        let result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertEqual(result.widths["output"], 2)
        XCTAssertTrue(result.issues.contains { $0.message.contains("2 million") }, "\(result.issues)")
    }
}
