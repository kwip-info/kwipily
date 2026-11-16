import XCTest
@testable import KwipMLCore

final class ConvTranspose1dGraphTests: XCTestCase {
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "conv_transpose1d" }
        registry.components.append(ConvTranspose1dComponent.definition)
        return registry
    }
    private func config(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        Dictionary(uniqueKeysWithValues: ConvTranspose1dComponent.definition.fields.map { ($0.key, $0.defaultValue) })
            .merging(changes) { _, new in new }
    }
    private func flow() -> GraphDocument {
        let nodes: [GraphNode] = [.init(id: "input", type: "input", config: ["features": 2]),
            .init(id: "reshape", type: "reshape", config: ["shape": .array([2, 1])]),
            .init(id: "conv", type: "conv_transpose1d", config: config(["in_channels": 2, "out_channels": 2, "kernel_size": 1])),
            .init(id: "flat", type: "flatten", config: ["start_dim": 1, "end_dim": -1]), .init(id: "output", type: "output")]
        return .init(nodes: nodes, edges: zip(nodes, nodes.dropFirst()).map { .init(source: $0.id, target: $1.id) })
    }
    func testConvolutionAloneProvidesTrainableParametersWithSchemaTwo() throws {
        let registry = try registry(), graph = flow()
        let result = GraphValidator.validate(graph, registry: registry)
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["conv"]?.sample.dimensions, [2, 1])
        XCTAssertEqual(result.tensorContracts["conv"]?.dtype, .float32)
        XCTAssertEqual(result.widths["output"], 2)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: graph.nodes[2], schemaVersion: 1, registry: registry))
        XCTAssertTrue(ConvTranspose1dComponent.definition.fields.allSatisfy(\.isValidDefinition))
        XCTAssertEqual(Set(ConvTranspose1dComponent.definition.fields.map(\.key)), ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "dilation", "groups", "bias"])
    }
    func testGroupedChannelsFirstGraphUsesExactAnalyticalShape() throws {
        var graph = flow()
        graph.nodes[1].config["shape"] = .array([4, 6])
        graph.nodes[2].config = config(["in_channels": 4, "out_channels": 4, "groups": 2, "padding": 1, "stride": 2, "output_padding": 1])
        graph.nodes.append(.init(id: "projection", type: "linear", config: ["in_features": 2, "out_features": 24, "bias": true]))
        graph.nodes.append(.init(id: "head", type: "linear", config: ["in_features": 48, "out_features": 2, "bias": true]))
        graph.edges = [.init(source: "input", target: "projection"), .init(source: "projection", target: "reshape"),
            .init(source: "reshape", target: "conv"), .init(source: "conv", target: "flat"),
            .init(source: "flat", target: "head"), .init(source: "head", target: "output")]
        let result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["conv"]?.sample.dimensions, [4, 12])
        XCTAssertEqual(try ConvTranspose1dConfiguration(values: graph.nodes[2].config).parameterCount, 28)
    }
    func testInvalidRankChannelsDtypeAndPaddingCannotProduceContract() throws {
        for mutation in 0..<5 {
            var graph = flow()
            switch mutation {
            case 0: graph.nodes[1].config["shape"] = .array([2])
            case 1: graph.nodes[2].config["in_channels"] = 3
            case 2: graph.nodes[0] = .init(id: "input", type: "index_input", config: ["shape": .array([2])])
            case 3: graph.nodes[2].config["output_padding"] = 1
            default: graph.nodes[2].config["padding"] = 1
            }
            let result = GraphValidator.validate(graph, registry: try registry())
            XCTAssertFalse(result.isValid)
            XCTAssertNil(result.tensorContracts["conv"], "mutation \(mutation)")
        }
    }
    func testLargePositiveCropAndAggregateParameters() throws {
        var graph = flow()
        graph.nodes[1].config["shape"] = .array([1, 2])
        graph.nodes[2].config = config(["in_channels": 1, "out_channels": 2, "kernel_size": 1, "stride": 4, "padding": 2])
        XCTAssertTrue(GraphValidator.validate(graph, registry: try registry()).isValid)
        graph = flow()
        graph.nodes.removeAll { ["conv", "flat"].contains($0.id) }
        graph.nodes[1].config["shape"] = .array([512, 1])
        graph.nodes.append(.init(id: "projection", type: "linear", config: ["in_features": 2, "out_features": 512, "bias": true]))
        graph.edges = [.init(source: "input", target: "projection"), .init(source: "projection", target: "reshape")]
        var previous = "reshape"
        for index in 0..<8 {
            let id = "conv\(index)"
            graph.nodes.append(.init(id: id, type: "conv_transpose1d", config: config(["in_channels": 512, "out_channels": 512, "kernel_size": 1])))
            graph.edges.append(.init(source: previous, target: id)); previous = id
        }
        graph.nodes.append(.init(id: "flat", type: "flatten", config: ["start_dim": 1, "end_dim": -1]))
        graph.nodes.append(.init(id: "head", type: "linear", config: ["in_features": 512, "out_features": 2, "bias": true]))
        graph.edges += [.init(source: previous, target: "flat"), .init(source: "flat", target: "head"), .init(source: "head", target: "output")]
        let result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertEqual(result.widths["output"], 2)
        XCTAssertTrue(result.issues.contains { $0.message.contains("2 million") })
    }

}
