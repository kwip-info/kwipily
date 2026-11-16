import XCTest
@testable import KwipMLCore

final class Conv2dGraphTests: XCTestCase {
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "conv2d" }
        registry.components.append(Conv2dComponent.definition)
        return registry
    }
    private func config(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        Dictionary(uniqueKeysWithValues: Conv2dComponent.definition.fields.map { ($0.key, $0.defaultValue) })
            .merging(changes) { _, new in new }
    }
    private func flow() -> GraphDocument {
        let nodes: [GraphNode] = [.init(id: "input", type: "input", config: ["features": 2]),
            .init(id: "reshape", type: "reshape", config: ["shape": .array([2, 1, 1])]),
            .init(id: "conv", type: "conv2d", config: config(["in_channels": 2, "out_channels": 2, "kernel_size": .array([1, 1])])),
            .init(id: "flat", type: "flatten", config: ["start_dim": 1, "end_dim": -1]), .init(id: "output", type: "output")]
        return .init(nodes: nodes, edges: zip(nodes, nodes.dropFirst()).map { .init(source: $0.id, target: $1.id) })
    }
    func testConvolutionAloneProvidesTrainableParametersWithSchemaTwo() throws {
        let registry = try registry(), graph = flow()
        let result = GraphValidator.validate(graph, registry: registry)
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["conv"]?.sample.dimensions, [2, 1, 1])
        XCTAssertEqual(result.tensorContracts["conv"]?.dtype, .float32)
        XCTAssertEqual(result.widths["output"], 2)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: graph.nodes[2], schemaVersion: 1, registry: registry))
        XCTAssertTrue(Conv2dComponent.definition.fields.allSatisfy(\.isValidDefinition))
        XCTAssertEqual(Conv2dComponent.definition.fields.first { $0.key == "padding_policy" }?.kind, .choice)
        XCTAssertEqual(Conv2dComponent.definition.fields.first { $0.key == "padding_mode" }?.kind, .choice)
        for key in ["kernel_size", "stride", "dilation", "padding"] {
            let field = try XCTUnwrap(Conv2dComponent.definition.fields.first { $0.key == key })
            XCTAssertEqual(field.kind, .integerList)
            XCTAssertEqual(field.minItems, 2); XCTAssertEqual(field.maxItems, 2)
        }
    }
    func testGroupedChannelsFirstGraphUsesExactAnalyticalShape() throws {
        var graph = flow()
        graph.nodes[1].config["shape"] = .array([4, 3, 5])
        graph.nodes[2].config = config(["in_channels": 4, "out_channels": 4, "groups": 2, "kernel_size": .array([1, 3]), "padding": .array([0, 1]), "stride": .array([1, 2])])
        graph.nodes.append(.init(id: "projection", type: "linear", config: ["in_features": 2, "out_features": 60, "bias": true]))
        graph.nodes.append(.init(id: "head", type: "linear", config: ["in_features": 36, "out_features": 2, "bias": true]))
        graph.edges = [.init(source: "input", target: "projection"), .init(source: "projection", target: "reshape"),
            .init(source: "reshape", target: "conv"), .init(source: "conv", target: "flat"),
            .init(source: "flat", target: "head"), .init(source: "head", target: "output")]
        let result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["conv"]?.sample.dimensions, [4, 3, 3])
        XCTAssertEqual(try Conv2dConfiguration(values: graph.nodes[2].config).parameterCount, 28)
    }
    func testInvalidRankChannelsDtypeAndPaddingCannotProduceContract() throws {
        for mutation in 0..<5 {
            var graph = flow()
            switch mutation {
            case 0: graph.nodes[1].config["shape"] = .array([2])
            case 1: graph.nodes[2].config["in_channels"] = 3
            case 2: graph.nodes[0] = .init(id: "input", type: "index_input", config: ["shape": .array([2])])
            case 3: graph.nodes[2].config["padding_policy"] = .string("same"); graph.nodes[2].config["stride"] = .array([1, 2])
            default: graph.nodes[2].config["padding_mode"] = .string("reflect"); graph.nodes[2].config["padding"] = .array([0, 1])
            }
            let result = GraphValidator.validate(graph, registry: try registry())
            XCTAssertFalse(result.isValid)
            XCTAssertNil(result.tensorContracts["conv"], "mutation \(mutation)")
        }
    }
}
