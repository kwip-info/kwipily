import XCTest
@testable import KwipMLCore

final class ConvTranspose2dGraphTests: XCTestCase {
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "conv_transpose2d" }
        registry.components.append(ConvTranspose2dComponent.definition)
        return registry
    }
    private func config(_ changes: [String: JSONValue] = [:]) -> [String: JSONValue] {
        Dictionary(uniqueKeysWithValues: ConvTranspose2dComponent.definition.fields.map { ($0.key, $0.defaultValue) })
            .merging(changes) { _, new in new }
    }
    private func flow() -> GraphDocument {
        let nodes: [GraphNode] = [.init(id: "input", type: "input", config: ["features": 2]),
            .init(id: "reshape", type: "reshape", config: ["shape": .array([2, 1, 1])]),
            .init(id: "conv", type: "conv_transpose2d", config: config(["in_channels": 2, "out_channels": 2, "kernel_size": .array([1, 1])])),
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
        XCTAssertTrue(ConvTranspose2dComponent.definition.fields.allSatisfy(\.isValidDefinition))
        for key in ["kernel_size", "stride", "padding", "output_padding", "dilation"] {
            let field = try XCTUnwrap(ConvTranspose2dComponent.definition.fields.first { $0.key == key })
            XCTAssertEqual(field.kind, .integerList)
            XCTAssertEqual(field.minItems, 2); XCTAssertEqual(field.maxItems, 2)
        }
        XCTAssertEqual(Set(ConvTranspose2dComponent.definition.fields.map(\.key)), ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "dilation", "groups", "bias"])
    }
    func testGroupedChannelsFirstGraphUsesExactAnalyticalShape() throws {
        var graph = flow()
        graph.nodes[1].config["shape"] = .array([4, 3, 5])
        graph.nodes[2].config = config(["in_channels": 4, "out_channels": 4, "groups": 2, "padding": .array([1, 0]), "stride": .array([2, 3]), "output_padding": .array([1, 2]), "kernel_size": .array([3, 2])])
        graph.nodes.append(.init(id: "projection", type: "linear", config: ["in_features": 2, "out_features": 60, "bias": true]))
        graph.nodes.append(.init(id: "head", type: "linear", config: ["in_features": 384, "out_features": 2, "bias": true]))
        graph.edges = [.init(source: "input", target: "projection"), .init(source: "projection", target: "reshape"),
            .init(source: "reshape", target: "conv"), .init(source: "conv", target: "flat"),
            .init(source: "flat", target: "head"), .init(source: "head", target: "output")]
        let result = GraphValidator.validate(graph, registry: try registry())
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.tensorLayouts["conv"]?.sample.dimensions, [4, 6, 16])
        XCTAssertEqual(try ConvTranspose2dConfiguration(values: graph.nodes[2].config).parameterCount, 52)
    }
    func testInvalidRankChannelsDtypeAndPaddingCannotProduceContract() throws {
        for mutation in 0..<6 {
            var graph = flow()
            switch mutation {
            case 0: graph.nodes[1].config["shape"] = .array([2])
            case 1: graph.nodes[2].config["in_channels"] = 3
            case 2: graph.nodes[0] = .init(id: "input", type: "index_input", config: ["shape": .array([2])])
            case 3: graph.nodes[2].config["output_padding"] = .array([1, 0])
            case 4: graph.nodes[2].config["padding"] = .array([1, 0])
            default: graph.nodes[2].config["kernel_size"] = 1
            }
            let result = GraphValidator.validate(graph, registry: try registry())
            XCTAssertFalse(result.isValid)
            XCTAssertNil(result.tensorContracts["conv"], "mutation \(mutation)")
        }
    }
    func testLargePositiveCropAndAggregateParameters() throws {
        var graph = flow()
        graph.nodes[1].config["shape"] = .array([1, 1, 2])
        graph.nodes[2].config = config(["in_channels": 1, "out_channels": 2, "kernel_size": .array([1, 1]), "stride": .array([4, 4]), "padding": .array([0, 2])])
        XCTAssertTrue(GraphValidator.validate(graph, registry: try registry()).isValid)
        graph = flow()
        graph.nodes.removeAll { ["conv", "flat"].contains($0.id) }
        graph.nodes[1].config["shape"] = .array([512, 1, 1])
        graph.nodes.append(.init(id: "projection", type: "linear", config: ["in_features": 2, "out_features": 512, "bias": true]))
        graph.edges = [.init(source: "input", target: "projection"), .init(source: "projection", target: "reshape")]
        var previous = "reshape"
        for index in 0..<8 {
            let id = "conv\(index)"
            graph.nodes.append(.init(id: id, type: "conv_transpose2d", config: config(["in_channels": 512, "out_channels": 512, "kernel_size": .array([1, 1])])))
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
