import XCTest
@testable import KwipMLCore

final class BilinearGraphTests: XCTestCase {
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "bilinear" }
        registry.components.append(.init(id: "bilinear", title: "Bilinear", category: "Layers", inputs: ["a", "b"], outputs: ["out"], fields:
            ["in1_features", "in2_features", "out_features"].map { .init(key: $0, title: $0, kind: .integer, defaultValue: 2, min: 1, max: 512) }
            + [.init(key: "bias", title: "Bias", kind: .boolean, defaultValue: true, min: nil, max: nil)]))
        return registry
    }
    private func bilinear() -> GraphNode {
        .init(id: "interaction", type: "bilinear", config: ["in1_features": 2, "in2_features": 2, "out_features": 2, "bias": true])
    }
    private func flow() -> GraphDocument {
        .init(nodes: [.init(id: "input", type: "input", config: ["features": 2]), bilinear(), .init(id: "output", type: "output")], edges: [
            .init(source: "input", target: "interaction", targetPort: "a"),
            .init(source: "input", target: "interaction", targetPort: "b"),
            .init(source: "interaction", target: "output")])
    }
    func testBilinearAloneIsTrainableAndSchemaTwoConfigurationIsRequired() throws {
        let registry = try registry(), graph = flow()
        let result = GraphValidator.validate(graph, registry: registry)
        XCTAssertTrue(result.isValid, "\(result.issues)")
        XCTAssertEqual(result.widths["output"], 2)
        XCTAssertEqual(result.tensorContracts["interaction"]?.dtype, .float32)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: bilinear(), schemaVersion: 1, registry: registry))
        var excessive = bilinear(); excessive.config = ["in1_features": 512, "in2_features": 512, "out_features": 512, "bias": true]
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: excessive, schemaVersion: 2, registry: registry))
    }
    func testInvalidEitherWidthOrIntegerOperandHasNoOutputContract() throws {
        let registry = try registry()
        for key in ["in1_features", "in2_features"] {
            var graph = flow(); graph.nodes[1].config[key] = 3
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertFalse(result.isValid); XCTAssertNil(result.tensorContracts["interaction"])
        }
        var graph = flow(); graph.nodes[0] = .init(id: "input", type: "index_input", config: ["shape": .array([2])])
        let result = GraphValidator.validate(graph, registry: registry)
        XCTAssertFalse(result.isValid); XCTAssertNil(result.tensorContracts["interaction"])
    }
    func testTrainabilityDoesNotCountFrozenOrNonAffineState() throws {
        let registry = try registry()
        let input = GraphNode(id: "input", type: "input", config: ["features": 2])
        for affine in [false, true] {
            let norm = GraphNode(id: "norm", type: "layer_norm", config: ["normalized_shape": .array([2]), "eps": .number(1e-5), "elementwise_affine": .bool(affine), "bias": true])
            let graph = GraphDocument(nodes: [input, norm, .init(id: "output", type: "output")], edges: [.init(source: "input", target: "norm"), .init(source: "norm", target: "output")])
            XCTAssertEqual(GraphValidator.validate(graph, registry: registry).isValid, affine)
        }
        for frozen in [false, true] {
            let definition = try XCTUnwrap(registry.component(type: "embedding"))
            var config = Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })
            config["num_embeddings"] = 3; config["embedding_dim"] = 2; config["freeze"] = .bool(frozen)
            let graph = GraphDocument(nodes: [.init(id: "input", type: "index_input", config: ["shape": .array([1])]),
                .init(id: "lookup", type: "embedding", config: config),
                .init(id: "flat", type: "flatten", config: ["start_dim": 1, "end_dim": -1]), .init(id: "output", type: "output")],
                edges: [.init(source: "input", target: "lookup"), .init(source: "lookup", target: "flat"), .init(source: "flat", target: "output")])
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertEqual(result.isValid, !frozen, "\(result.issues)")
        }
    }
    func testConnectionCommandsCheckEachWidthAndKnownSiblingPrefix() throws {
        let registry = try registry()
        var graph = flow(); graph.edges = []
        graph.nodes[1].config["in1_features"] = 3
        let command = GraphCommand.connect(source: .init("input", "out"), target: .init("interaction", "a"))
        XCTAssertThrowsError(try GraphCommandEngine.apply(command, to: .init(document: graph), registry: registry))
        let second = GraphCommand.connect(source: .init("input", "out"), target: .init("interaction", "b"))
        XCTAssertNoThrow(try GraphCommandEngine.apply(second, to: .init(document: graph), registry: registry))
        graph.nodes[1].config["in1_features"] = 2
        graph.nodes.append(.init(id: "shaped", type: "reshape", config: ["shape": .array([1,2])]))
        graph.edges = [.init(source: "input", target: "shaped"), .init(source: "shaped", target: "interaction", targetPort: "a")]
        XCTAssertThrowsError(try GraphCommandEngine.apply(second, to: .init(document: graph), registry: registry))
        XCTAssertEqual(graph.edges.count, 2)
    }

}
