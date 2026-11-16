import XCTest
@testable import KwipMLCore

final class LayerNormGraphTests: XCTestCase {
    func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll {$0.id == "layer_norm"}; registry.components.append(LayerNormComponent.definition)
        return registry
    }
    func normalization(_ shape: [JSONValue]) -> GraphNode {
        GraphNode(id: "norm", type: "layer_norm", config: ["normalized_shape": .array(shape), "eps": .number(1e-5), "elementwise_affine": true, "bias": true])
    }
    func graph(_ nodes: [GraphNode]) -> GraphDocument {
        GraphDocument(nodes: nodes, edges: zip(nodes, nodes.dropFirst()).map { GraphEdge(source: $0.id, target: $1.id) })
    }
    func testTrailingShapePropagationAndInvalidDescendants() throws {
        let registry = try registry()
        var flow = graph([
            GraphNode(id: "input", type: "input", config: ["features": 2]),
            GraphNode(id: "expand", type: "linear", config: ["in_features": 2, "out_features": 6]),
            GraphNode(id: "reshape", type: "reshape", config: ["shape": .array([2, 3])]),
            normalization([2, 3]),
            GraphNode(id: "flatten", type: "flatten", config: ["start_dim": 1, "end_dim": -1]),
            GraphNode(id: "head", type: "linear", config: ["in_features": 6, "out_features": 2]),
            GraphNode(id: "output", type: "output")
        ])
        let valid = GraphValidator.validate(flow, registry: registry)
        XCTAssertTrue(valid.isValid, "\(valid.issues)")
        XCTAssertEqual(valid.tensorLayouts["norm"], try BatchedTensorLayout(sample: [2, 3]))
        XCTAssertEqual(valid.widths["output"], 2)
        for shape: [JSONValue] in [[2], [3, 2], [1, 2, 3]] {
            flow.nodes[3].config["normalized_shape"] = .array(shape)
            let invalid = GraphValidator.validate(flow, registry: registry)
            XCTAssertFalse(invalid.isValid)
            XCTAssertTrue(invalid.issues.contains { $0.nodeID == "norm" })
            XCTAssertNil(invalid.tensorLayouts["norm"])
            XCTAssertNil(invalid.tensorLayouts["output"])
        }
    }
    func testCanonicalConfigurationVersionAndIdentity() throws {
        let registry = try registry()
        var node = normalization([.number(2), .number(3)])
        let canonical = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry)
        XCTAssertEqual(canonical["normalized_shape"], .array([.integer(2), .integer(3)]))
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 1, registry: registry))
        node.config = canonical
        let original = graph([node])
        let identity = try GraphSemantics.canonicalData(original, registry: registry)
        let restored = try JSONDecoder().decode(GraphDocument.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(try GraphSemantics.canonicalData(restored, registry: registry), identity)
        for (key, value): (String, JSONValue) in [("eps", .number(1e-4)), ("normalized_shape", .array([3])), ("elementwise_affine", false), ("bias", false)] {
            var changed = original; changed.nodes[0].config[key] = value
            XCTAssertNotEqual(try GraphSemantics.canonicalData(changed, registry: registry), identity)
        }
    }
    func testNormalizationParametersCountTowardModelBudget() throws {
        let registry = try registry()
        for width in [309, 310] {
            var nodes = [GraphNode(id: "input", type: "input", config: ["features": 2]), GraphNode(id: "expand", type: "linear", config: ["in_features": 2, "out_features": 512])]
            for index in 0..<7 { nodes.append(GraphNode(id: "wide\(index)", type: "linear", config: ["in_features": 512, "out_features": 512])) }
            nodes += [GraphNode(id: "reduce", type: "linear", config: ["in_features": 512, "out_features": .integer(Int64(width))]), normalization([.integer(Int64(width))]), GraphNode(id: "head", type: "linear", config: ["in_features": .integer(Int64(width)), "out_features": 2]), GraphNode(id: "output", type: "output")]
            var flow = graph(nodes)
            let result = GraphValidator.validate(flow, registry: registry)
            XCTAssertEqual(result.isValid, width == 309, "\(result.issues)")
            if width == 310 { XCTAssertTrue(result.issues.contains { $0.message.contains("2 million") }) }
            flow.nodes[flow.nodes.count - 3].config["elementwise_affine"] = false
            XCTAssertTrue(GraphValidator.validate(flow, registry: registry).isValid)
        }
    }
}
