import XCTest
@testable import KwipMLCore

final class LayerNormCapabilityBridgeTests: XCTestCase {
    func testCanonicalDefinitionAndStatefulCapability() throws {
        let definition = LayerNormComponent.definition
        XCTAssertEqual(definition.id, "layer_norm")
        XCTAssertEqual(definition.category, "Normalization")
        XCTAssertEqual(definition.inputs, ["in"])
        XCTAssertEqual(definition.outputs, ["out"])
        XCTAssertTrue(definition.fields.allSatisfy(\.isValidDefinition))
        let defaults = Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })
        XCTAssertEqual(try LayerNormConfiguration(values: defaults).normalizedShape, [2])
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll {$0.id == "layer_norm"}; registry.components.append(definition)
        let package = try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1")
        let capability = try XCTUnwrap(package.capabilities.first { $0.id == "layer_norm" })
        XCTAssertEqual(capability.fields, definition.fields)
        XCTAssertEqual(capability.lifecycle.state.map(\.kind), [.parameters])
        XCTAssertEqual(capability.adapterID, "kwip.runner.graph-v2.layer_norm")
        XCTAssertEqual(capability.ports[0].schema.tensor?.leadingAxes, .init(minimum: 2, maximum: 8))
        let plans=try LayerNormLearningTask.allCases.map {try LayerNormAssessment.plan(task:$0,runtimeVersion:"2.14.1",seed:17)}
        XCTAssertEqual(Set(capability.learning.lessonIDs),Set(plans.map(\.id)))
        XCTAssertEqual(capability.learning.exampleIDs,["kwip.example.layer-norm-rows-classifier.v1","kwip.example.layer-norm-sample-classifier.v1"])
        XCTAssertTrue(capability.learning.conceptIDs.contains("linear-algebra.population-variance"))
        XCTAssertTrue(CapabilityContractValidator.validate(package).isValid)
    }
    func testConcreteGraphSpecializationAndMismatch() throws {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll {$0.id == "layer_norm"}; registry.components.append(LayerNormComponent.definition)
        let defaults = Dictionary(uniqueKeysWithValues: LayerNormComponent.definition.fields.map { ($0.key, $0.defaultValue) })
        let nodes = [GraphNode(id: "input", type: "input", config: ["features": 2]),
                     GraphNode(id: "linear", type: "linear", config: ["in_features": 2, "out_features": 2]),
                     GraphNode(id: "norm", type: "layer_norm", config: defaults), GraphNode(id: "output", type: "output")]
        var graph = GraphDocument(nodes: nodes, edges: zip(nodes, nodes.dropFirst()).map { GraphEdge(source: $0.id, target: $1.id) })
        let result = BuiltInCapabilityBridge.validate(graph: graph, registry: registry, runtimeVersion: "2.14.1")
        XCTAssertTrue(result.isValid, "\(result.issues)")
        graph.nodes[2].config["normalized_shape"] = .array([3])
        XCTAssertFalse(BuiltInCapabilityBridge.validate(graph: graph, registry: registry, runtimeVersion: "2.14.1").isValid)
    }
    func testAlteredNormalizationContractCannotClaimSupport() throws {
        for key in ["eps", "normalized_shape", "elementwise_affine", "bias"] {
            var definition = LayerNormComponent.definition
            let index = try XCTUnwrap(definition.fields.firstIndex { $0.key == key })
            definition.fields[index].defaultValue = key == "eps" ? .number(0.1) : .null
            var registry = try ComponentRegistry.builtIn(); registry.components.removeAll {$0.id == "layer_norm"}; registry.components.append(definition)
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry: registry), key)
        }
    }
}
