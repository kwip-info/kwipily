import XCTest
@testable import KwipMLCore

final class AdaptivePoolingGraphTests: XCTestCase {
    func testPreparedExamplesResolveTypedOutputsAndOwnDeclarations() throws {
        let registry = try ComponentRegistry.builtIn()
        let declarations = try BuiltInCapabilityBridge.declarations(registry: registry)
        for id in ["avg_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d"] {
            let values: [String:JSONValue]
            if id == "avg_pool2d" {
                values = ["kernel_size": .array([3,3]), "stride": .array([2,3]), "padding": .array([1,1]), "ceil_mode": true, "count_include_pad": true, "divisor_override": -2]
            } else if id == "adaptive_avg_pool1d" { values = ["output_size": 3] }
            else { values = ["output_height": 3, "output_width": .null].merging(id == "adaptive_max_pool2d" ? ["return_indices": true] : [:]) { _, new in new } }
            var graph = try ExampleFlows.adaptivePoolingClassifier(componentID: id, configuration: values)
            let result = GraphValidator.validate(graph, registry: registry)
            XCTAssertTrue(result.isValid, "\(id): \(result.issues)")
            let capability = try XCTUnwrap(declarations.first { $0.id == id })
            XCTAssertEqual(capability.lifecycle.state, [])
            XCTAssertEqual(capability.fields, registry.components.first { $0.id == id }?.fields)
            XCTAssertEqual(result.outputTensorContracts[.init("pool", "out")]?.dtype, .float32)
            if id == "adaptive_max_pool2d" {
                XCTAssertEqual(result.outputTensorContracts[.init("pool", "indices")]?.dtype, .int64)
                let pool = try XCTUnwrap(graph.nodes.firstIndex { $0.id == "pool" })
                graph.nodes[pool].config["return_indices"] = false
                let edge = try XCTUnwrap(graph.edges.firstIndex { $0.source == "pool" })
                graph.edges[edge].sourcePort = "indices"
                XCTAssertFalse(GraphValidator.validate(graph, registry: registry).isValid)
            }
            _ = try BundledPoolingComponents.example(id, device: "cpu")
        }
    }
    func testDivisorZeroRefusedBeforeExecutionAndNullableFieldEnablesAtOne() throws {
        let registry = try ComponentRegistry.builtIn()
        let definition = try XCTUnwrap(registry.components.first { $0.id == "avg_pool2d" })
        let divisor = try XCTUnwrap(definition.fields.first { $0.key == "divisor_override" })
        XCTAssertEqual(divisor.valueWhenEnabled, 1)
        var config = Dictionary(uniqueKeysWithValues: definition.fields.map { ($0.key, $0.defaultValue) })
        config["divisor_override"] = 0
        let node = GraphNode(id: "pool", type: "avg_pool2d", name: "Area summary", x: 0, y: 0, config: config)
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry))
    }
}
