import XCTest
@testable import KwipMLCore

final class ConfigurableActivationConfigurationTests: XCTestCase {
    func testDefaultsSignedCoefficientsAndStorageRefusal() throws {
        XCTAssertEqual(try ConfigurableActivationConfiguration(componentID: "silu", values: [:]).values, [:])
        XCTAssertEqual(try ConfigurableActivationConfiguration(componentID: "leaky_relu", values: [:]).values, ["negative_slope": .number(0.01)])
        XCTAssertEqual(try ConfigurableActivationConfiguration(componentID: "elu", values: [:]).values, ["alpha": .number(1)])
        for (id, key) in [("leaky_relu", "negative_slope"), ("elu", "alpha")] {
            for value in [-2.0, 0, 0.2, 2] {
                XCTAssertEqual(try ConfigurableActivationConfiguration(componentID: id, values: [key: .number(value)]).values[key], .number(value))
            }
            for bad in [JSONValue.bool(true), .null, .string("1"), .number(.nan), .number(.infinity), .number(3.5e38)] {
                XCTAssertThrowsError(try ConfigurableActivationConfiguration(componentID: id, values: [key: bad]))
            }
        }
        for id in ["silu", "leaky_relu", "elu"] {
            XCTAssertThrowsError(try ConfigurableActivationConfiguration(componentID: id, values: ["inplace": .bool(true)]))
        }
        XCTAssertThrowsError(try ConfigurableActivationConfiguration(componentID: "unknown", values: [:]))
    }
    func testSchemaTwoShapePreservationAndFloatContract() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in ["silu", "leaky_relu", "elu"] {
            var graph = ExampleFlows.geluClassifier()
            for i in graph.nodes.indices where graph.nodes[i].type == "gelu" {
                graph.nodes[i].type = id
                graph.nodes[i].config = Dictionary(uniqueKeysWithValues: registry.component(type: id)!.fields.map { ($0.key, $0.defaultValue) })
            }
            XCTAssertTrue(GraphValidator.validate(graph, registry: registry).isValid)
            graph.schemaVersion = 1
            XCTAssertFalse(GraphValidator.validate(graph, registry: registry).isValid)
        }
    }
}
