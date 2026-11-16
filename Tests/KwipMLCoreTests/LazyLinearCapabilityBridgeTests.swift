import XCTest
@testable import KwipMLCore

final class LazyLinearCapabilityBridgeTests: XCTestCase {
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "lazy_linear" }
        registry.components.append(LazyLinearComponent.definition)
        return registry
    }
    func testReviewedDefinitionAndInferredInputContract() throws {
        let definition = LazyLinearComponent.definition
        XCTAssertEqual(definition.id, "lazy_linear")
        XCTAssertEqual(definition.title, "Lazy Linear")
        XCTAssertEqual(definition.category, "Layers")
        XCTAssertEqual(definition.fields.map(\.key), ["out_features", "bias"])
        XCTAssertEqual(definition.fields.map(\.defaultValue), [2, true])
        XCTAssertTrue(definition.fields.allSatisfy(\.isValidDefinition))
        let package = try BuiltInCapabilityBridge.package(registry: registry(), runtimeVersion: "2.14.1")
        let value = try XCTUnwrap(package.capabilities.first { $0.id == "lazy_linear" })
        XCTAssertEqual(value.adapterID, "kwip.runner.graph-v2.lazy-linear")
        XCTAssertEqual(value.ports.map(\.id), ["in", "out"])
        XCTAssertEqual(value.ports[0].schema.tensor?.dimensions, [.symbol("inferredFeatures")])
        XCTAssertEqual(value.ports[1].schema.tensor?.dimensions, [.parameter("out_features")])
        for port in value.ports {
            XCTAssertEqual(port.schema.tensor?.dtype, .float32)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.minimum, 1)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.maximum, 7)
        }
        XCTAssertEqual(value.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        XCTAssertEqual(value.lifecycle.phases, [.initialize, .forward, .export])
        XCTAssertEqual(value.learning.lessonIDs, ["assessment.lazy-linear-inferred-width"])
        XCTAssertEqual(value.learning.exampleIDs, ["prepared-lazy-linear-classifier"])
    }
    func testMutatedFieldsOrPortsCannotAdvertiseReviewedAdapter() throws {
        let changes: [(inout ComponentDefinition) -> Void] = [
            { $0.inputs = ["a"] }, { $0.outputs = ["result"] },
            { $0.fields.removeLast() }, { $0.fields.append($0.fields[0]) },
            { $0.fields[0].defaultValue = 3 }, { $0.fields[0].max = 4096 },
            { $0.fields[0].key = "in_features" }, { $0.fields[1].kind = .integer }
        ]
        for change in changes {
            var registry = try registry()
            change(&registry.components[registry.components.count - 1])
            XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1"))
        }
    }
}
