import XCTest
@testable import KwipMLCore

final class BilinearCapabilityBridgeTests: XCTestCase {
    func testSharedRegistryExposesReviewedAuthoringDefinition() throws {
        XCTAssertEqual(try ComponentRegistry.builtIn().component(type: "bilinear"), BilinearComponent.definition)
    }
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        registry.components.removeAll { $0.id == "bilinear" }
        registry.components.append(BilinearComponent.definition)
        return registry
    }
    func testReviewedIndependentPortWidthsAndLearnedStateWithDeliveredLessons() throws {
        let package = try BuiltInCapabilityBridge.package(registry: registry(), runtimeVersion: "2.14.1")
        let value = try XCTUnwrap(package.capabilities.first { $0.id == "bilinear" })
        XCTAssertEqual(value.adapterID, "kwip.runner.graph-v2.bilinear")
        XCTAssertEqual(value.ports.map(\.id), ["a", "b", "out"])
        XCTAssertEqual(value.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        XCTAssertEqual(value.learning.conceptIDs, ["linear-algebra.bilinear-contraction", "linear-algebra.parameter-tensor"])
        XCTAssertEqual(value.learning.lessonIDs, BilinearLearningTask.allCases.map { "assessment." + $0.rawValue }); XCTAssertEqual(value.learning.exampleIDs, ["prepared-bilinear-classifier"])
        for (port, width) in zip(value.ports, ["in1_features", "in2_features", "out_features"]) {
            let tensor = try XCTUnwrap(port.schema.tensor)
            XCTAssertEqual(tensor.dtype, .float32)
            XCTAssertEqual(tensor.dimensions, [.parameter(width)])
            XCTAssertEqual(tensor.leadingAxes?.minimum, 1); XCTAssertEqual(tensor.leadingAxes?.maximum, 7)
        }
        XCTAssertEqual(value.fields, BilinearComponent.definition.fields)
    }
    func testMutatedReviewedFieldsAndPortsCannotAdvertiseAdapter() throws {
        let changes: [(inout ComponentDefinition) -> Void] = [
            { $0.inputs = ["b", "a"] }, { $0.outputs = ["result"] },
            { $0.fields.removeLast() }, { $0.fields.append($0.fields[0]) },
            { $0.fields[0] = .init(key: "in1_features", title: "First input features", kind: .integer, defaultValue: 3, min: 1, max: 512) },
            { $0.fields[1] = .init(key: "in2_features", title: "Second input features", kind: .integer, defaultValue: 2, min: 1, max: 1024) }
        ]
        for change in changes {
            var registry = try registry()
            change(&registry.components[registry.components.count - 1])
            XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1"))
        }
    }
}
