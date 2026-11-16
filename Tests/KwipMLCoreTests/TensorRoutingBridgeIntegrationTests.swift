import XCTest
@testable import KwipMLCore

final class TensorRoutingBridgeIntegrationTests: XCTestCase {
    func testSpecializedBridgeUsesExactNamedShapesAndWinningIndexDtype() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in ["split", "chunk", "gather", "einsum"] {
            let graph = try ExampleFlows.tensorRoutingClassifier(componentID: id)
            let validation = BuiltInCapabilityBridge.validate(graph: graph, registry: registry, runtimeVersion: "2.14.1")
            XCTAssertTrue(validation.isValid, "\(id): \(validation.issues.map(\.message))")
        }
    }
}
