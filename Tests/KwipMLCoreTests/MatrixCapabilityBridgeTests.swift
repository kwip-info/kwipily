import XCTest
@testable import KwipMLCore

final class MatrixCapabilityBridgeTests:XCTestCase {
    func testReviewedMatrixPortsAndLearningContract()throws {
        var registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type:"matmul"),MatrixMultiplyComponent.definition)
        registry.components.removeAll {$0.id == "matmul"}
        registry.components.append(MatrixMultiplyComponent.definition)
        let package=try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.14.1")
        let matrix=try XCTUnwrap(package.capabilities.first {$0.id == "matmul"})
        XCTAssertEqual(matrix.adapterID,"kwip.runner.graph-v2.matrixMultiply")
        XCTAssertEqual(matrix.ports.map(\.id),["a","b","out"])
        XCTAssertTrue(matrix.lifecycle.state.isEmpty)
        XCTAssertEqual(Set(matrix.learning.lessonIDs),Set(MatrixLearningTask.allCases.map { "assessment." + $0.rawValue }))
        for port in matrix.ports {
            XCTAssertEqual(port.schema.tensor?.dtype,.float32)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.minimum,3)
            XCTAssertEqual(port.schema.tensor?.leadingAxes?.maximum,8)
        }
        registry.components[registry.components.count-1].inputs=["in"]
        XCTAssertThrowsError(try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.14.1"))
    }
}
